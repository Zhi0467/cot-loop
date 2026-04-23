from __future__ import annotations

import faulthandler
import json
import math
import multiprocessing as mp
import os
import platform
import signal
import subprocess
import sys
import tempfile
from enum import Enum
from typing import Any

from ._common import (
    collect_row_metadata,
    load_rows_from_dataset,
    resolve_sample_id,
    row_matches_filter,
)
from ..prompt_format import format_user_prompt
from ..types import DatasetSpec, SampleRecord

_GLOBAL_TIMEOUT_SEC = 10
_PER_TEST_TIMEOUT_SEC = 4
_CB_IMPORTS = """import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import *
"""


class _CodeType(Enum):
    CALL_BASED = "call_based"
    STANDARD_INPUT = "standard_input"


class TimeoutException(Exception):
    pass


def preflight() -> None:
    return None


def load_samples(
    spec: DatasetSpec,
    *,
    question_field: str = "question",
    starter_code_field: str | None = "starter_code",
    record_id_field: str | None = None,
    metadata_fields: list[str] | None = None,
    row_filter: dict[str, dict[str, object]] | None = None,
) -> list[SampleRecord]:
    if spec.max_samples is not None and spec.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")

    rows = load_rows_from_dataset(
        spec.dataset,
        config=spec.config,
        split=spec.split,
    )
    if rows:
        first_row = rows[0]
        if question_field not in first_row:
            raise SystemExit(
                f"Dataset '{spec.dataset}' must include '{question_field}'; "
                f"found {sorted(first_row.keys())}."
            )

    samples: list[SampleRecord] = []
    for idx, row in enumerate(rows):
        if not row_matches_filter(row, row_filter):
            continue
        if question_field not in row:
            raise SystemExit(f"Row {idx} is missing '{question_field}'.")
        prompt = row[question_field]
        if prompt is None:
            continue
        sample_id = resolve_sample_id(row.get("_source_sample_id", idx), idx)
        metadata = collect_row_metadata(row, metadata_fields=metadata_fields) or {}
        if starter_code_field and starter_code_field in row and row[starter_code_field] is not None:
            metadata[starter_code_field] = str(row[starter_code_field])
        samples.append(
            SampleRecord(
                sample_id=sample_id,
                prompt=str(prompt),
                source_split=spec.split,
                prompt_style="taco_codegen",
                record_id=(
                    str(row.get(record_id_field))
                    if record_id_field and row.get(record_id_field) is not None
                    else None
                ),
                metadata=metadata or None,
            )
        )
        if spec.max_samples is not None and len(samples) >= spec.max_samples:
            break
    return samples


def build_codegen_prompt(
    tokenizer: Any | None,
    prompt: str,
    *,
    starter_code: str | None,
    prompt_format: str,
    thinking_mode: str = "default",
) -> str:
    prompt_text = prompt.rstrip()
    starter = (starter_code or "").strip()
    if starter:
        prompt_text = (
            f"{prompt_text}\n\n"
            "Starter code:\n"
            f"{starter}"
        )
    if prompt_format == "raw":
        return prompt_text
    if tokenizer is None:
        raise SystemExit("Tokenizer is required for chat_template codegen prompts.")
    return format_user_prompt(
        tokenizer,
        prompt_text,
        prompt_format=prompt_format,
        thinking_mode=thinking_mode,
    )


def estimate_pass_at_k(
    num_samples: list[int] | int,
    num_correct: list[int],
    k: int,
) -> list[float]:
    if isinstance(num_samples, int):
        num_samples_iter = [num_samples] * len(num_correct)
    else:
        if len(num_samples) != len(num_correct):
            raise ValueError("num_samples and num_correct must have the same length.")
        num_samples_iter = list(num_samples)

    estimates: list[float] = []
    for n_raw, c_raw in zip(num_samples_iter, num_correct, strict=True):
        n = int(n_raw)
        c = int(c_raw)
        if n < 1:
            estimates.append(0.0)
            continue
        if n - c < k:
            estimates.append(1.0)
            continue
        product = 1.0
        for denom in range(n - c + 1, n + 1):
            product *= 1.0 - (float(k) / float(denom))
        estimates.append(1.0 - product)
    return estimates


def evaluate_prompt_archive_rows(
    prompt_archive_rows: list[dict[str, object]],
    *,
    k_values: tuple[int, ...] = (1, 10),
) -> tuple[dict[str, object], dict[tuple[int, int], bool]]:
    grading_by_key: dict[tuple[int, int], bool] = {}
    total_by_prompt: list[int] = []
    correct_by_prompt: list[int] = []
    detail_labels: list[str] = []

    for row in sorted(prompt_archive_rows, key=lambda item: int(item.get("sample_id", -1))):
        if bool(row.get("prompt_too_long")):
            continue
        sample_id = int(row.get("sample_id", -1))
        if sample_id < 0:
            raise RuntimeError("TACO archive row is missing sample_id.")
        metadata = row.get("record_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"TACO archive row {sample_id} is missing record_metadata.")
        if "input_output" not in metadata:
            raise RuntimeError(f"TACO archive row {sample_id} is missing input_output.")
        label = (
            str(row.get("record_id"))
            if row.get("record_id") is not None
            else str(sample_id)
        )
        detail_labels.append(label)
        rollouts = row.get("rollouts")
        if not isinstance(rollouts, list):
            raise RuntimeError(f"TACO archive row {sample_id} is missing rollouts.")
        prompt_total = 0
        prompt_correct = 0
        sample = {"input_output": metadata["input_output"]}
        for rollout in rollouts:
            if not isinstance(rollout, dict):
                raise RuntimeError(f"TACO archive row {sample_id} contains a malformed rollout.")
            rollout_index = int(rollout.get("rollout_index", prompt_total))
            completion = str(rollout.get("completion_text", ""))
            verdicts = check_correctness(sample, completion)
            passed = all(bool(verdict is True or (isinstance(verdict, int) and verdict > 0)) for verdict in verdicts)
            grading_by_key[(sample_id, rollout_index)] = passed
            prompt_total += 1
            prompt_correct += int(passed)
        total_by_prompt.append(prompt_total)
        correct_by_prompt.append(prompt_correct)

    metrics: dict[str, object] = {"detail": {}}
    for k in k_values:
        if not total_by_prompt or any(total < k for total in total_by_prompt):
            continue
        estimates = estimate_pass_at_k(total_by_prompt, correct_by_prompt, k)
        metrics[f"pass@{k}"] = sum(estimates) / float(len(estimates))
        metrics["detail"][f"pass@{k}"] = {
            label: estimate
            for label, estimate in zip(detail_labels, estimates, strict=True)
        }
    return metrics, grading_by_key


def check_correctness(
    sample: dict[str, object],
    generation: str,
    *,
    timeout: int = _GLOBAL_TIMEOUT_SEC,
    debug: bool = False,
) -> list[bool | int]:
    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(ctx_name)
    queue = ctx.Queue()
    process = ctx.Process(
        target=_run_test_subprocess,
        args=(sample, generation, debug, queue),
    )
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()
    if not queue.empty():
        result = queue.get()
        if isinstance(result, list):
            return result

    try:
        in_outs = json.loads(str(sample["input_output"]))
        num_cases = len(in_outs.get("inputs", []))
    except Exception:
        num_cases = 1
    return [-1 for _ in range(max(1, num_cases))]


def _run_test_subprocess(
    sample: dict[str, object],
    generation: str,
    debug: bool,
    queue: mp.Queue,
) -> None:
    try:
        queue.put(_run_test(sample, generation, debug=debug))
    except Exception:
        queue.put([-3])


def _run_test(
    sample: dict[str, object],
    generation: str,
    *,
    debug: bool = False,
) -> list[bool | int]:
    in_outs = json.loads(str(sample["input_output"]))
    method_name = in_outs.get("fn_name")
    code_type = (
        _CodeType.CALL_BASED
        if isinstance(method_name, str) and method_name.strip()
        else _CodeType.STANDARD_INPUT
    )
    inputs_list: list[object] = []
    outputs_list: list[object] = []
    for raw_inputs, raw_outputs in zip(
        in_outs.get("inputs", []),
        in_outs.get("outputs", []),
        strict=True,
    ):
        inputs, outputs = _process_input_output(raw_inputs, raw_outputs)
        inputs_list.append(inputs)
        outputs_list.append(outputs)

    if code_type is _CodeType.CALL_BASED:
        program = _synthesize_call_based_program(generation)
        method = _compile_and_get_callable(program, str(method_name))
        if method is None:
            return [-2]
        detail = _execute_call_based(
            method,
            inputs_list,
            outputs_list,
            timeout=_PER_TEST_TIMEOUT_SEC,
            debug=debug,
        )
    else:
        compiled_program, exec_program = _synthesize_standard_input_program(generation)
        detail = _execute_standard_input(
            compiled_program,
            exec_program,
            inputs_list,
            outputs_list,
            timeout=_PER_TEST_TIMEOUT_SEC,
            debug=debug,
        )

    results: list[bool | int] = []
    for passed, label in detail:
        if label == "passed":
            results.append(True)
        elif label == "false":
            results.append(False)
        elif label == "timeout":
            results.append(-1)
        else:
            results.append(-3)
    return results


def _process_input_output(inputs: object, outputs: object) -> tuple[object, object]:
    try:
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
            inputs = [{int(key): value for key, value in inputs[0].items()}]
    except Exception:
        pass
    try:
        if isinstance(outputs, dict):
            outputs = [{int(key): value for key, value in outputs.items()}]
    except Exception:
        pass
    try:
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            outputs = [{int(key): value for key, value in outputs[0].items()}]
    except Exception:
        pass
    return inputs, outputs


def _synthesize_call_based_program(raw_code: str) -> str:
    return _CB_IMPORTS + raw_code


def _synthesize_standard_input_program(raw_code: str) -> tuple[str, str]:
    compiled_program = ""
    executable_program = ""
    code_lines = raw_code.splitlines()
    code_types: list[int] = []
    for line in code_lines:
        if "import *" in line:
            code_types.append(2)
        elif line.startswith("from ") or line.startswith("import "):
            code_types.append(1)
        else:
            code_types.append(0)

    started = False
    special_imports = "\n".join(
        line.lstrip("\t")
        for line, code_type in zip(code_lines, code_types, strict=True)
        if code_type == 2
    )

    for line, code_type in zip(code_lines, code_types, strict=True):
        if code_type == 0 and not started:
            executable_program += _CB_IMPORTS
            executable_program += "\nstdin = sys.stdin\nstdout = sys.stdout\n"
            executable_program += f"{line}\n"

            compiled_program += _CB_IMPORTS
            compiled_program += special_imports
            compiled_program += "\nstdin = sys.stdin\nstdout = sys.stdout\n"
            compiled_program += "def code():\n"
            compiled_program += f"\t{line}\n"
            started = True
            continue

        executable_program += f"{line}\n"
        if code_type < 2:
            if started:
                compiled_program += "\t"
            compiled_program += f"{line}\n"

    return compiled_program, executable_program


def _compile_and_get_callable(program: str, method_name: str) -> Any | None:
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_PER_TEST_TIMEOUT_SEC)
    try:
        namespace: dict[str, object] = {}
        exec(program, namespace)
        target: object
        if "class Solution" in program and "Solution" in namespace:
            target = namespace["Solution"]()  # type: ignore[operator]
        else:
            target = type("_Namespace", (), namespace)()
        return getattr(target, method_name)
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _execute_call_based(
    method: Any,
    inputs_list: list[object],
    outputs_list: list[object],
    *,
    timeout: int,
    debug: bool = False,
) -> list[tuple[bool, str]]:
    reliability_guard()
    results: list[tuple[bool, str]] = []
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        for inputs, expected_outputs in zip(inputs_list, outputs_list, strict=True):
            signal.alarm(timeout)
            faulthandler.enable()
            try:
                if isinstance(inputs, list):
                    exec_outputs = method(*inputs)
                else:
                    exec_outputs = method(inputs)
            except TimeoutException:
                results.append((False, "timeout"))
                continue
            except Exception:
                results.append((False, "runtime_error"))
                continue
            finally:
                faulthandler.disable()
                signal.alarm(0)

            if isinstance(exec_outputs, tuple):
                exec_outputs = list(exec_outputs)
            passed = exec_outputs == expected_outputs
            if isinstance(expected_outputs, list) and expected_outputs:
                passed = passed or exec_outputs == expected_outputs[0]
            try:
                if isinstance(exec_outputs[0], tuple):
                    exec_outputs = [list(item) for item in exec_outputs]
                    passed = passed or exec_outputs == expected_outputs[0]
            except Exception:
                pass
            results.append((bool(passed), "passed" if passed else "false"))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    return results


def _execute_standard_input(
    compiled_program: str,
    executable_program: str,
    inputs_list: list[object],
    outputs_list: list[object],
    *,
    timeout: int,
    debug: bool = False,
) -> list[tuple[bool, str]]:
    temp_program_path = _create_temp_file(executable_program)
    compiled_program_path = _create_temp_file(compiled_program)
    results: list[tuple[bool, str]] = []
    try:
        for raw_inputs, raw_outputs in zip(inputs_list, outputs_list, strict=True):
            _remove_tmp_files()
            inputs = "\n".join(raw_inputs) if isinstance(raw_inputs, list) else str(raw_inputs)
            outputs = "\n".join(raw_outputs) if isinstance(raw_outputs, list) else str(raw_outputs)
            try:
                result = subprocess.run(
                    [sys.executable, temp_program_path],
                    input=inputs,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                exec_code = 999
            except subprocess.TimeoutExpired:
                results.append((False, "timeout"))
                continue
            except Exception:
                results.append((False, "runtime_error"))
                continue

            if result.returncode != 0:
                exec_code = _retry_standard_input(
                    compiled_program_path,
                    inputs,
                    outputs,
                    timeout=timeout,
                )
                if exec_code is None:
                    results.append((False, "runtime_error"))
                    continue
                results.append((exec_code, "passed" if exec_code else "false"))
                continue

            passed = _compare_standard_output(result.stdout, outputs, debug=debug)
            results.append((passed, "passed" if passed else "false"))
    finally:
        for path in (temp_program_path, compiled_program_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    return results


def _retry_standard_input(
    program_path: str,
    inputs: str,
    outputs: str,
    *,
    timeout: int,
) -> bool | None:
    try:
        inputs_path = _create_temp_file(inputs)
        with open(inputs_path, "r", encoding="utf-8") as handle:
            result = subprocess.run(
                [sys.executable, program_path],
                stdin=handle,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        os.remove(inputs_path)
        if result.returncode == 0:
            return _compare_standard_output(result.stdout, outputs)
    except Exception:
        pass

    try:
        with open("input.txt", "w", encoding="utf-8") as handle:
            handle.write(inputs)
        result = subprocess.run(
            [sys.executable, program_path],
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and os.path.exists("output.txt"):
            with open("output.txt", "r", encoding="utf-8") as handle:
                return _compare_standard_output(handle.read(), outputs)
    except Exception:
        return None
    return None


def _compare_standard_output(
    exec_outputs: object,
    outputs: object,
    *,
    debug: bool = False,
) -> bool:
    if _stripped_string_compare(str(exec_outputs), str(outputs)):
        return True

    if isinstance(exec_outputs, list):
        joined = "\n".join(str(item) for item in exec_outputs)
        if _stripped_string_compare(joined, str(outputs)):
            return True
        stripped = "\n".join(str(item).strip() for item in exec_outputs)
        if _stripped_string_compare(stripped, str(outputs)):
            return True

    normalized_outputs = outputs
    if isinstance(normalized_outputs, tuple):
        normalized_outputs = list(normalized_outputs)

    try:
        if exec_outputs == [normalized_outputs]:
            return True
        if isinstance(normalized_outputs, list) and exec_outputs == normalized_outputs:
            return True
        if (
            isinstance(normalized_outputs, list)
            and isinstance(exec_outputs, list)
            and exec_outputs
            and isinstance(exec_outputs[0], str)
            and [item.strip() for item in exec_outputs] == normalized_outputs
        ):
            return True
    except Exception:
        pass

    split_outputs = normalized_outputs
    if isinstance(split_outputs, list):
        split_outputs = [
            [part.strip() for part in str(item).split("\n") if part.strip()]
            for item in split_outputs
        ]
    else:
        split_outputs = [part.strip() for part in str(split_outputs).split("\n") if part.strip()]

    try:
        if exec_outputs == [split_outputs]:
            return True
        if isinstance(split_outputs, list) and exec_outputs == split_outputs:
            return True
    except Exception:
        pass

    filtered_exec_outputs = exec_outputs
    if isinstance(filtered_exec_outputs, list):
        filtered_exec_outputs = [item for item in filtered_exec_outputs if item]
    try:
        if filtered_exec_outputs == [split_outputs]:
            return True
        if isinstance(split_outputs, list) and filtered_exec_outputs == split_outputs:
            return True
    except Exception:
        pass

    if _compare_float_sequences(filtered_exec_outputs, split_outputs):
        return True

    if isinstance(split_outputs, list):
        token_outputs = [set(str(item).split()) for item in split_outputs]
    else:
        token_outputs = set(str(split_outputs).split())
    try:
        if filtered_exec_outputs == token_outputs:
            return True
    except Exception:
        pass

    token_exec_outputs = filtered_exec_outputs
    if isinstance(token_exec_outputs, list):
        token_exec_outputs = [set(str(item).split()) for item in token_exec_outputs if str(item).split()]
    else:
        token_exec_outputs = set(str(token_exec_outputs).split())
    try:
        if set(frozenset(item) for item in token_exec_outputs) == set(
            frozenset(item) for item in token_outputs
        ):
            return True
    except Exception:
        pass
    try:
        rounded_exec = {
            frozenset(round(float(token), 3) for token in item)
            for item in token_exec_outputs
        }
        rounded_outputs = {
            frozenset(round(float(token), 3) for token in item)
            for item in token_outputs
        }
        return rounded_exec == rounded_outputs
    except Exception:
        if debug:
            return False
    return False


def _compare_float_sequences(exec_outputs: object, outputs: object) -> bool:
    try:
        exec_floats = [float(item) for item in exec_outputs]
        output_floats = [float(item) for item in outputs]
        if len(exec_floats) == len(output_floats):
            return all(
                math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)
                for lhs, rhs in zip(exec_floats, output_floats, strict=True)
            )
    except Exception:
        pass
    try:
        if isinstance(exec_outputs, list) and exec_outputs and isinstance(exec_outputs[0], list):
            exec_floats = [float(item) for item in exec_outputs[0]]
            output_floats = [float(item) for item in outputs[0]]
            if len(exec_floats) == len(output_floats):
                return all(
                    math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)
                    for lhs, rhs in zip(exec_floats, output_floats, strict=True)
                )
    except Exception:
        pass
    return False


def _stripped_string_compare(lhs: str, rhs: str) -> bool:
    return lhs.strip() == rhs.strip()


def _create_temp_file(content: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as handle:
        handle.write(content)
        return handle.name


def _remove_tmp_files() -> None:
    for path in ("input.txt", "output.txt"):
        if os.path.exists(path):
            os.remove(path)


def _timeout_handler(signum, frame) -> None:  # type: ignore[override]
    raise TimeoutException


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (maximum_memory_bytes, maximum_memory_bytes),
        )
        resource.setrlimit(
            resource.RLIMIT_DATA,
            (maximum_memory_bytes, maximum_memory_bytes),
        )
        if platform.uname().system != "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK,
                (maximum_memory_bytes, maximum_memory_bytes),
            )

    faulthandler.disable()

    import builtins
    import shutil

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None  # type: ignore[assignment]
    os.system = None  # type: ignore[assignment]
    os.putenv = None  # type: ignore[assignment]
    os.remove = None  # type: ignore[assignment]
    os.removedirs = None  # type: ignore[assignment]
    os.rmdir = None  # type: ignore[assignment]
    os.fchdir = None  # type: ignore[assignment]
    os.setuid = None  # type: ignore[assignment]
    os.fork = None  # type: ignore[assignment]
    os.forkpty = None  # type: ignore[assignment]
    os.killpg = None  # type: ignore[assignment]
    os.rename = None  # type: ignore[assignment]
    os.renames = None  # type: ignore[assignment]
    os.truncate = None  # type: ignore[assignment]
    os.replace = None  # type: ignore[assignment]
    os.unlink = None  # type: ignore[assignment]
    os.fchmod = None  # type: ignore[assignment]
    os.fchown = None  # type: ignore[assignment]
    os.chmod = None  # type: ignore[assignment]
    os.chown = None  # type: ignore[assignment]
    os.chroot = None  # type: ignore[assignment]
    os.lchflags = None  # type: ignore[assignment]
    os.lchmod = None  # type: ignore[assignment]
    os.lchown = None  # type: ignore[assignment]
    os.getcwd = None  # type: ignore[assignment]
    os.chdir = None  # type: ignore[assignment]
    shutil.rmtree = None  # type: ignore[assignment]
    shutil.move = None  # type: ignore[assignment]
    shutil.chown = None  # type: ignore[assignment]
    subprocess.Popen = None  # type: ignore[assignment]
    __builtins__["help"] = None  # type: ignore[index]
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
