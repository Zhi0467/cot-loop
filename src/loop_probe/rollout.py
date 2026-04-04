import multiprocessing as mp
import os
from dataclasses import dataclass
from multiprocessing.connection import wait

from transformers import AutoTokenizer, GenerationConfig

from .configs import RolloutConfig

DEFAULT_GROUPED_MAX_NUM_SEQS = 16


@dataclass(frozen=True)
class GeneratedRollout:
    token_ids: list[int]
    text: str
    finish_reason: str | None


def resolve_sampling_defaults(
    model_id: str,
    *,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[float, int]:
    try:
        gen_config = GenerationConfig.from_pretrained(model_id)
    except OSError as exc:
        if "generation_config.json" not in str(exc):
            raise
        resolved_top_p = 1.0
        resolved_top_k = -1
    else:
        resolved_top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
        resolved_top_k = gen_config.top_k if gen_config.top_k is not None else -1
    if top_p is not None:
        resolved_top_p = top_p
    if top_k is not None:
        resolved_top_k = top_k
    if resolved_top_k == 0:
        resolved_top_k = -1
    return resolved_top_p, resolved_top_k


def _normalize_finish_reason(reason: object) -> str:
    if hasattr(reason, "value"):
        reason = getattr(reason, "value")
    if reason is None:
        return "unknown"
    text = str(reason).strip()
    if not text:
        return "unknown"
    return text.split(".")[-1].lower()


def _get_visible_devices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return [v.strip() for v in visible.split(",") if v.strip()]
    try:
        import torch  # type: ignore

        count = torch.cuda.device_count()
        return [str(i) for i in range(count)]
    except Exception:
        return []


def _suppress_sem_unlink_errors() -> None:
    if os.environ.get("VLLM_SUPPRESS_SEM_UNLINK_ERRORS") != "1":
        return
    try:
        import multiprocessing.synchronize as mp_sync  # type: ignore
    except Exception:
        return
    if not hasattr(mp_sync, "_cleanup"):
        return
    if getattr(mp_sync, "_vllm_cleanup_patched", False):
        return

    orig_cleanup = mp_sync._cleanup

    def _cleanup(name: str) -> None:
        try:
            orig_cleanup(name)
        except FileNotFoundError:
            pass

    mp_sync._cleanup = _cleanup  # type: ignore[attr-defined]
    mp_sync._vllm_cleanup_patched = True

    try:
        import multiprocessing.resource_tracker as rt  # type: ignore
    except Exception:
        return
    if getattr(rt, "_vllm_rt_patched", False):
        return

    orig_register = rt.register
    orig_unregister = rt.unregister

    def _register(name: str, rtype: str) -> None:
        if rtype == "semaphore":
            return
        orig_register(name, rtype)

    def _unregister(name: str, rtype: str) -> None:
        if rtype == "semaphore":
            return
        orig_unregister(name, rtype)

    rt.register = _register  # type: ignore[assignment]
    rt.unregister = _unregister  # type: ignore[assignment]
    rt._vllm_rt_patched = True


def _generate_grouped_rollouts_single_process(
    prompts: list[str],
    cfg: RolloutConfig,
    *,
    seed: int,
    log_prefix: str,
) -> list[list[GeneratedRollout]]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit(
            "vLLM is required for rollout generation. Install vLLM first."
        ) from exc

    if not prompts:
        return []
    if cfg.max_num_seqs is not None and cfg.max_num_seqs < 1:
        raise SystemExit("--max-num-seqs must be >= 1 when provided.")
    if cfg.num_generations < 1:
        raise SystemExit("--num-generations must be >= 1.")
    if cfg.max_num_seqs is not None and cfg.max_num_seqs < cfg.num_generations:
        raise SystemExit(
            "--max-num-seqs must be >= --num-generations when repeated rollouts "
            "are enabled."
        )

    top_p, top_k = resolve_sampling_defaults(
        cfg.model_id,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )

    llm_kwargs = {
        "model": cfg.model_id,
        "tensor_parallel_size": cfg.tp,
        "dtype": cfg.dtype,
        "max_model_len": cfg.max_model_len,
        "trust_remote_code": cfg.trust_remote_code,
    }
    gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "").strip()
    if gpu_mem_util:
        try:
            gpu_mem_value = float(gpu_mem_util)
        except Exception as exc:
            raise SystemExit(
                "VLLM_GPU_MEMORY_UTILIZATION must be a float in (0, 1]."
            ) from exc
        if not (0.0 < gpu_mem_value <= 1.0):
            raise SystemExit(
                "VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]."
            )
        llm_kwargs["gpu_memory_utilization"] = gpu_mem_value
    if cfg.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = cfg.max_num_seqs
    if cfg.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens

    llm = LLM(**llm_kwargs)
    all_rollouts: list[list[GeneratedRollout]] = []

    if cfg.max_num_seqs is not None:
        chunk_size = max(cfg.max_num_seqs // cfg.num_generations, 1)
    else:
        chunk_size = max(DEFAULT_GROUPED_MAX_NUM_SEQS // cfg.num_generations, 1)
    for start in range(0, len(prompts), chunk_size):
        end = min(start + chunk_size, len(prompts))
        batch_prompts = prompts[start:end]
        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=cfg.max_tokens,
            n=cfg.num_generations,
            repetition_penalty=1.0,
            seed=seed + start,
        )

        outputs = llm.generate(batch_prompts, sampling_params)
        if len(outputs) != len(batch_prompts):
            raise RuntimeError(
                f"Expected {len(batch_prompts)} rollout outputs, got {len(outputs)}."
            )

        for out in outputs:
            if len(out.outputs) != cfg.num_generations:
                raise RuntimeError(
                    "Expected "
                    f"{cfg.num_generations} rollout sample(s) per prompt, got "
                    f"{len(out.outputs)}."
                )
            prompt_rollouts: list[GeneratedRollout] = []
            for sample in out.outputs:
                token_ids = getattr(sample, "token_ids", None)
                if not token_ids:
                    token_ids = tokenizer.encode(sample.text, add_special_tokens=False)
                finish_reason = getattr(sample, "finish_reason", None)
                if finish_reason is None:
                    finish_reason = getattr(sample, "stop_reason", None)
                if finish_reason is None:
                    finish_reason = getattr(out, "finish_reason", None)
                if finish_reason is None:
                    finish_reason = getattr(out, "stop_reason", None)
                prompt_rollouts.append(
                    GeneratedRollout(
                        token_ids=list(token_ids),
                        text=str(sample.text),
                        finish_reason=_normalize_finish_reason(finish_reason),
                    )
                )
            all_rollouts.append(prompt_rollouts)

        print(f"{log_prefix} generated {end}/{len(prompts)}", flush=True)

    return all_rollouts


def _dp_rollout_worker(
    rank: int,
    device: str,
    shard_items: list[tuple[int, str]],
    cfg: RolloutConfig,
    seed: int,
    out_conn,
) -> None:
    _suppress_sem_unlink_errors()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    prompts = [prompt for _, prompt in shard_items]
    grouped_rollouts = _generate_grouped_rollouts_single_process(
        prompts,
        cfg,
        seed=seed + rank * 100_000,
        log_prefix=f"[rollout-dp-rank {rank}]",
    )
    indexed = [(idx, rollouts) for (idx, _), rollouts in zip(shard_items, grouped_rollouts)]
    out_conn.send((rank, indexed))
    out_conn.close()


def generate_grouped_rollouts(
    prompts: list[str],
    cfg: RolloutConfig,
    *,
    seed: int,
) -> list[list[GeneratedRollout]]:
    if cfg.dp < 1:
        raise SystemExit("--dp must be >= 1.")
    if cfg.dp == 1:
        return _generate_grouped_rollouts_single_process(
            prompts,
            cfg,
            seed=seed,
            log_prefix="[rollout]",
        )

    if cfg.tp != 1:
        raise SystemExit("Data-parallel rollouts require tp=1.")

    devices = _get_visible_devices()
    worker_count = min(cfg.dp, len(prompts))
    if worker_count == 0:
        return []
    if len(devices) < worker_count:
        raise SystemExit(
            f"Requested dp={cfg.dp}, but only {len(devices)} visible GPU(s)."
        )

    ctx = mp.get_context("spawn")
    conn_to_rank: dict[object, int] = {}
    processes = []
    for rank in range(worker_count):
        shard_items = [(idx, prompts[idx]) for idx in range(rank, len(prompts), worker_count)]
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        conn_to_rank[parent_conn] = rank
        p = ctx.Process(
            target=_dp_rollout_worker,
            args=(rank, devices[rank], shard_items, cfg, seed, child_conn),
        )
        p.start()
        child_conn.close()
        processes.append(p)

    by_rank: dict[int, list[tuple[int, list[GeneratedRollout]]]] = {}
    pending = set(conn_to_rank.keys())
    while len(by_rank) < worker_count:
        ready = wait(list(pending), timeout=30)
        if not ready:
            dead_missing = []
            for rank, proc in enumerate(processes):
                if rank in by_rank:
                    continue
                if proc.exitcode is not None:
                    dead_missing.append((rank, proc.exitcode))
            if dead_missing:
                raise SystemExit(
                    f"Rollout worker(s) exited before reporting outputs: {dead_missing}"
                )
            continue
        for conn in ready:
            rank = conn_to_rank[conn]
            try:
                msg_rank, indexed = conn.recv()
            except EOFError as exc:
                raise SystemExit(
                    f"Rollout worker {rank} exited before reporting outputs."
                ) from exc
            if int(msg_rank) != int(rank):
                raise SystemExit(
                    f"Rollout worker rank mismatch: expected {rank}, got {msg_rank}."
                )
            by_rank[int(rank)] = indexed
            conn.close()
            pending.discard(conn)

    failures = []
    for rank, proc in enumerate(processes):
        proc.join(timeout=30)
        if proc.is_alive():
            if rank in by_rank:
                proc.terminate()
                proc.join(timeout=10)
            else:
                proc.terminate()
                proc.join(timeout=10)
                failures.append((rank, "alive_without_outputs"))
                continue

        if proc.exitcode not in (0, None) and rank not in by_rank:
            failures.append((rank, proc.exitcode))

    if failures:
        raise SystemExit(f"Rollout worker(s) failed: {failures}")

    merged: list[list[GeneratedRollout] | None] = [None] * len(prompts)
    for rank in range(worker_count):
        for idx, rollouts in by_rank.get(rank, []):
            merged[idx] = rollouts

    if any(rollouts is None for rollouts in merged):
        raise SystemExit("Missing rollout outputs for some prompts.")

    return [rollouts for rollouts in merged if rollouts is not None]


def generate_rollout_token_ids(
    prompts: list[str],
    cfg: RolloutConfig,
    *,
    seed: int,
) -> list[list[int]]:
    if cfg.num_generations != 1:
        raise SystemExit(
            "generate_rollout_token_ids expects rollout_cfg.num_generations == 1. "
            "Use generate_grouped_rollouts() for repeated rollouts."
        )
    grouped_rollouts = generate_grouped_rollouts(
        prompts,
        cfg,
        seed=seed,
    )
    return [prompt_rollouts[0].token_ids for prompt_rollouts in grouped_rollouts]
