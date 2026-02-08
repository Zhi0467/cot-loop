import multiprocessing as mp
import os
import queue as queue_module

from transformers import AutoTokenizer, GenerationConfig

from .configs import RolloutConfig


def resolve_sampling_defaults(model_id: str) -> tuple[float, int]:
    gen_config = GenerationConfig.from_pretrained(model_id)
    top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
    top_k = gen_config.top_k if gen_config.top_k is not None else -1
    if top_k == 0:
        top_k = -1
    return top_p, top_k


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


def _generate_rollout_token_ids_single_process(
    prompts: list[str],
    cfg: RolloutConfig,
    *,
    seed: int,
    log_prefix: str,
) -> list[list[int]]:
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

    top_p, top_k = resolve_sampling_defaults(cfg.model_id)

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
    if cfg.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = cfg.max_num_seqs
    if cfg.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens

    llm = LLM(**llm_kwargs)
    all_token_ids: list[list[int]] = []

    chunk_size = cfg.max_num_seqs if cfg.max_num_seqs is not None else len(prompts)
    for start in range(0, len(prompts), chunk_size):
        end = min(start + chunk_size, len(prompts))
        batch_prompts = prompts[start:end]
        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=cfg.max_tokens,
            n=1,
            repetition_penalty=1.0,
            seed=seed + start,
        )

        outputs = llm.generate(batch_prompts, sampling_params)
        if len(outputs) != len(batch_prompts):
            raise RuntimeError(
                f"Expected {len(batch_prompts)} rollout outputs, got {len(outputs)}."
            )

        for out in outputs:
            if len(out.outputs) != 1:
                raise RuntimeError(
                    f"Expected 1 rollout sample per prompt, got {len(out.outputs)}."
                )
            sample = out.outputs[0]
            token_ids = getattr(sample, "token_ids", None)
            if not token_ids:
                token_ids = tokenizer.encode(sample.text, add_special_tokens=False)
            all_token_ids.append(list(token_ids))

        print(f"{log_prefix} generated {end}/{len(prompts)}", flush=True)

    return all_token_ids


def _dp_rollout_worker(
    rank: int,
    device: str,
    shard_items: list[tuple[int, str]],
    cfg: RolloutConfig,
    seed: int,
    out_queue: "mp.queues.SimpleQueue",
) -> None:
    _suppress_sem_unlink_errors()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    prompts = [prompt for _, prompt in shard_items]
    token_ids = _generate_rollout_token_ids_single_process(
        prompts,
        cfg,
        seed=seed + rank * 100_000,
        log_prefix=f"[rollout-dp-rank {rank}]",
    )
    indexed = [(idx, toks) for (idx, _), toks in zip(shard_items, token_ids)]
    out_queue.put((rank, indexed))


def generate_rollout_token_ids(
    prompts: list[str],
    cfg: RolloutConfig,
    *,
    seed: int,
) -> list[list[int]]:
    if cfg.dp < 1:
        raise SystemExit("--dp must be >= 1.")
    if cfg.dp == 1:
        return _generate_rollout_token_ids_single_process(
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
    out_queue: "mp.queues.Queue" = ctx.Queue()
    processes = []
    for rank in range(worker_count):
        shard_items = [(idx, prompts[idx]) for idx in range(rank, len(prompts), worker_count)]
        p = ctx.Process(
            target=_dp_rollout_worker,
            args=(rank, devices[rank], shard_items, cfg, seed, out_queue),
        )
        p.start()
        processes.append(p)

    by_rank: dict[int, list[tuple[int, list[int]]]] = {}
    while len(by_rank) < worker_count:
        try:
            rank, indexed = out_queue.get(timeout=30)
        except queue_module.Empty:
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
        by_rank[int(rank)] = indexed

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

    merged: list[list[int] | None] = [None] * len(prompts)
    for rank in range(worker_count):
        for idx, toks in by_rank.get(rank, []):
            merged[idx] = toks

    if any(toks is None for toks in merged):
        raise SystemExit("Missing rollout outputs for some prompts.")

    return [toks for toks in merged if toks is not None]
