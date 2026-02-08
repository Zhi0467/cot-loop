#!/usr/bin/env python3
"""Train a probe on last-token prefill activations."""

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.dataloader import make_dataloader, read_manifest
from loop_probe.probes.linear_probe import LinearProbe
from loop_probe.train_utils import (
    choose_device,
    evaluate_binary_metrics,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", default=None)

    return parser.parse_args()


def _evaluate(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    return evaluate_binary_metrics(labels_cat, logits_cat)


def _write_jsonl(path: str, row: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _checkpoint_payload(
    model, epoch: int, step: int, metrics: dict[str, float]
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }


def main() -> None:
    args = _parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    set_seed(args.seed)
    device = choose_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from dotenv import load_dotenv
    except Exception as exc:
        raise SystemExit(
            "python-dotenv is required. Install dependencies with `uv sync`."
        ) from exc

    load_dotenv()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not found. Put it in .env or environment.")

    import wandb

    manifest = read_manifest(args.data_dir)
    input_dim = int(manifest["input_dim"])

    train_loader = make_dataloader(
        args.data_dir,
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = make_dataloader(
        args.data_dir,
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_info = manifest.get("train", {})
    num_pos = int(train_info.get("num_positive", 0))
    num_neg = int(train_info.get("num_negative", 0))
    if num_pos > 0 and num_neg > 0:
        pos_weight = torch.tensor(float(num_neg / num_pos), dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)

    model = LinearProbe(input_dim=input_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "data_dir": args.data_dir,
            "input_dim": input_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pos_weight": float(pos_weight.item()),
        },
    )

    metrics_jsonl = os.path.join(args.out_dir, "metrics.jsonl")
    best_ckpt = os.path.join(args.out_dir, "best.pt")
    last_ckpt = os.path.join(args.out_dir, "last.pt")

    best_auc = float("-inf")
    best_f1 = float("-inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_logits = []
        epoch_labels = []

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(y.size(0))
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            global_step += 1

            # Store for epoch-level metrics
            epoch_logits.append(logits.detach().cpu())
            epoch_labels.append(y.detach().cpu())

            if batch_idx % args.log_every == 0:
                avg_loss = running_loss / max(seen, 1)
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "epoch": epoch,
                        "step": global_step,
                    },
                    step=global_step,
                )
                print(
                    f"epoch={epoch} batch={batch_idx} step={global_step} "
                    f"train_loss={avg_loss:.6f}",
                    flush=True,
                )

        train_loss_epoch = running_loss / max(seen, 1)
        
        # Compute full epoch training metrics (only once per epoch)
        epoch_logits_cat = torch.cat(epoch_logits, dim=0)
        epoch_labels_cat = torch.cat(epoch_labels, dim=0)
        train_metrics_epoch = evaluate_binary_metrics(epoch_labels_cat, epoch_logits_cat)

        # Always log training metrics to wandb (every epoch)
        wandb.log(
            {
                "train/loss_epoch": train_loss_epoch,
                "train/accuracy_epoch": train_metrics_epoch["accuracy"],
                "train/macro_f1_epoch": train_metrics_epoch["macro_f1"],
                "train/roc_auc_epoch": train_metrics_epoch["roc_auc"],
                "epoch": epoch,
                "step": global_step,
            },
            step=global_step,
        )

        # Evaluate on test set based on eval_every
        if epoch % args.eval_every == 0:
            eval_metrics = _evaluate(model, test_loader, device)
            
            # Log to metrics.jsonl
            log_row = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss_epoch,
                "train_accuracy": train_metrics_epoch["accuracy"],
                "train_macro_f1": train_metrics_epoch["macro_f1"],
                "train_roc_auc": train_metrics_epoch["roc_auc"],
                **eval_metrics,
            }
            _write_jsonl(metrics_jsonl, log_row)

            # Log eval metrics to wandb
            wandb.log(
                {
                    "eval/accuracy": eval_metrics["accuracy"],
                    "eval/macro_f1": eval_metrics["macro_f1"],
                    "eval/roc_auc": eval_metrics["roc_auc"],
                    "epoch": epoch,
                    "step": global_step,
                },
                step=global_step,
            )

            # Update best checkpoint
            auc = eval_metrics["roc_auc"]
            auc_rank = auc if not math.isnan(auc) else float("-inf")
            is_better = (auc_rank > best_auc) or (
                auc_rank == best_auc and eval_metrics["macro_f1"] > best_f1
            )
            if is_better:
                best_auc = auc_rank
                best_f1 = eval_metrics["macro_f1"]
                torch.save(
                    _checkpoint_payload(model, epoch, global_step, eval_metrics),
                    best_ckpt,
                )

            print(
                " ".join(
                    [
                        f"epoch={epoch}",
                        f"train_loss={train_loss_epoch:.6f}",
                        f"train_acc={train_metrics_epoch['accuracy']:.4f}",
                        f"eval_acc={eval_metrics['accuracy']:.4f}",
                        f"eval_f1={eval_metrics['macro_f1']:.4f}",
                        f"eval_auc={eval_metrics['roc_auc']:.4f}",
                    ]
                ),
                flush=True,
            )
        else:
            # Print training metrics only
            print(
                " ".join(
                    [
                        f"epoch={epoch}",
                        f"train_loss={train_loss_epoch:.6f}",
                        f"train_acc={train_metrics_epoch['accuracy']:.4f}",
                        f"train_f1={train_metrics_epoch['macro_f1']:.4f}",
                        f"train_auc={train_metrics_epoch['roc_auc']:.4f}",
                    ]
                ),
                flush=True,
            )

        torch.save(
            _checkpoint_payload(model, epoch, global_step, {
                "train_loss": train_loss_epoch,
                "train_accuracy": train_metrics_epoch["accuracy"],
                "train_macro_f1": train_metrics_epoch["macro_f1"],
                "train_roc_auc": train_metrics_epoch["roc_auc"],
            }),
            last_ckpt,
        )

    run.finish()
    print(f"Saved checkpoints to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
