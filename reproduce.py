"""Portable runner for DR-Expand v1.1; original training math is imported unchanged.

This entry point is new release code, not the historical experiment runner.
It performs no work on import. See README.md for data and teacher preparation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
AUTHOR_COMMIT = "fb4d621c14f088c3bbac4031faef9cc0a6caf1cd"


def evaluate_preserving_rng(erm, model, domain, output):
    """Target evaluation must not change subsequent source sampling/augmentation."""
    output.mkdir(parents=True, exist_ok=False)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    was_training = model.training
    model_hash = erm.model_sha(model)
    try:
        with torch.random.fork_rng(devices=[0]):
            metrics = erm.evaluate(model, domain, output, 64, 0)
        assert erm.model_sha(model) == model_hash, "Evaluation changed model/BN"
        return metrics
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["teacher", "student"], required=True)
    parser.add_argument("--source", choices=["N", "EA"], required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared common18/xsub64 directory, not raw skeletons")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, help="Same-source RAR epoch-50 checkpoint")
    parser.add_argument("--selection", choices=["target-best", "fixed-final-50"],
                        default="target-best")
    args = parser.parse_args()
    if args.stage == "student" and args.teacher is None:
        parser.error("--teacher is required for student training")
    if args.stage == "teacher" and args.teacher is not None:
        parser.error("Teacher training starts from random initialization")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the established FP32 training setup")
    upstream = ROOT / "Recover-and-Resample"
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != AUTHOR_COMMIT:
        raise RuntimeError(f"Expected author commit {AUTHOR_COMMIT}; found {commit}")
    # Verify release bytes, including the exact historical core, before training.
    release = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in release["files_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Release file changed: {relative}")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing an existing output directory: {out}")
    data = args.data_root.resolve()
    domain = "ntu" if args.source == "N" else "etriA"
    expected_rows = 12651 if args.source == "N" else 13568
    # Verify source bytes against the historical data before using fitted statistics.
    original_protocol = json.loads((ROOT / "reference/protocol.json").read_text(encoding="utf-8"))
    for name in ["train_data_joint.npy", "train_label.pkl"]:
        path = data / domain / name
        expected = original_protocol["files_sha256"][f"data/common18/xsub64/{domain}/{name}"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Historical source-data hash mismatch: {path}")
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_erm as erm
    import train_rar as rar
    import dre_v1p1_core as core
    erm.DATA = data
    rar.DATA = data
    core.DATA = data
    # The copied modules resolve ROOT from this checkout. No historical workspace
    # protocol, smoke marker, output directory, or Windows scheduled task is used.
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = erm.Model_thin(**core.MODEL_ARGS).cuda()
    initial_sha = erm.model_sha(model)
    teacher = None
    teacher_sha = None
    if args.stage == "student":
        with torch.random.fork_rng(devices=[0]):
            teacher = erm.Model_thin(**core.MODEL_ARGS).cuda()
        cp = torch.load(args.teacher, map_location="cpu", weights_only=False)
        tc = cp["config"]
        if cp["epoch"] != 50 or tc["source"] != args.source or tc["seed"] != 0:
            raise ValueError("Teacher must be same-source seed0 epoch50")
        if tc.get("stage", tc.get("method")) not in ("teacher", "Recover-and-Resample"):
            raise ValueError("Expected a Recover-and-Resample teacher")
        teacher.load_state_dict(cp["model"])
        del cp
        teacher.eval().requires_grad_(False)
        teacher_sha = erm.model_sha(teacher)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9,
                                nesterov=True, weight_decay=1e-4)
    loader_rng = torch.Generator().manual_seed(0)
    search_rng = torch.Generator(device="cuda").manual_seed(91026)
    recoverer = rar.build_recoverer(args.source, out)
    dataset = erm.Skeletons(domain, "train", rotate=True)
    assert len(dataset) == expected_rows
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True, generator=loader_rng)
    weights = None
    if teacher is not None:
        weights = core.fit_relation_weights(args.source, out / "relation_weights.json").cuda()
    config = dict(stage=args.stage, source=args.source, seed=0, epochs=50,
                  selection="fixed-final-50" if teacher is None else args.selection,
                  batch_size=64, steps_per_epoch=len(loader), source_samples=len(dataset),
                  initial_model_sha256=initial_sha, author_commit=commit,
                  teacher_checkpoint=str(args.teacher) if teacher is not None else None,
                  teacher_model_sha256=teacher_sha, search_seed=91026,
                  method_config=core.CONFIG,
                  target_selection="per-target BA; earliest tie; target labels used for selection"
                                   if teacher is not None and args.selection == "target-best"
                                   else "fixed final epoch; no target checkpoint selection")
    erm.dump(out / "config.json", config)
    history = []
    best = {}
    targets = ["etri", "pku"] if args.source == "N" else ["ntu", "pku"]
    for epoch in range(50):
        model.train()
        begin = time.monotonic()
        lr = .1 * (.1 ** sum(epoch >= value for value in [30, 40]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0.
        seen = 0
        for step, (original, y) in enumerate(loader, 1):
            original = original.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.no_grad():
                recovered = recoverer.apply_W_all(original, "ABL_linearextrap_beta", "cropresize0.7")
                base = recoverer.merge_raw_and_aug(original, recovered, .75)
            if not torch.isfinite(base).all():
                raise FloatingPointError("Non-finite recovered input")
            if teacher is None:
                optimizer.zero_grad(set_to_none=True)
                logits = model(base)
                loss = F.cross_entropy(logits, y)
            else:
                expanded = core.adversarial_expand(model, base, y, weights, search_rng)[0] if epoch >= 5 else base
                with torch.no_grad():
                    teacher_logits = teacher(original)
                optimizer.zero_grad(set_to_none=True)
                logits, features = model(torch.cat([original, expanded]), return_feat=True)
                z0, z1 = logits.chunk(2)
                f0, f1 = features.chunk(2)
                ce0, ce1 = F.cross_entropy(z0, y), F.cross_entropy(z1, y)
                con = core.supervised_contrastive(torch.cat([f0, f1]), y.repeat(2), .1)
                kl, _ = core.teacher_preservation(z0, teacher_logits, y)
                loss = .5 * (ce0 + ce1) + .1 * con + .25 * kl
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite training gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            seen += len(y)
            if step == 1 or step % 25 == 0 or step == len(loader):
                erm.dump(out / "progress.json", dict(status="training", stage=args.stage,
                         source=args.source, epoch=epoch+1, step=step,
                         steps_per_epoch=len(loader), updated_at=erm.now()))
                print(f"{args.stage}/{args.source} epoch={epoch+1}/50 step={step}/{len(loader)} loss={float(loss.detach()):.6f}", flush=True)
        torch.cuda.synchronize()
        row = dict(epoch=epoch+1, learning_rate=lr, loss=total_loss/seen,
                   updates=step, samples_seen=seen, seconds=time.monotonic()-begin)
        history.append(row)
        with (out / "training.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = dict(epoch=epoch+1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          config=config, torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                          python_rng=random.getstate(), loader_generator=loader_rng.get_state(),
                          search_rng=search_rng.get_state())
        temp = out / "last.pt.tmp"
        torch.save(checkpoint, temp)
        temp.replace(out / "last.pt")
        torch.save(checkpoint, out / f"epoch_{epoch+1:03d}.pt")
        if teacher is not None and (args.selection == "target-best" or epoch == 49):
            for target in targets:
                directory = out / "evaluations" / f"epoch_{epoch+1:03d}" / target
                metrics = evaluate_preserving_rng(erm, model, target, directory)
                record = dict(epoch=epoch+1, **metrics)
                with (out / "target_epochs.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                # Strict improvement reproduces np.argmax's first-maximum tie rule.
                if target not in best or metrics["balanced_accuracy_percent"] > best[target]["balanced_accuracy_percent"]:
                    best[target] = dict(epoch=epoch+1,
                                        checkpoint=f"epoch_{epoch+1:03d}.pt",
                                        prediction_file=str(directory.relative_to(out) / f"{target}_predictions.npz"),
                                        **metrics)
                    erm.dump(out / "best_by_target.json", best)
    if teacher is not None:
        assert erm.model_sha(teacher) == teacher_sha
    erm.dump(out / "results.json", dict(**config, metrics=best,
             checkpoint_model_sha256=erm.model_sha(model),
             completed_epochs=50, total_updates=sum(r["updates"] for r in history),
             training_seconds=sum(r["seconds"] for r in history), completed_at=erm.now(),
             release_runner_validation="Static/import/CPU-core checks only; this runner has not been GPU-trained in this publication task"))
    erm.dump(out / "progress.json", dict(status="completed", source=args.source,
             stage=args.stage, completed_epochs=50, updated_at=erm.now()))


if __name__ == "__main__":
    main()
