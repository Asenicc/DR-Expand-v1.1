"""CPU-only release checks; never trains, opens datasets, or initializes CUDA."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"Release hash mismatch: {name}"
        if name.endswith(".py"):
            compile((root / name).read_text(encoding="utf-8-sig"), name, "exec")
    import numpy as np
    import torch
    for source in ["N", "EA"]:
        payload = json.loads((root / f"weights/relation_{source}.json").read_text(encoding="utf-8"))
        w = np.asarray(payload["weights"], dtype=np.float64)
        assert w.shape == (18, 51)
        assert np.isfinite(w).all() and (w > 0).all()
        assert np.allclose(w.mean(1), 1)
        assert np.allclose(w.sum(1), 51)
        assert w[:, :45].sum(1).min() > 1
        assert payload["smoke"] is False and payload["target_data_used"] is False
    sys.path.insert(0, str(root / "scripts"))
    import dre_v1p1_core as core
    # Independent local generator, CPU only. Identity and empty actor behavior.
    gen = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn((2, 3, 64, 25, 2), generator=gen)
    x[0, :, :, :, 1] = 0
    p = torch.zeros((2, 10), requires_grad=True)
    candidate = core.kinematic_transform(x, p)
    assert torch.allclose(candidate, x, atol=2e-6)
    assert torch.count_nonzero(candidate[0, :, :, :, 1]) == 0
    ref, valid = core.relations(x)
    r, _ = core.relations(candidate)
    q = core.relation_error_squared(r, ref, valid, torch.ones((2, 51)))
    assert torch.isfinite(q).all() and q.max() < 1e-10
    grad, = torch.autograd.grad(q.sum(), p)
    assert torch.isfinite(grad).all()
    assert valid[0, :45].all() and not valid[0, 45:].any()
    assert valid[1].all()
    print(json.dumps(dict(passed=True, files_checked=len(manifest["files_sha256"]),
          python_syntax=True, row_mean=1, row_sum=51, identity_and_empty_actor=True,
          finite_zero_residual_gradient=True, training_updates=0,
          target_data_opened=False, cuda_initialized=torch.cuda.is_initialized())))
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
