"""Test dell'infrastruttura di training (Fase 8): schedule del learning rate e clipping."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.autograd import Tensor  # noqa: E402
from ronklm.train import _clip_gradients, cosine_lr  # noqa: E402


def test_warmup_ramps_up_linearly():
    # durante il warmup il lr cresce da ~0 al valore base
    base, warmup, steps, min_lr = 1.0, 100, 1000, 0.1
    assert cosine_lr(0, base, warmup, steps, min_lr) < cosine_lr(50, base, warmup, steps, min_lr)
    assert abs(cosine_lr(warmup - 1, base, warmup, steps, min_lr) - base) < 1e-6


def test_cosine_decays_to_min():
    base, warmup, steps, min_lr = 1.0, 100, 1000, 0.1
    # subito dopo il warmup ~ base; a fine training ~ min_lr
    assert abs(cosine_lr(warmup, base, warmup, steps, min_lr) - base) < 1e-2
    assert abs(cosine_lr(steps, base, warmup, steps, min_lr) - min_lr) < 1e-6
    # monotono in discesa nella fase di decay
    mid = cosine_lr(steps // 2, base, warmup, steps, min_lr)
    assert min_lr < mid < base


def test_grad_clip_limits_norm():
    rng = np.random.default_rng(0)
    ps = [Tensor(rng.normal(size=(4, 4))) for _ in range(3)]
    for p in ps:
        p.grad = rng.normal(scale=10.0, size=p.data.shape)  # gradienti grandi
    _clip_gradients(ps, max_norm=1.0)
    total = np.sqrt(sum((p.grad ** 2).sum() for p in ps))
    assert total <= 1.0 + 1e-6


def test_grad_clip_leaves_small_grads_untouched():
    rng = np.random.default_rng(1)
    ps = [Tensor(rng.normal(size=(2, 2)))]
    ps[0].grad = np.full((2, 2), 0.1)  # norma piccola (< 1)
    before = ps[0].grad.copy()
    _clip_gradients(ps, max_norm=1.0)
    assert np.allclose(before, ps[0].grad)


if __name__ == "__main__":
    from _runner import run

    run(globals())
