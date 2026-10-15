# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""pytest entry point for the torch context-parallel overlap test.

The accuracy check itself lives in cp_overlap_main.py, which has to run as one
process per GPU; this wrapper just launches it through run_cp_overlap.sh (torchrun)
and asserts it exited cleanly. It skips (rather than fails) when the machine cannot
host the test at all: too few GPUs, no SM100 device, or a FlashMask built without
the optional 'ovl' component.
"""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "run_cp_overlap.sh"
MIN_GPUS = 4


def _gpu_count():
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def _is_sm100():
    try:
        return torch.cuda.get_device_capability()[0] >= 10
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash is needed to run the launcher"),
    pytest.mark.skipif(_gpu_count() < MIN_GPUS, reason=f"context parallel needs >= {MIN_GPUS} GPUs"),
    pytest.mark.skipif(not _is_sm100(), reason="FM-4 overlap is SM100+ only"),
    pytest.mark.skipif(
        importlib.util.find_spec("flashmask.overlap") is None,
        reason="FlashMask was built without the 'ovl' (overlap) component",
    ),
]

# cp_size, head dim, kv-head counts, mask. Two launches only: d=128 exercises the
# BHSD layout and split KV, d=512 the big-headdim / shared-KV path.
CASES = [
    (4, 128, "16,1", "document"),
    (4, 512, "1", "prefix_lm"),
]


@pytest.mark.parametrize(
    "cp_size, d, hks, mask",
    CASES,
    ids=[f"cp{c[0]}-d{c[1]}-{c[3]}" for c in CASES],
)
def test_cp_overlap(cp_size, d, hks, mask):
    env = dict(
        os.environ,
        CP_SIZE=str(cp_size),
        D=str(d),
        HQ="16",
        HKS=hks,
        BS="1",
        MASKS=mask,
        REPEATS="1",
    )
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"run_cp_overlap.sh exited {proc.returncode} for cp={cp_size} d={d} mask={mask}\n"
            f"--- stdout (tail) ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-4000:]}"
        )
