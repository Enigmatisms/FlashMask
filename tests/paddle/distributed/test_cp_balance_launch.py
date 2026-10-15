# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""pytest entry point for the paddle cp_balance accuracy test.

The check itself lives in cp_balance_main.py, which has to run as one process per
GPU (fixed CP=4); this wrapper just launches it through run_cp_balance.sh and
asserts it exited cleanly. It skips (rather than fails) when the machine cannot
host the test: too few GPUs, no SM100 device, or a FlashMask built without the
'cpb' (cp_balance) component.
"""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

paddle = pytest.importorskip("paddle")

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "run_cp_balance.sh"
MIN_GPUS = 4


def _gpu_count():
    try:
        return paddle.device.cuda.device_count()
    except Exception:
        return 0


def _is_sm100():
    try:
        return paddle.device.cuda.get_device_capability()[0] >= 10
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash is needed to run the launcher"),
    pytest.mark.skipif(_gpu_count() < MIN_GPUS, reason=f"cp_balance needs >= {MIN_GPUS} GPUs"),
    pytest.mark.skipif(not _is_sm100(), reason="FM-4 cp_balance is SM100+ only"),
    pytest.mark.skipif(
        importlib.util.find_spec("flashmask.cp_balance") is None,
        reason="FlashMask was built without the 'cpb' (cp_balance) component",
    ),
]


def test_cp_balance():
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=HERE,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"run_cp_balance.sh exited {proc.returncode}\n"
            f"--- stdout (tail) ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-4000:]}"
        )
