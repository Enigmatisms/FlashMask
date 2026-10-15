# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Framework-agnostic test helpers.

Nothing here imports paddle or torch: mask generation, the accuracy policy and
the benchmark timer are plain numpy / stdlib, so `tests/paddle` and `tests/torch`
share one implementation instead of drifting apart.
"""
