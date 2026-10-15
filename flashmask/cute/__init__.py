# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Flash Mask CUTE (CUDA Template Engine) implementation."""

__version__ = "4.0.0"

import cutlass.cute as cute

from flashmask.cute.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flashmask_attention",
]


def __getattr__(name):
    """Resolve the two public entry points on first access.

    They live in the paddle frontend, which imports this package -- binding them
    eagerly here is a cycle (`flashmask.cute` -> frontend -> `flashmask.cute`).
    Deferring also keeps `import flashmask.cute.<kernel>` free of any framework
    dependency: the kernel modules themselves import neither paddle nor torch.
    """
    if name in __all__:
        from flashmask.cute import interface

        return getattr(interface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
