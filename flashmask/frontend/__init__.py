# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Per-framework public APIs.

Each subpackage holds only what cannot be abstracted: autograd wiring, argument
validation in the framework's own idiom, and (later) collectives. Import the one
you need directly -- importing this package does not import either framework:

    from flashmask.frontend.paddle import flashmask_attention
    from flashmask.frontend.torch import flashmask_attention
"""

__all__ = []
