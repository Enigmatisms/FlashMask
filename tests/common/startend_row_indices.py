# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FlashMask ``startend_row_indices`` generators, in numpy.

A FlashMask mask is an int32 ``(B, H_mask, S, n)`` array of per-column row
bounds (``n`` is 1 or 2 for causal, 2 or 4 otherwise); ``H_mask`` is 1 here, i.e.
broadcast over query heads. Every generator returns ``(startend_row_indices,
causal)``, and ``None`` means dense attention.

Numpy on purpose: `tests/paddle` and `tests/torch` build the same mask from the
same code and convert to their own tensor type, so the two suites cannot drift.

The defaults are tuned for ``seqlen_k == 8192`` and scale proportionally
otherwise. ``DOC_PACKING`` is the packing the context-parallel tests use: its doc
lengths sum to exactly 32768, so it fits S_total = s_local * cp = 32768 with no
padding or scaling.
"""

import numpy as np

__all__ = [
    "DOC_PACKING",
    "GENERATORS",
    "startend_row_indices_to_attn_bias",
    "none_mask",
    "non_causal_mask",
    "causal_mask",
    "sliding_window_mask",
    "causal_document_mask",
    "document_mask",
    "prefix_lm_document_mask",
    "prefix_lm_causal_mask",
]

DOC_PACKING = [(360, 1478), (581, 3207), (225, 839), (266, 1721), (742, 3329), (666, 3018),
               (276, 2302), (424, 2872), (381, 2381), (455, 2022), (312, 1995), (597, 2786),
               (406, 2373), (310, 2445)]

_DEFAULT_DOCS = [2538, 1742, 3213]
_DEFAULT_PREFIX_DOCS = [(1024, 2538), (1742, 1742), (512, 3213)]


def _scaled(values, seqlen_k):
    """Scale the seqlen_k=8192 defaults to another sequence length."""
    if seqlen_k == 8192:
        return list(values)
    scale = seqlen_k / 8192
    if isinstance(values[0], tuple):
        return [tuple(int(v * scale) for v in pair) for pair in values]
    return [int(v * scale) for v in values]


def _as_mask(columns, batch_size, seqlen_q, seqlen_k):
    """Stack per-column bound lists into (B, 1, S, n) int32, clipped to seqlen_q.

    Clipping is what keeps a bound meaning "no masking here": a row index past
    seqlen_q and seqlen_q itself both fall outside the query range.
    """
    arrs = [np.asarray(col, dtype=np.int32).reshape((1, 1, seqlen_k, 1)) for col in columns]
    arr = np.concatenate(arrs, axis=-1)
    arr = np.repeat(arr, batch_size, axis=0)
    return np.clip(arr, None, seqlen_q).astype(np.int32)


def _doc_lens(doc_seqlens, seqlen_k):
    """Normalise a doc-length spec to a list of ints (drops prefix lengths)."""
    if doc_seqlens is None:
        doc_seqlens = _scaled(_DEFAULT_DOCS, seqlen_k)
    doc_seqlens = list(doc_seqlens)
    if isinstance(doc_seqlens[0], tuple):
        doc_seqlens = [doc for _, doc in doc_seqlens]
    return doc_seqlens


def none_mask(batch_size, seqlen_q, seqlen_k, h, causal=True):
    """No mask: dense attention, optionally causal."""
    return None, causal


def non_causal_mask(batch_size, seqlen_q, seqlen_k, h):
    """Explicit all-visible mask (the non-causal identity)."""
    columns = [[seqlen_k] * seqlen_k, [0] * seqlen_k]
    return _as_mask(columns, batch_size, seqlen_q, seqlen_k), False


def causal_mask(batch_size, seqlen_q, seqlen_k, h):
    """Explicit causal mask."""
    return _as_mask([[seqlen_k] * seqlen_k], batch_size, seqlen_q, seqlen_k), True


def sliding_window_mask(batch_size, seqlen_q, seqlen_k, h, window_size=None):
    """Causal sliding window of ``window_size`` keys."""
    if window_size is None:
        window_size = 1024 if seqlen_k == 8192 else int(1024 * (seqlen_k / 8192))
    bounds = np.arange(window_size, seqlen_k + window_size, dtype=np.int32)
    return _as_mask([bounds], batch_size, seqlen_q, seqlen_k), True


def causal_document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """Causal attention inside each packed document (block-diagonal + causal)."""
    doc_seqlens = _doc_lens(doc_seqlens, seqlen_k)
    assert len(doc_seqlens) >= 3
    assert sum(doc_seqlens) <= seqlen_k
    # the tail padding joins the last document
    doc_seqlens[-1] += seqlen_k - sum(doc_seqlens)
    bounds = np.repeat(np.cumsum(doc_seqlens), doc_seqlens)
    return _as_mask([bounds], batch_size, seqlen_q, seqlen_k), True


def document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """Bidirectional attention inside each packed document (block-diagonal)."""
    doc_seqlens = _doc_lens(doc_seqlens, seqlen_k)
    assert len(doc_seqlens) >= 3
    assert sum(doc_seqlens) <= seqlen_k
    padding = seqlen_k - sum(doc_seqlens)

    down, up = [], []
    cur = doc_seqlens[0]
    for i, doc in enumerate(doc_seqlens):
        down.extend([cur] * doc)
        if i < len(doc_seqlens) - 1:
            cur += doc_seqlens[i + 1]
    if padding > 0:
        down.extend([cur] * padding)

    cur = 0
    for i, doc in enumerate(doc_seqlens):
        up.extend([cur] * doc)
        if i < len(doc_seqlens) - 1:
            cur += doc_seqlens[i + 1]
    if padding > 0:
        up.extend([cur] * padding)

    return _as_mask([down, up], batch_size, seqlen_q, seqlen_k), False


def prefix_lm_document_mask(batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None):
    """Per-document prefix-LM: bidirectional over the prefix, causal after it.

    ``doc_seqlens`` is a list of ``(prefix_length, doc_length)`` pairs.
    """
    if doc_seqlens is None:
        doc_seqlens = _scaled(_DEFAULT_PREFIX_DOCS, seqlen_k)
    doc_seqlens = list(doc_seqlens)
    assert len(doc_seqlens) >= 2
    total = sum(doc for _, doc in doc_seqlens)
    assert total <= seqlen_k
    padding = seqlen_k - total

    down = []
    cur = doc_seqlens[0][1]
    for i, (_, doc) in enumerate(doc_seqlens):
        down.extend([cur] * doc)
        if i < len(doc_seqlens) - 1:
            cur += doc_seqlens[i + 1][1]
    if padding > 0:
        down.extend([cur] * padding)

    up = []
    cur = 0
    for prefix, doc in doc_seqlens:
        up.extend([cur] * prefix + list(range(cur + prefix, cur + doc)))
        cur += doc
    if padding > 0:
        up.extend([total] * padding)

    return _as_mask([down, up], batch_size, seqlen_q, seqlen_k), False


def prefix_lm_causal_mask(batch_size, seqlen_q, seqlen_k, h, prefix_length=None):
    """One shared prefix, causal afterwards."""
    if prefix_length is None:
        prefix_length = 1024 if seqlen_k == 8192 else int(1024 * (seqlen_k / 8192))
    assert prefix_length <= seqlen_k
    down = [seqlen_k] * seqlen_k
    up = [0] * prefix_length + list(range(prefix_length, seqlen_k))
    return _as_mask([down, up], batch_size, seqlen_q, seqlen_k), False


#: mask name -> generator, for the context-parallel tests' ``--masks`` option
GENERATORS = {
    "document": document_mask,
    "prefix_lm": prefix_lm_document_mask,
}


def startend_row_indices_to_attn_bias(startend_row_indices, seqlen_q, nheads, causal=True):
    """Expand row bounds into a dense additive bias, float32 (B, nheads, Sq, Sk).

    0 where a key is visible, ``-inf`` where it is masked out -- the form an eager
    attention reference can just add to the scores. Written as a plain loop over
    columns to stay a readable statement of the mask semantics; the reference is
    only ever run on the small shapes these tests use.
    """
    if startend_row_indices is None:
        return None
    bz, num_head, seqlen_k, bound_num = startend_row_indices.shape
    assert nheads % num_head == 0
    m = np.zeros((bz, num_head, seqlen_q, seqlen_k), dtype=np.float32)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seqlen_k):
                downstart = int(startend_row_indices[bi, hi, j, 0])
                if has_end:
                    downend = int(startend_row_indices[bi, hi, j, 1])
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    # from flash-attention 2.1 on, a causal mask with seqlen_q != seqlen_k
                    # is aligned to the BOTTOM-RIGHT corner of the attention matrix.
                    m[bi, hi, : max(0, j - (seqlen_k - seqlen_q)), j] = -np.inf
                elif has_end:
                    upstart = int(startend_row_indices[bi, hi, j, 2])
                    upend = int(startend_row_indices[bi, hi, j, 3])
                    m[bi, hi, upstart:upend, j] = -np.inf
                else:
                    upend = int(startend_row_indices[bi, hi, j, 1])
                    m[bi, hi, :upend, j] = -np.inf
    return np.repeat(m, nheads // num_head, axis=1)
