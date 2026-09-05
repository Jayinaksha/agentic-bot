#!/usr/bin/env python3
"""
Text embeddings for semantic recall.

Deliberately small. all-MiniLM-L6-v2 is 384-dimensional, ~90 MB, and runs on a
CPU core in single-digit milliseconds per sentence. On a robot that is already
sending frames to a cloud VLA, spending GPU on embeddings would be the wrong
trade: the strings being embedded are short labels and one-line instructions,
where a large model buys almost nothing.

Three backends, chosen at import time by what is available:

    sentence-transformers   local, default, no network
    NVIDIA NIM              remote, if R2D2_EMBED_URL is set - use this when the
                            robot is a Pi and even MiniLM is too much
    hashing                 deterministic fallback so the stack still runs, and
                            the tests still pass, with neither installed

The hashing backend is a real fallback, not a stub: it produces stable,
normalised vectors so inserts and queries stay consistent. Its recall is poor,
which is why it logs loudly.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import struct
from typing import List, Optional, Sequence

log = logging.getLogger('r2d2.embeddings')

EMBED_DIM = 384
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class Embedder:
    """Base interface: text in, unit-norm float list out."""

    dim = EMBED_DIM

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


class LocalEmbedder(Embedder):
    """sentence-transformers, run locally."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        if self.dim != EMBED_DIM:
            raise ValueError(
                f'{model_name} produces {self.dim}-d vectors but the schema '
                f'declares vector({EMBED_DIM}). Change EMBED_DIM and the '
                f'schema together, then reindex.')

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class RemoteEmbedder(Embedder):
    """Any OpenAI-compatible /embeddings endpoint, including NVIDIA NIM."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: float = 20.0):
        import httpx
        self.base_url = base_url.rstrip('/')
        self.model = model
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        response = self._client.post(
            f'{self.base_url}/embeddings',
            json={'model': self.model, 'input': list(texts),
                  # NIM's retrieval models distinguish document and query text;
                  # passage is right for what we store.
                  'input_type': 'passage'},
        )
        response.raise_for_status()
        data = response.json()['data']
        return [_normalise(item['embedding']) for item in
                sorted(data, key=lambda d: d.get('index', 0))]


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing. Works everywhere, recalls poorly.

    Present so that a machine with no model and no network can still run the
    whole stack end to end - useful in CI and when bringing a new robot up.
    Token hashes are spread across the vector with a fixed seed so the same
    text always lands in the same place.
    """

    def __init__(self):
        log.warning(
            'using the hashing embedder: semantic recall will be keyword-like. '
            'Install sentence-transformers, or set R2D2_EMBED_URL, for real '
            'retrieval.')

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> List[float]:
        vector = [0.0] * EMBED_DIM
        tokens = [t for t in text.lower().replace('-', ' ').split() if t]
        for token in tokens:
            digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
            index = struct.unpack('<Q', digest)[0] % EMBED_DIM
            # Sign from a second hash so unrelated tokens can cancel rather than
            # only ever accumulating.
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[index] += sign
        return _normalise(vector)


def _normalise(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-12:
        # An all-zero vector would make cosine distance undefined; park it on a
        # fixed unit axis so it is merely unhelpful rather than an error.
        out = [0.0] * len(vector)
        out[0] = 1.0
        return out
    return [v / norm for v in vector]


def make_embedder(prefer_remote: bool = False) -> Embedder:
    """Pick the best embedder available in this environment."""
    remote_url = os.environ.get('R2D2_EMBED_URL')
    remote_model = os.environ.get('R2D2_EMBED_MODEL', 'nvidia/nv-embedqa-e5-v5')

    if remote_url and prefer_remote:
        try:
            return RemoteEmbedder(remote_url, remote_model,
                                  os.environ.get('R2D2_NVIDIA_API_KEY'))
        except Exception as exc:                  # noqa: BLE001 - httpx optional
            log.warning('remote embedder unavailable (%s); trying local', exc)

    try:
        return LocalEmbedder(os.environ.get('R2D2_EMBED_MODEL_LOCAL', DEFAULT_MODEL))
    except Exception as exc:                      # noqa: BLE001 - model optional
        log.warning('local embedder unavailable (%s)', exc)

    if remote_url:
        try:
            return RemoteEmbedder(remote_url, remote_model,
                                  os.environ.get('R2D2_NVIDIA_API_KEY'))
        except Exception as exc:                  # noqa: BLE001
            log.warning('remote embedder unavailable (%s)', exc)

    return HashingEmbedder()
