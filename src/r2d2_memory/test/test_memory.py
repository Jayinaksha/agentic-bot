#!/usr/bin/env python3
"""Unit tests for the ledger hash chain, object fusion and embeddings.

    python3 -m pytest src/r2d2_memory/test/test_memory.py

No NATS and no Postgres needed: the chain logic runs against NullLedger and the
fusion maths is pure. The parts that genuinely need a database are exercised by
scripts/check_memory.py, not here.
"""

import asyncio
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from r2d2_memory.embeddings import (  # noqa: E402
    EMBED_DIM, HashingEmbedder, _normalise)
from r2d2_memory.ledger import (  # noqa: E402
    GENESIS_HASH, LedgerRecord, NullLedger, verify_chain)
from r2d2_memory.store import (  # noqa: E402
    MERGE_RADIUS_MAX, MERGE_RADIUS_MIN, ObjectHit, fuse_position, merge_radius)


def _record(seq, prev_hash, payload=None, kind='percept'):
    return LedgerRecord(
        seq=seq, subject=f'r2d2.{kind}.0.chair', kind=kind,
        payload=payload if payload is not None else {'label': 'chair'},
        ts=1000.0 + seq, record_id=f'id-{seq}', prev_hash=prev_hash,
    ).sealed()


# ------------------------------------------------------------- hash chaining

def test_a_sealed_record_verifies():
    assert _record(1, GENESIS_HASH).verify()


def test_an_unsealed_record_does_not_verify():
    record = LedgerRecord(seq=1, subject='s', kind='percept', payload={},
                          ts=1.0, record_id='x', prev_hash=GENESIS_HASH)
    assert not record.verify()


def test_hash_covers_the_payload():
    record = _record(1, GENESIS_HASH)
    original = record.hash
    record.payload['label'] = 'table'
    assert record.hash == original       # stored hash is stale
    assert not record.verify()           # and no longer matches the content


def test_hash_covers_the_timestamp():
    record = _record(1, GENESIS_HASH)
    record.ts += 1.0
    assert not record.verify()


def test_hash_covers_the_predecessor_link():
    record = _record(1, GENESIS_HASH)
    record.prev_hash = 'f' * 64
    assert not record.verify()


def test_intact_chain_reports_no_break():
    chain = []
    prev = GENESIS_HASH
    for seq in range(1, 20):
        record = _record(seq, prev)
        chain.append(record)
        prev = record.hash
    assert verify_chain(chain) is None


def test_a_tampered_payload_is_located():
    """The property the ledger exists for: editing history is detectable."""
    chain = []
    prev = GENESIS_HASH
    for seq in range(1, 10):
        record = _record(seq, prev)
        chain.append(record)
        prev = record.hash

    chain[4].payload['confidence'] = 0.99
    assert verify_chain(chain) == 4


def test_a_removed_record_is_located():
    chain = []
    prev = GENESIS_HASH
    for seq in range(1, 10):
        record = _record(seq, prev)
        chain.append(record)
        prev = record.hash

    del chain[3]
    # The record that used to follow the deleted one now points at a hash that
    # is no longer its predecessor.
    assert verify_chain(chain) == 3


def test_a_resealed_record_still_breaks_the_chain():
    """Re-sealing a tampered record fixes its own hash but not its successor's
    prev_hash, which is what makes the chain, not the hash, the protection."""
    chain = []
    prev = GENESIS_HASH
    for seq in range(1, 6):
        record = _record(seq, prev)
        chain.append(record)
        prev = record.hash

    chain[2].payload['label'] = 'sofa'
    chain[2].sealed()
    assert chain[2].verify()
    assert verify_chain(chain) == 3


def test_json_round_trip_preserves_the_hash():
    record = _record(7, 'a' * 64)
    revived = LedgerRecord.from_json(record.to_json())
    assert revived.hash == record.hash
    assert revived.verify()


# ------------------------------------------------------------- null ledger

def test_null_ledger_chains_appends():
    async def scenario():
        ledger = NullLedger()
        await ledger.connect()
        first = await ledger.append('percept', {'label': 'chair'}, floor=0)
        second = await ledger.append('percept', {'label': 'table'}, floor=0)
        assert first.prev_hash == GENESIS_HASH
        assert second.prev_hash == first.hash
        audit = await ledger.audit()
        assert audit['intact']
        assert audit['records'] == 2
        await ledger.close()

    asyncio.run(scenario())


def test_null_ledger_replays_in_order():
    async def scenario():
        ledger = NullLedger()
        for i in range(5):
            await ledger.append('fact', {'content': f'note {i}'})
        seen = [r.payload['content'] async for r in ledger.replay()]
        assert seen == [f'note {i}' for i in range(5)]

    asyncio.run(scenario())


# ------------------------------------------------------------ object fusion

def test_merge_radius_shrinks_with_evidence():
    assert merge_radius(1) == pytest.approx(MERGE_RADIUS_MAX)
    assert merge_radius(4) < merge_radius(1)
    assert merge_radius(100) == pytest.approx(MERGE_RADIUS_MIN)


def test_merge_radius_never_leaves_its_bounds():
    for n in (0, 1, 2, 9, 50, 10000):
        assert MERGE_RADIUS_MIN <= merge_radius(n) <= MERGE_RADIUS_MAX


def test_merge_radius_handles_a_zero_count():
    assert merge_radius(0) == pytest.approx(MERGE_RADIUS_MAX)


def test_fusing_equal_weights_lands_midway():
    x, y, w = fuse_position(0.0, 0.0, 1.0, 2.0, 0.0, 1.0)
    assert (x, y) == pytest.approx((1.0, 0.0))
    assert w == pytest.approx(2.0)


def test_a_well_observed_object_resists_a_single_detection():
    """Forty observations should not be dragged far by the forty-first."""
    x, y, _ = fuse_position(2.0, 4.0, old_weight=40.0,
                            new_x=3.0, new_y=4.0, new_weight=0.6)
    assert x == pytest.approx(2.0, abs=0.02)


def test_a_new_object_takes_its_first_detection():
    x, y, w = fuse_position(0.0, 0.0, 0.0, 5.0, 6.0, 0.7)
    assert (x, y) == pytest.approx((5.0, 6.0))
    assert w == pytest.approx(0.7)


def test_repeated_observations_converge_on_the_truth():
    """Noisy detections around a true position must average towards it."""
    true_x, true_y = 3.0, 4.0
    offsets = [(0.18, -0.12), (-0.15, 0.09), (0.07, 0.14),
               (-0.09, -0.06), (0.11, 0.03)]
    x, y, w = true_x + offsets[0][0], true_y + offsets[0][1], 0.6
    for dx, dy in offsets[1:]:
        x, y, w = fuse_position(x, y, w, true_x + dx, true_y + dy, 0.6)
    assert math.dist((x, y), (true_x, true_y)) < 0.08


# -------------------------------------------------------------- object hits

def test_object_hit_serialises_for_a_tool_response():
    hit = ObjectHit(id='abc', label='chair', description='wooden dining chair',
                    floor=0, place='kitchen', x=2.041, y=3.926,
                    confidence=0.8123, observation_count=12,
                    position_sigma=0.1449, distance=1.4142, similarity=0.9012)
    d = hit.as_dict()
    assert d['label'] == 'chair'
    assert d['x'] == 2.041
    assert d['distance_m'] == 1.41
    assert d['similarity'] == 0.901


def test_object_hit_omits_unknown_distance():
    hit = ObjectHit(id='a', label='sofa', description=None, floor=0, place=None,
                    x=0.0, y=0.0, confidence=0.5, observation_count=1,
                    position_sigma=0.5)
    assert 'distance_m' not in hit.as_dict()
    assert 'similarity' not in hit.as_dict()


# --------------------------------------------------------------- embeddings

def test_hashing_embedder_shape_and_norm():
    embedder = HashingEmbedder()
    vector = embedder.encode_one('a wooden kitchen chair')
    assert len(vector) == EMBED_DIM
    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)


def test_hashing_embedder_is_deterministic():
    """Inserts and queries must land in the same place across processes."""
    a = HashingEmbedder().encode_one('fridge')
    b = HashingEmbedder().encode_one('fridge')
    assert a == b


def test_hashing_embedder_separates_different_text():
    embedder = HashingEmbedder()
    chair = embedder.encode_one('chair')
    fridge = embedder.encode_one('fridge')
    similarity = sum(a * b for a, b in zip(chair, fridge))
    assert similarity < 0.5


def test_hashing_embedder_finds_shared_words():
    embedder = HashingEmbedder()
    a = embedder.encode_one('wooden kitchen chair')
    b = embedder.encode_one('wooden kitchen table')
    similarity = sum(x * y for x, y in zip(a, b))
    assert similarity > 0.3


def test_empty_text_does_not_produce_a_zero_vector():
    """A zero vector makes cosine distance undefined, which pgvector will
    happily accept and then return nonsense for."""
    vector = HashingEmbedder().encode_one('')
    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)


def test_normalise_handles_a_zero_input():
    assert math.sqrt(sum(v * v for v in _normalise([0.0] * 8))) == pytest.approx(1.0)


def test_batch_encoding_matches_single_encoding():
    embedder = HashingEmbedder()
    batch = embedder.encode(['chair', 'table'])
    assert batch[0] == embedder.encode_one('chair')
    assert batch[1] == embedder.encode_one('table')
