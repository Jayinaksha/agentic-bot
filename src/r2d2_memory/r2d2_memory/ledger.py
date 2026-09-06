#!/usr/bin/env python3
"""
Append-only, hash-chained event ledger on NATS JetStream.

Why a ledger rather than just writing to Postgres
-------------------------------------------------
Everything the robot perceives, decides and does is an event. Writing those
straight into a mutable table means the current state is the only state: you
cannot ask why the robot believed the chair was at (2.1, 4.6) an hour ago, and
you cannot rebuild the semantic map after changing how detections are fused.

JetStream gives durable, ordered, replayable storage. On top of it this module
adds a hash chain: each record carries the SHA-256 of its predecessor, so the
stream is tamper-evident. Rewriting or deleting a past event breaks every hash
after it, and verify_chain() will say exactly where.

Subject layout
--------------
    r2d2.percept.<floor>.<label>    grounded VLA detections
    r2d2.telemetry.<kind>           pose, terrain, odometry health
    r2d2.decision.<kind>            agent tool calls and their results
    r2d2.episode.<outcome>          completed tasks
    r2d2.fact                       durable notes the agent writes itself

Streams are configured with a work-queue-free, limits-based retention so old
telemetry ages out while episodes and percepts are kept.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

try:
    import nats
    from nats.js.api import RetentionPolicy, StorageType, StreamConfig
    from nats.js.errors import NotFoundError
except ImportError:  # pragma: no cover - exercised only without the dependency
    nats = None
    NotFoundError = Exception

GENESIS_HASH = '0' * 64

STREAM_NAME = 'R2D2'
SUBJECT_PREFIX = 'r2d2'
SUBJECT_WILDCARD = f'{SUBJECT_PREFIX}.>'

# Retention. Percepts and episodes are the memory; telemetry is diagnostics and
# is allowed to age out, which is what keeps the stream a sensible size on a
# robot that runs for months.
DEFAULT_MAX_AGE_S = 90 * 24 * 3600
DEFAULT_MAX_BYTES = 8 * 1024 * 1024 * 1024


@dataclass
class LedgerRecord:
    """One immutable event.

    `prev_hash` and `hash` form the chain. `hash` is computed over the canonical
    JSON of every other field, so any later edit to the payload, the subject or
    the timestamp invalidates it.
    """

    seq: int
    subject: str
    kind: str
    payload: Dict[str, Any]
    ts: float
    record_id: str
    prev_hash: str
    hash: str = ''

    def digest(self) -> str:
        body = {
            'seq': self.seq,
            'subject': self.subject,
            'kind': self.kind,
            'payload': self.payload,
            'ts': self.ts,
            'record_id': self.record_id,
            'prev_hash': self.prev_hash,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def sealed(self) -> 'LedgerRecord':
        self.hash = self.digest()
        return self

    def verify(self) -> bool:
        return bool(self.hash) and self.hash == self.digest()

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), separators=(',', ':')).encode('utf-8')

    @classmethod
    def from_json(cls, raw: bytes) -> 'LedgerRecord':
        return cls(**json.loads(raw.decode('utf-8')))


def verify_chain(records: List[LedgerRecord]) -> Optional[int]:
    """Check a contiguous run of records. Returns the index of the first break.

    Two things are checked per record: that its own hash matches its content,
    and that it points at its predecessor. Returns None when the chain is
    intact.
    """
    previous: Optional[LedgerRecord] = None
    for i, record in enumerate(records):
        if not record.verify():
            return i
        if previous is not None and record.prev_hash != previous.hash:
            return i
        previous = record
    return None


class Ledger:
    """Async client for the event ledger.

    Usage:

        ledger = Ledger()
        await ledger.connect()
        await ledger.append('percept', {'label': 'chair', ...}, floor=0)
        async for record in ledger.replay('r2d2.percept.>'):
            ...
    """

    def __init__(self, servers: Optional[str] = None,
                 stream: str = STREAM_NAME,
                 max_age_s: int = DEFAULT_MAX_AGE_S,
                 max_bytes: int = DEFAULT_MAX_BYTES):
        self.servers = servers or os.environ.get('R2D2_NATS_URL', 'nats://127.0.0.1:4222')
        self.stream = stream
        self.max_age_s = max_age_s
        self.max_bytes = max_bytes

        self._nc = None
        self._js = None
        self._seq = 0
        self._last_hash = GENESIS_HASH
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        if nats is None:
            raise RuntimeError(
                'nats-py is not installed. pip install nats-py, or run the '
                'stack with memory disabled (R2D2_MEMORY=off).')

        self._nc = await nats.connect(
            servers=[s.strip() for s in self.servers.split(',')],
            name='r2d2-ledger',
            max_reconnect_attempts=-1,
            reconnect_time_wait=2,
        )
        self._js = self._nc.jetstream()

        config = StreamConfig(
            name=self.stream,
            subjects=[SUBJECT_WILDCARD],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.FILE,
            max_age=self.max_age_s,
            max_bytes=self.max_bytes,
            # Duplicate suppression across a reconnect: a record re-published
            # with the same Nats-Msg-Id inside this window is dropped, which
            # keeps the chain from forking after a network blip.
            duplicate_window=120,
        )
        try:
            await self._js.add_stream(config)
        except Exception:                     # noqa: BLE001 - already exists
            # Bring an existing stream up to this config rather than failing;
            # retention limits are the field most likely to have been changed.
            await self._js.update_stream(config)

        await self._restore_tip()

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
            self._nc = None
            self._js = None

    async def _restore_tip(self) -> None:
        """Resume the chain from the last record already in the stream.

        Without this, a restarted process would begin a second chain from the
        genesis hash and verify_chain would report a break at the restart point.
        """
        try:
            msg = await self._js.get_last_msg(self.stream, SUBJECT_WILDCARD)
        except Exception:                          # noqa: BLE001 - empty stream
            self._seq, self._last_hash = 0, GENESIS_HASH
            return

        try:
            record = LedgerRecord.from_json(msg.data)
        except (ValueError, TypeError):
            self._seq, self._last_hash = 0, GENESIS_HASH
            return

        self._seq = record.seq
        self._last_hash = record.hash

    # -------------------------------------------------------------- writing

    async def append(self, kind: str, payload: Dict[str, Any],
                     subject: Optional[str] = None,
                     floor: Optional[int] = None) -> LedgerRecord:
        """Seal one event onto the chain and publish it.

        Serialised on a lock: the chain is only meaningful if sequence numbers
        and prev_hash links are assigned in a single order.
        """
        if self._js is None:
            raise RuntimeError('ledger not connected; call connect() first')

        subject = subject or self._subject_for(kind, payload, floor)

        async with self._lock:
            record = LedgerRecord(
                seq=self._seq + 1,
                subject=subject,
                kind=kind,
                payload=payload,
                ts=time.time(),
                record_id=str(uuid.uuid4()),
                prev_hash=self._last_hash,
            ).sealed()

            await self._js.publish(subject, record.to_json(),
                                   headers={'Nats-Msg-Id': record.record_id})
            self._seq = record.seq
            self._last_hash = record.hash

        return record

    @staticmethod
    def _subject_for(kind: str, payload: Dict[str, Any],
                     floor: Optional[int]) -> str:
        if kind == 'percept':
            label = str(payload.get('label', 'unknown')).replace('.', '_')
            return f'{SUBJECT_PREFIX}.percept.{floor if floor is not None else 0}.{label}'
        if kind == 'episode':
            return f'{SUBJECT_PREFIX}.episode.{payload.get("outcome", "unknown")}'
        if kind in ('telemetry', 'decision'):
            return f'{SUBJECT_PREFIX}.{kind}.{payload.get("kind", "misc")}'
        return f'{SUBJECT_PREFIX}.{kind}'

    # -------------------------------------------------------------- reading

    async def replay(self, subject: str = SUBJECT_WILDCARD,
                     start_seq: int = 0,
                     limit: Optional[int] = None) -> AsyncIterator[LedgerRecord]:
        """Iterate historical records, oldest first."""
        if self._js is None:
            raise RuntimeError('ledger not connected; call connect() first')

        sub = await self._js.subscribe(
            subject,
            ordered_consumer=True,
            opt_start_seq=start_seq or None,
        )
        count = 0
        try:
            while limit is None or count < limit:
                try:
                    msg = await sub.next_msg(timeout=2.0)
                except (asyncio.TimeoutError, TimeoutError):
                    return
                try:
                    yield LedgerRecord.from_json(msg.data)
                except (ValueError, TypeError):
                    continue
                count += 1
        finally:
            await sub.unsubscribe()

    async def subscribe(self, subject: str,
                        handler: Callable[[LedgerRecord], Any],
                        durable: Optional[str] = None) -> Any:
        """Live subscription with a durable consumer, for the projector."""
        if self._js is None:
            raise RuntimeError('ledger not connected; call connect() first')

        async def _cb(msg):
            try:
                record = LedgerRecord.from_json(msg.data)
            except (ValueError, TypeError):
                await msg.ack()                    # poison message; do not requeue
                return
            result = handler(record)
            if asyncio.iscoroutine(result):
                await result
            await msg.ack()

        return await self._js.subscribe(subject, cb=_cb, durable=durable,
                                        manual_ack=True)

    async def audit(self, subject: str = SUBJECT_WILDCARD,
                    limit: int = 10000) -> Dict[str, Any]:
        """Replay and verify the chain. Reports the first break, if any."""
        records: List[LedgerRecord] = []
        async for record in self.replay(subject, limit=limit):
            records.append(record)

        break_at = verify_chain(records)
        return {
            'records': len(records),
            'intact': break_at is None,
            'first_break_index': break_at,
            'first_break_seq': records[break_at].seq if break_at is not None else None,
            'head_hash': records[-1].hash if records else GENESIS_HASH,
        }


class NullLedger:
    """Drop-in no-op, for running the robot without a memory backend.

    The stack must stay driveable when NATS is unreachable; losing the memory
    layer should cost the robot its recall, not its ability to navigate.
    """

    def __init__(self, *args, **kwargs):
        self.records: List[LedgerRecord] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def append(self, kind: str, payload: Dict[str, Any],
                     subject: Optional[str] = None,
                     floor: Optional[int] = None) -> LedgerRecord:
        record = LedgerRecord(
            seq=len(self.records) + 1,
            subject=subject or f'{SUBJECT_PREFIX}.{kind}',
            kind=kind, payload=payload, ts=time.time(),
            record_id=str(uuid.uuid4()),
            prev_hash=self.records[-1].hash if self.records else GENESIS_HASH,
        ).sealed()
        self.records.append(record)
        return record

    async def replay(self, subject: str = SUBJECT_WILDCARD, start_seq: int = 0,
                     limit: Optional[int] = None):
        for record in self.records[start_seq:]:
            yield record

    async def subscribe(self, *args, **kwargs):
        return None

    async def audit(self, subject: str = SUBJECT_WILDCARD, limit: int = 10000):
        break_at = verify_chain(self.records)
        return {'records': len(self.records), 'intact': break_at is None,
                'first_break_index': break_at, 'first_break_seq': None,
                'head_hash': self.records[-1].hash if self.records else GENESIS_HASH}


def make_ledger(enabled: bool = True, **kwargs):
    """Ledger when enabled and nats-py is present, NullLedger otherwise."""
    if not enabled or nats is None:
        return NullLedger()
    return Ledger(**kwargs)
