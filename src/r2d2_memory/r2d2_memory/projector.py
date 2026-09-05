#!/usr/bin/env python3
"""
Ledger -> Postgres projector.

Drains the JetStream ledger and folds each event into the pgvector store. This
is the only writer to the derived tables, which is what makes the projection
rebuildable: drop the database, replay the stream, get the same state back.

    python3 -m r2d2_memory.projector                 # follow live
    python3 -m r2d2_memory.projector --rebuild       # wipe and replay all
    python3 -m r2d2_memory.projector --audit         # verify the hash chain

Idempotency matters because JetStream delivers at least once. Every handler
either upserts or is naturally convergent, and the checkpoint records the last
sequence applied so a restart resumes rather than double-counting.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from typing import Any, Dict, Optional

from r2d2_memory.ledger import STREAM_NAME, SUBJECT_WILDCARD, Ledger, LedgerRecord
from r2d2_memory.store import MemoryStore

log = logging.getLogger('r2d2.projector')


class Projector:

    def __init__(self, ledger: Ledger, store: MemoryStore,
                 stream: str = STREAM_NAME):
        self.ledger = ledger
        self.store = store
        self.stream = stream
        self._applied = 0
        self._skipped = 0
        self._last_seq = 0

    async def handle(self, record: LedgerRecord) -> None:
        """Fold one ledger record into the store."""
        if record.seq <= self._last_seq:
            # Redelivery of something already applied.
            self._skipped += 1
            return

        try:
            handler = getattr(self, f'_on_{record.kind}', None)
            if handler is None:
                self._skipped += 1
            else:
                await handler(record)
                self._applied += 1
        except Exception as exc:                  # noqa: BLE001 - keep draining
            # A single malformed event must not stall the whole projection.
            # It is logged with its sequence so it can be found in the ledger,
            # which still holds the original.
            log.error('failed to project seq %d (%s): %s',
                      record.seq, record.subject, exc)
            self._skipped += 1

        self._last_seq = record.seq
        if self._last_seq % 50 == 0:
            await self.store.set_checkpoint(self.stream, record.seq, record.hash)

    # ---------------------------------------------------------- event kinds

    async def _on_percept(self, record: LedgerRecord) -> None:
        p: Dict[str, Any] = record.payload
        pose = p.get('robot_pose') or {}
        await self.store.record_observation(
            label=p['label'],
            floor=int(p.get('floor', 0)),
            world_x=float(p['world_x']),
            world_y=float(p['world_y']),
            confidence=float(p.get('confidence', 0.5)),
            robot_pose=(float(pose.get('x', 0.0)),
                        float(pose.get('y', 0.0)),
                        float(pose.get('yaw', 0.0))),
            bearing=float(p.get('bearing', 0.0)),
            range_m=p.get('range_m'),
            description=p.get('description'),
            attributes=p.get('attributes'),
            source=p.get('source', 'vla'),
            ledger_seq=record.seq,
        )

    async def _on_episode(self, record: LedgerRecord) -> None:
        p = record.payload
        await self.store.record_episode(
            instruction=p['instruction'],
            outcome=p.get('outcome', 'success'),
            summary=p.get('summary'),
            failure_reason=p.get('failure_reason'),
            tool_calls=p.get('tool_calls'),
            route=p.get('route'),
            start_floor=p.get('start_floor'),
            end_floor=p.get('end_floor'),
            duration_s=p.get('duration_s'),
        )

    async def _on_fact(self, record: LedgerRecord) -> None:
        p = record.payload
        await self.store.remember(
            content=p['content'],
            floor=p.get('floor'),
            source=p.get('source', 'agent'),
        )

    async def _on_place(self, record: LedgerRecord) -> None:
        p = record.payload
        extent = None
        if all(k in p for k in ('min_x', 'min_y', 'max_x', 'max_y')):
            extent = (p['min_x'], p['min_y'], p['max_x'], p['max_y'])
        await self.store.upsert_place(
            name=p['name'], floor=int(p.get('floor', 0)),
            x=float(p['x']), y=float(p['y']), yaw=float(p.get('yaw', 0.0)),
            extent=extent, description=p.get('description'),
        )

    # Telemetry and decisions stay in the ledger only. They are high volume and
    # nothing queries them by similarity; keeping them out of Postgres is what
    # stops the projection from becoming a second copy of the whole stream.
    async def _on_telemetry(self, record: LedgerRecord) -> None:
        self._skipped += 1

    async def _on_decision(self, record: LedgerRecord) -> None:
        self._skipped += 1

    # ------------------------------------------------------------- entry points

    async def rebuild(self) -> Dict[str, int]:
        """Wipe the projection and replay the entire ledger into it."""
        log.warning('rebuilding projection from sequence 0')
        await self.store.reset_projection()
        self._last_seq = 0
        async for record in self.ledger.replay(SUBJECT_WILDCARD):
            await self.handle(record)
        if self._last_seq:
            await self.store.set_checkpoint(self.stream, self._last_seq, '')
        log.info('rebuild complete: %d applied, %d skipped',
                 self._applied, self._skipped)
        return {'applied': self._applied, 'skipped': self._skipped}

    async def follow(self, stop: Optional[asyncio.Event] = None) -> None:
        """Resume from the checkpoint and then stay live."""
        self._last_seq, _ = await self.store.get_checkpoint(self.stream)
        log.info('following ledger from sequence %d', self._last_seq)

        await self.ledger.subscribe(SUBJECT_WILDCARD, self.handle,
                                    durable='r2d2-projector')
        stop = stop or asyncio.Event()
        await stop.wait()


async def _main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    ledger = Ledger(servers=args.nats)
    store = MemoryStore(dsn=args.dsn)

    await ledger.connect()
    await store.connect()
    if args.apply_schema:
        await store.apply_schema()

    try:
        if args.audit:
            result = await ledger.audit()
            print(f'ledger records : {result["records"]}')
            print(f'chain intact   : {result["intact"]}')
            if not result['intact']:
                print(f'first break at : sequence {result["first_break_seq"]} '
                      f'(index {result["first_break_index"]})')
                print('A break means a record was altered or removed after it '
                      'was written. The ledger is append-only by design, so '
                      'this is worth investigating rather than repairing.')
                return 1
            print(f'head hash      : {result["head_hash"]}')
            return 0

        projector = Projector(ledger, store)
        if args.rebuild:
            await projector.rebuild()
            return 0

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await projector.follow(stop)
        return 0
    finally:
        await store.close()
        await ledger.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nats', default=os.environ.get('R2D2_NATS_URL',
                                                         'nats://127.0.0.1:4222'))
    parser.add_argument('--dsn', default=os.environ.get('R2D2_PG_DSN'))
    parser.add_argument('--rebuild', action='store_true',
                        help='wipe the projection and replay the whole ledger')
    parser.add_argument('--audit', action='store_true',
                        help='verify the ledger hash chain and exit')
    parser.add_argument('--apply-schema', action='store_true',
                        help='create the schema before starting')
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
