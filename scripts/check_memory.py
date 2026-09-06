#!/usr/bin/env python3
"""
End-to-end check of the memory stack against real NATS and Postgres.

    docker compose -f src/r2d2_memory/docker-compose.yml up -d
    python3 scripts/check_memory.py
    python3 scripts/check_memory.py --keep     # leave the test rows behind

The unit tests cover the maths - hash chaining, merge radius, position fusion -
without a database. This covers the things only a real backend can answer:

  1. JetStream accepts the stream config and the hash chain survives a
     round trip through it
  2. The chain resumes correctly after a reconnect, rather than restarting from
     the genesis hash and reporting a false break
  3. The Postgres schema applies cleanly, including both pgvector extensions
  4. A vector query returns rows in similarity order
  5. Repeated observations of one object merge instead of multiplying
  6. Two nearby objects with the same label stay separate

Everything it writes is namespaced and removed afterwards unless --keep is
passed, so it is safe to run against a live robot's database.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'r2d2_memory'))

PASS, FAIL = '  ok  ', ' FAIL '
_failures = 0


def check(name: str, ok: bool, detail: str = ''):
    global _failures
    print(f'[{PASS if ok else FAIL}] {name}' + (f' - {detail}' if detail else ''))
    if not ok:
        _failures += 1
    return ok


async def check_ledger(nats_url: str) -> bool:
    from r2d2_memory.ledger import GENESIS_HASH, Ledger, verify_chain

    stream = f'R2D2CHECK{uuid.uuid4().hex[:8].upper()}'
    ledger = Ledger(servers=nats_url, stream=stream, max_age_s=3600,
                    max_bytes=64 * 1024 * 1024)
    try:
        await ledger.connect()
    except Exception as exc:                       # noqa: BLE001
        check('nats reachable', False, str(exc))
        return False
    check('nats reachable', True, nats_url)

    written = []
    for i in range(25):
        written.append(await ledger.append(
            'percept', {'label': f'probe_{i}', 'world_x': float(i),
                        'world_y': 0.0, 'confidence': 0.7}, floor=0))
    check('appended 25 records', len(written) == 25)
    check('first record starts the chain', written[0].prev_hash == GENESIS_HASH)
    check('records link to their predecessor',
          all(b.prev_hash == a.hash for a, b in zip(written, written[1:])))

    replayed = [r async for r in ledger.replay(limit=25)]
    check('replayed everything back', len(replayed) == 25,
          f'{len(replayed)} of 25')
    check('replayed records verify individually',
          all(r.verify() for r in replayed))
    check('replayed chain is intact', verify_chain(replayed) is None)

    audit = await ledger.audit(limit=100)
    check('audit reports the chain intact', audit['intact'],
          f'{audit["records"]} records')

    # A fresh client must pick the chain up where the last one left it, not
    # start a second chain from the genesis hash.
    await ledger.close()
    resumed = Ledger(servers=nats_url, stream=stream)
    await resumed.connect()
    continuation = await resumed.append('fact', {'content': 'after reconnect'})
    check('chain resumes across a reconnect',
          continuation.prev_hash == written[-1].hash,
          f'seq {continuation.seq} follows {written[-1].seq}')

    try:
        await resumed._js.delete_stream(stream)
    except Exception:                              # noqa: BLE001
        pass
    await resumed.close()
    return True


async def check_store(dsn: str, keep: bool) -> bool:
    from r2d2_memory.store import MemoryStore, merge_radius

    store = MemoryStore(dsn=dsn)
    try:
        await store.connect()
    except Exception as exc:                       # noqa: BLE001
        check('postgres reachable', False, str(exc))
        return False
    check('postgres reachable', True, dsn.split('@')[-1])

    try:
        await store.apply_schema()
        check('schema applies', True)
    except Exception as exc:                       # noqa: BLE001
        check('schema applies', False, str(exc))
        await store.close()
        return False

    async with store._pool.acquire() as conn:
        extensions = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector','pgcrypto')")
    check('pgvector and pgcrypto are installed', len(extensions) == 2,
          ', '.join(r['extname'] for r in extensions))

    tag = f'probe{uuid.uuid4().hex[:6]}'
    floor = 900                                    # far from any real floor

    # --- fusion: one object seen repeatedly must stay one object ------------
    ids = set()
    for i in range(8):
        ids.add(await store.record_observation(
            label=f'{tag}_chair', floor=floor,
            world_x=2.0 + 0.04 * math.cos(i), world_y=4.0 + 0.04 * math.sin(i),
            confidence=0.7, robot_pose=(0.0, 0.0, 0.0), bearing=0.1,
            range_m=2.0, description='a wooden dining chair'))
    check('repeat observations merge into one object', len(ids) == 1,
          f'{len(ids)} object(s) from 8 observations')

    hits = await store.list_objects(floor=floor)
    chair = next((h for h in hits if h.label == f'{tag}_chair'), None)
    check('the merged object counted every observation',
          chair is not None and chair.observation_count == 8,
          f'count {chair.observation_count if chair else "-"}')
    check('confidence rose with repeat sightings',
          chair is not None and chair.confidence > 0.7,
          f'{chair.confidence:.3f}' if chair else '')
    check('position converged near the truth',
          chair is not None and math.dist((chair.x, chair.y), (2.0, 4.0)) < 0.1,
          f'({chair.x:.3f}, {chair.y:.3f})' if chair else '')

    # --- separation: two chairs a metre apart must stay two chairs ----------
    far_id = await store.record_observation(
        label=f'{tag}_chair', floor=floor, world_x=2.0 + 1.5, world_y=4.0,
        confidence=0.7, robot_pose=(0.0, 0.0, 0.0), bearing=0.0, range_m=3.0)
    check('a distant same-label detection stays separate',
          far_id not in ids,
          f'merge radius at 8 observations is {merge_radius(8):.2f} m')

    # --- vector search ------------------------------------------------------
    await store.record_observation(
        label=f'{tag}_fridge', floor=floor, world_x=0.5, world_y=6.4,
        confidence=0.8, robot_pose=(0.0, 0.0, 0.0), bearing=-0.3, range_m=4.0,
        description='a tall white kitchen refrigerator')

    results = await store.find_objects(f'{tag}_fridge', floor=floor, limit=3)
    check('vector search returns rows', bool(results),
          f'{len(results)} hit(s)')
    check('the best hit is the right object',
          bool(results) and results[0].label == f'{tag}_fridge',
          results[0].label if results else '')
    check('results carry a similarity score',
          bool(results) and results[0].similarity is not None)
    check('results are ordered by similarity',
          len(results) < 2 or all(
              a.similarity >= b.similarity
              for a, b in zip(results, results[1:])))

    # --- episodes -----------------------------------------------------------
    await store.record_episode(
        instruction=f'{tag} go to the kitchen and check the table',
        outcome='failure', summary='could not get there',
        failure_reason='the study door was shut')
    episodes = await store.recall_episodes(f'{tag} go to the kitchen', limit=2)
    check('episodic recall finds the attempt', bool(episodes))
    check('failures are recalled, not just successes',
          bool(episodes) and episodes[0]['outcome'] == 'failure',
          episodes[0]['failure_reason'] if episodes else '')

    # --- facts --------------------------------------------------------------
    await store.remember(f'{tag} the study door sticks and needs a firm push')
    facts = await store.recall(f'{tag} study door', limit=2)
    check('free-text recall works', bool(facts))

    if not keep:
        async with store._pool.acquire() as conn:
            await conn.execute('DELETE FROM semantic_object WHERE floor = $1', floor)
            await conn.execute("DELETE FROM episode WHERE instruction LIKE $1",
                               f'{tag}%')
            await conn.execute("DELETE FROM fact WHERE content LIKE $1", f'{tag}%')
        check('test rows cleaned up', True)
    else:
        print(f'       (left behind: floor {floor}, tag {tag})')

    await store.close()
    return True


async def _main(args) -> int:
    print('memory stack check\n')
    await check_ledger(args.nats)
    print()
    await check_store(args.dsn, args.keep)
    print()
    if _failures:
        print(f'{_failures} check(s) failed')
    else:
        print('all checks passed')
    return 1 if _failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nats', default=os.environ.get(
        'R2D2_NATS_URL', 'nats://127.0.0.1:4222'))
    parser.add_argument('--dsn', default=os.environ.get(
        'R2D2_PG_DSN', 'postgresql://r2d2:r2d2@127.0.0.1:5432/r2d2'))
    parser.add_argument('--keep', action='store_true',
                        help='leave the test rows in the database')
    return asyncio.run(_main(parser.parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())
