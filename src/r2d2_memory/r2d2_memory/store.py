#!/usr/bin/env python3
"""
Postgres + pgvector store: the queryable projection of the ledger.

This is where "what does the robot know" is answered. It is not the source of
truth - every row was derived from a ledger event and the whole database can be
rebuilt by replaying the stream (projector.py) - which is what makes it safe to
change the fusion rules below and simply reproject.

The interesting logic is object fusion. A VLA looking at the same chair from
four angles produces four detections with different positions and confidences.
Storing them as four chairs makes the map useless. Merging anything with the
same label makes two dining chairs into one. The rule used here:

    a detection joins an existing object when it has the same label, is on the
    same floor, and lies within a merge radius that shrinks as the object
    becomes better observed

so an object seen once is generous about accepting nearby evidence, and an
object seen fifty times is not. Position is then a confidence-weighted running
mean, and the spread of contributing detections is kept as position_sigma - a
large sigma is the signal that a merge was probably wrong.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import asyncpg
except ImportError:  # pragma: no cover - exercised only without the dependency
    asyncpg = None

from r2d2_memory.embeddings import EMBED_DIM, Embedder, make_embedder

log = logging.getLogger('r2d2.store')

DEFAULT_DSN = 'postgresql://r2d2:r2d2@127.0.0.1:5432/r2d2'

# Merge radius bounds, metres. The upper bound is roughly the positional error
# of a bearing-plus-LiDAR grounding at 3 m; the lower bound stops a
# well-observed object from swallowing its neighbour on a table.
MERGE_RADIUS_MAX = 0.90
MERGE_RADIUS_MIN = 0.25


@dataclass
class ObjectHit:
    """One semantic object returned from a query."""

    id: str
    label: str
    description: Optional[str]
    floor: int
    place: Optional[str]
    x: float
    y: float
    confidence: float
    observation_count: int
    position_sigma: float
    distance: Optional[float] = None      # metres from the robot, when known
    similarity: Optional[float] = None    # cosine similarity, when a vector query

    def as_dict(self) -> Dict[str, Any]:
        out = {
            'id': self.id, 'label': self.label, 'description': self.description,
            'floor': self.floor, 'place': self.place,
            'x': round(self.x, 3), 'y': round(self.y, 3),
            'confidence': round(self.confidence, 3),
            'observations': self.observation_count,
            'position_sigma': round(self.position_sigma, 3),
        }
        if self.distance is not None:
            out['distance_m'] = round(self.distance, 2)
        if self.similarity is not None:
            out['similarity'] = round(self.similarity, 3)
        return out


def merge_radius(observation_count: int) -> float:
    """Shrinking association gate.

    One observation: accept evidence up to MERGE_RADIUS_MAX away, because the
    single estimate could be off by that much. Many observations: tighten
    towards MERGE_RADIUS_MIN, because the estimate is now good and a detection
    that far away is probably a different object.
    """
    n = max(observation_count, 1)
    radius = MERGE_RADIUS_MAX / math.sqrt(n)
    return max(MERGE_RADIUS_MIN, min(MERGE_RADIUS_MAX, radius))


def fuse_position(old_x: float, old_y: float, old_weight: float,
                  new_x: float, new_y: float, new_weight: float
                  ) -> Tuple[float, float, float]:
    """Confidence-weighted running mean of two position estimates."""
    total = old_weight + new_weight
    if total <= 0.0:
        return new_x, new_y, new_weight
    return ((old_x * old_weight + new_x * new_weight) / total,
            (old_y * old_weight + new_y * new_weight) / total,
            total)


def _vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text input format."""
    return '[' + ','.join(f'{v:.6f}' for v in vector) + ']'


class MemoryStore:
    """Async pgvector-backed semantic and episodic memory."""

    def __init__(self, dsn: Optional[str] = None,
                 embedder: Optional[Embedder] = None,
                 min_pool: int = 1, max_pool: int = 8):
        self.dsn = dsn or os.environ.get('R2D2_PG_DSN', DEFAULT_DSN)
        self.embedder = embedder or make_embedder()
        self._pool = None
        self._min_pool = min_pool
        self._max_pool = max_pool

    # ------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        if asyncpg is None:
            raise RuntimeError(
                'asyncpg is not installed. pip install asyncpg, or run with '
                'R2D2_MEMORY=off.')
        self._pool = await asyncpg.create_pool(
            self.dsn, min_size=self._min_pool, max_size=self._max_pool)
        log.info('memory store connected (%s)', _redact(self.dsn))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def apply_schema(self, path: Optional[str] = None) -> None:
        """Create the schema if it is not already there."""
        path = path or os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(path) as fh:
            sql = fh.read()
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        log.info('schema applied from %s', path)

    # ------------------------------------------------------- semantic objects

    async def record_observation(self, label: str, floor: int,
                                 world_x: float, world_y: float,
                                 confidence: float,
                                 robot_pose: Tuple[float, float, float],
                                 bearing: float,
                                 range_m: Optional[float] = None,
                                 description: Optional[str] = None,
                                 attributes: Optional[Dict[str, Any]] = None,
                                 source: str = 'vla',
                                 ledger_seq: Optional[int] = None) -> str:
        """Fold one grounded detection into the object map. Returns object id."""
        text = f'{label}: {description}' if description else label
        embedding = _vector_literal(self.embedder.encode_one(text))

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                existing = await self._find_mergeable(conn, label, floor,
                                                      world_x, world_y)
                if existing is None:
                    object_id = await conn.fetchval(
                        """
                        INSERT INTO semantic_object
                            (label, description, floor, x, y, confidence,
                             observation_count, position_sigma, embedding,
                             attributes, place_id)
                        VALUES ($1, $2, $3, $4, $5, $6, 1, $7, $8::vector, $9,
                                (SELECT id FROM place
                                  WHERE floor = $3
                                    AND $4 BETWEEN COALESCE(min_x, -1e9) AND COALESCE(max_x, 1e9)
                                    AND $5 BETWEEN COALESCE(min_y, -1e9) AND COALESCE(max_y, 1e9)
                                  LIMIT 1))
                        RETURNING id
                        """,
                        label, description, floor, world_x, world_y,
                        confidence, MERGE_RADIUS_MAX, embedding,
                        json.dumps(attributes or {}))
                else:
                    object_id = await self._merge_into(
                        conn, existing, world_x, world_y, confidence,
                        description, embedding)

                await conn.execute(
                    """
                    INSERT INTO observation
                        (object_id, label, floor, robot_x, robot_y, robot_yaw,
                         bearing, range_m, world_x, world_y, confidence,
                         source, ledger_seq)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    """,
                    object_id, label, floor, robot_pose[0], robot_pose[1],
                    robot_pose[2], bearing, range_m, world_x, world_y,
                    confidence, source, ledger_seq)

        return str(object_id)

    @staticmethod
    async def _find_mergeable(conn, label: str, floor: int,
                              x: float, y: float) -> Optional[Dict[str, Any]]:
        """Nearest same-label object on this floor inside its own merge gate."""
        rows = await conn.fetch(
            """
            SELECT id, x, y, confidence, observation_count, position_sigma
            FROM semantic_object
            WHERE label = $1 AND floor = $2
            ORDER BY (x - $3) * (x - $3) + (y - $4) * (y - $4)
            LIMIT 4
            """, label, floor, x, y)

        for row in rows:
            distance = math.dist((row['x'], row['y']), (x, y))
            if distance <= merge_radius(row['observation_count']):
                return dict(row)
        return None

    @staticmethod
    async def _merge_into(conn, existing: Dict[str, Any],
                          x: float, y: float, confidence: float,
                          description: Optional[str], embedding: str) -> Any:
        """Update an object with a new detection."""
        old_weight = existing['confidence'] * existing['observation_count']
        new_x, new_y, _ = fuse_position(
            existing['x'], existing['y'], old_weight, x, y, confidence)

        # Running estimate of how far detections sit from the fused centre. A
        # growing sigma means this object is absorbing things it should not.
        offset = math.dist((existing['x'], existing['y']), (x, y))
        n = existing['observation_count']
        sigma = math.sqrt((existing['position_sigma'] ** 2 * n + offset ** 2) / (n + 1))

        return await conn.fetchval(
            """
            UPDATE semantic_object
            SET x = $2,
                y = $3,
                position_sigma = $4,
                observation_count = observation_count + 1,
                -- Asymptotic, so repeated sightings raise confidence without
                -- ever letting a noisy detector reach certainty.
                confidence = LEAST(0.99, confidence + (1.0 - confidence) * $5),
                description = COALESCE($6, description),
                embedding = $7::vector,
                last_seen = now()
            WHERE id = $1
            RETURNING id
            """,
            existing['id'], new_x, new_y, sigma, confidence * 0.3,
            description, embedding)

    async def find_objects(self, query: str, floor: Optional[int] = None,
                           limit: int = 5,
                           min_confidence: float = 0.2,
                           robot_xy: Optional[Tuple[float, float]] = None
                           ) -> List[ObjectHit]:
        """Semantic search over the object map.

        Cosine distance on the label+description embedding, so "somewhere to
        sit" can retrieve a chair. Restricted by floor when given, because
        directing the robot to an object one storey up without routing it there
        is the single easiest way to produce a confidently wrong plan.
        """
        embedding = _vector_literal(self.embedder.encode_one(query))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT o.id, o.label, o.description, o.floor, p.name AS place,
                       o.x, o.y, o.confidence, o.observation_count,
                       o.position_sigma,
                       1 - (o.embedding <=> $1::vector) AS similarity
                FROM semantic_object o
                LEFT JOIN place p ON p.id = o.place_id
                WHERE o.confidence >= $2
                  AND ($3::int IS NULL OR o.floor = $3)
                  AND o.embedding IS NOT NULL
                ORDER BY o.embedding <=> $1::vector
                LIMIT $4
                """, embedding, min_confidence, floor, limit)

        hits = [ObjectHit(
            id=str(r['id']), label=r['label'], description=r['description'],
            floor=r['floor'], place=r['place'], x=r['x'], y=r['y'],
            confidence=r['confidence'],
            observation_count=r['observation_count'],
            position_sigma=r['position_sigma'],
            similarity=r['similarity'],
        ) for r in rows]

        if robot_xy is not None:
            for hit in hits:
                hit.distance = math.dist(robot_xy, (hit.x, hit.y))
        return hits

    async def list_objects(self, floor: Optional[int] = None,
                           limit: int = 50) -> List[ObjectHit]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT o.id, o.label, o.description, o.floor, p.name AS place,
                       o.x, o.y, o.confidence, o.observation_count,
                       o.position_sigma
                FROM semantic_object o
                LEFT JOIN place p ON p.id = o.place_id
                WHERE ($1::int IS NULL OR o.floor = $1)
                ORDER BY o.observation_count DESC, o.last_seen DESC
                LIMIT $2
                """, floor, limit)
        return [ObjectHit(
            id=str(r['id']), label=r['label'], description=r['description'],
            floor=r['floor'], place=r['place'], x=r['x'], y=r['y'],
            confidence=r['confidence'],
            observation_count=r['observation_count'],
            position_sigma=r['position_sigma'],
        ) for r in rows]

    # -------------------------------------------------------------- episodes

    async def record_episode(self, instruction: str, outcome: str,
                             summary: Optional[str] = None,
                             failure_reason: Optional[str] = None,
                             tool_calls: Optional[List[Dict]] = None,
                             route: Optional[List[Dict]] = None,
                             start_floor: Optional[int] = None,
                             end_floor: Optional[int] = None,
                             duration_s: Optional[float] = None) -> str:
        text = f'{instruction} -> {outcome}: {summary or ""}'
        embedding = _vector_literal(self.embedder.encode_one(text))
        async with self._pool.acquire() as conn:
            episode_id = await conn.fetchval(
                """
                INSERT INTO episode
                    (instruction, outcome, summary, failure_reason, tool_calls,
                     route, start_floor, end_floor, duration_s, embedding,
                     ended_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::vector, now())
                RETURNING id
                """,
                instruction, outcome, summary, failure_reason,
                json.dumps(tool_calls or []), json.dumps(route or []),
                start_floor, end_floor, duration_s, embedding)
        return str(episode_id)

    async def recall_episodes(self, instruction: str, limit: int = 3,
                              include_failures: bool = True) -> List[Dict[str, Any]]:
        """Past attempts at something similar.

        Failures are included by default and that is deliberate. The old sqlite
        version stored only successes and then retrieved the most recent one
        regardless of the query, which meant the planner was shown an unrelated
        example and encouraged to imitate it. Knowing that the last attempt at
        this instruction failed because the study door was shut is worth more
        than a success at a different task.
        """
        embedding = _vector_literal(self.embedder.encode_one(instruction))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, instruction, outcome, summary, failure_reason,
                       route, start_floor, end_floor, duration_s, started_at,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM episode
                WHERE embedding IS NOT NULL
                  AND ($2 OR outcome = 'success')
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """, embedding, include_failures, limit)
        return [{
            'id': str(r['id']),
            'instruction': r['instruction'],
            'outcome': r['outcome'],
            'summary': r['summary'],
            'failure_reason': r['failure_reason'],
            'route': json.loads(r['route']) if r['route'] else [],
            'start_floor': r['start_floor'],
            'end_floor': r['end_floor'],
            'duration_s': r['duration_s'],
            'when': r['started_at'].isoformat() if r['started_at'] else None,
            'similarity': round(r['similarity'], 3),
        } for r in rows]

    # ------------------------------------------------------------------ facts

    async def remember(self, content: str, floor: Optional[int] = None,
                       source: str = 'agent') -> str:
        embedding = _vector_literal(self.embedder.encode_one(content))
        async with self._pool.acquire() as conn:
            fact_id = await conn.fetchval(
                """
                INSERT INTO fact (content, source, floor, embedding)
                VALUES ($1, $2, $3, $4::vector)
                RETURNING id
                """, content, source, floor, embedding)
        return str(fact_id)

    async def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        embedding = _vector_literal(self.embedder.encode_one(query))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, source, floor, created_at,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM fact
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """, embedding, limit)
        return [{
            'id': str(r['id']), 'content': r['content'], 'source': r['source'],
            'floor': r['floor'],
            'when': r['created_at'].isoformat() if r['created_at'] else None,
            'similarity': round(r['similarity'], 3),
        } for r in rows]

    # ------------------------------------------------------------------ places

    async def upsert_place(self, name: str, floor: int, x: float, y: float,
                           yaw: float = 0.0,
                           extent: Optional[Tuple[float, float, float, float]] = None,
                           description: Optional[str] = None) -> str:
        embedding = _vector_literal(
            self.embedder.encode_one(f'{name}: {description or ""}'))
        min_x, min_y, max_x, max_y = extent or (None, None, None, None)
        async with self._pool.acquire() as conn:
            place_id = await conn.fetchval(
                """
                INSERT INTO place
                    (name, floor, x, y, yaw, min_x, min_y, max_x, max_y,
                     description, embedding)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::vector)
                ON CONFLICT (name, floor) DO UPDATE
                SET x = EXCLUDED.x, y = EXCLUDED.y, yaw = EXCLUDED.yaw,
                    min_x = EXCLUDED.min_x, min_y = EXCLUDED.min_y,
                    max_x = EXCLUDED.max_x, max_y = EXCLUDED.max_y,
                    description = COALESCE(EXCLUDED.description, place.description),
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                RETURNING id
                """, name, floor, x, y, yaw, min_x, min_y, max_x, max_y,
                description, embedding)
        return str(place_id)

    async def find_place(self, query: str,
                         floor: Optional[int] = None) -> Optional[Dict[str, Any]]:
        embedding = _vector_literal(self.embedder.encode_one(query))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, floor, x, y, yaw, description,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM place
                WHERE ($2::int IS NULL OR floor = $2)
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT 1
                """, embedding, floor)
        if row is None:
            return None
        return {'id': str(row['id']), 'name': row['name'], 'floor': row['floor'],
                'x': row['x'], 'y': row['y'], 'yaw': row['yaw'],
                'description': row['description'],
                'similarity': round(row['similarity'], 3)}

    # ------------------------------------------------------------ checkpoint

    async def get_checkpoint(self, stream: str) -> Tuple[int, str]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT last_sequence, last_hash FROM ledger_checkpoint WHERE stream = $1',
                stream)
        return (row['last_sequence'], row['last_hash']) if row else (0, '0' * 64)

    async def set_checkpoint(self, stream: str, sequence: int, hash_: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger_checkpoint (stream, last_sequence, last_hash)
                VALUES ($1, $2, $3)
                ON CONFLICT (stream) DO UPDATE
                SET last_sequence = EXCLUDED.last_sequence,
                    last_hash = EXCLUDED.last_hash,
                    updated_at = now()
                """, stream, sequence, hash_)

    async def reset_projection(self) -> None:
        """Drop the derived state so the ledger can be replayed from scratch."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                'TRUNCATE observation, semantic_object, episode, fact, '
                'ledger_checkpoint RESTART IDENTITY CASCADE')
        log.warning('projection reset; replay the ledger to rebuild it')


def _redact(dsn: str) -> str:
    if '@' not in dsn:
        return dsn
    scheme, rest = dsn.split('://', 1) if '://' in dsn else ('', dsn)
    _, host = rest.split('@', 1)
    return f'{scheme}://***@{host}'
