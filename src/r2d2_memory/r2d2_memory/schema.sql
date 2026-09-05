-- R2D2-Redux v2 semantic and episodic memory.
--
-- This database is a PROJECTION, not the source of truth. Every row here was
-- derived from an event on the NATS JetStream ledger (see ledger.py), and the
-- whole schema can be dropped and rebuilt by replaying the stream. That is the
-- point of the split: the ledger is append-only and tamper-evident, while this
-- side is free to be reshaped whenever the queries change.
--
-- Apply with:
--     psql "$R2D2_PG_DSN" -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- --------------------------------------------------------------------------
-- Ledger checkpoint
--
-- Where the projector has read up to, so a restart resumes instead of
-- replaying from the beginning, and so a mismatch between the stored hash and
-- the stream's is visible rather than silent.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ledger_checkpoint (
    stream          TEXT PRIMARY KEY,
    last_sequence   BIGINT      NOT NULL,
    last_hash       TEXT        NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------
-- Places: rooms and named locations, one row per floor per place.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS place (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT        NOT NULL,
    floor           INTEGER     NOT NULL,
    x               DOUBLE PRECISION NOT NULL,
    y               DOUBLE PRECISION NOT NULL,
    yaw             DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- Axis-aligned extent, used to decide whether a detection falls inside a
    -- room without needing full polygon containment.
    min_x           DOUBLE PRECISION,
    min_y           DOUBLE PRECISION,
    max_x           DOUBLE PRECISION,
    max_y           DOUBLE PRECISION,
    description     TEXT,
    embedding       vector(384),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, floor)
);

-- --------------------------------------------------------------------------
-- Semantic objects: what the VLA has grounded into the map.
--
-- One row per persistent object instance, not per detection. Detections
-- accumulate into an instance via observation_count and a running position
-- average, so a chair seen forty times is one chair with a tight estimate
-- rather than forty chairs.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_object (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label           TEXT        NOT NULL,
    description     TEXT,
    floor           INTEGER     NOT NULL,
    place_id        UUID        REFERENCES place(id) ON DELETE SET NULL,

    x               DOUBLE PRECISION NOT NULL,
    y               DOUBLE PRECISION NOT NULL,
    -- Positional spread across observations. A large value means the detections
    -- disagree, which usually means two different objects have been merged.
    position_sigma  DOUBLE PRECISION NOT NULL DEFAULT 0.5,

    confidence      REAL        NOT NULL DEFAULT 0.5,
    observation_count INTEGER   NOT NULL DEFAULT 1,
    first_seen      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Text embedding of "label: description", so "something to sit on" can
    -- retrieve a chair without an exact label match.
    embedding       vector(384),
    attributes      JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS semantic_object_floor_idx ON semantic_object (floor);
CREATE INDEX IF NOT EXISTS semantic_object_label_idx ON semantic_object (label);
CREATE INDEX IF NOT EXISTS semantic_object_last_seen_idx ON semantic_object (last_seen DESC);

-- IVFFlat over cosine distance. lists=100 suits the few thousand objects a
-- house produces; rebuild with a larger value only if the table grows by an
-- order of magnitude. ANALYZE after bulk loading or the planner will ignore it.
CREATE INDEX IF NOT EXISTS semantic_object_embedding_idx
    ON semantic_object USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- --------------------------------------------------------------------------
-- Observations: the raw grounded detections behind the objects above.
--
-- Kept so an object's position estimate can be recomputed, and so a bad merge
-- can be diagnosed after the fact. Pruned by age, not kept forever.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS observation (
    id              BIGSERIAL PRIMARY KEY,
    object_id       UUID        REFERENCES semantic_object(id) ON DELETE CASCADE,
    label           TEXT        NOT NULL,
    floor           INTEGER     NOT NULL,
    -- Robot pose at the moment of observation, needed to re-project later.
    robot_x         DOUBLE PRECISION NOT NULL,
    robot_y         DOUBLE PRECISION NOT NULL,
    robot_yaw       DOUBLE PRECISION NOT NULL,
    bearing         DOUBLE PRECISION NOT NULL,   -- radians, robot frame
    range_m         DOUBLE PRECISION,            -- NULL when LiDAR gave no return
    world_x         DOUBLE PRECISION,
    world_y         DOUBLE PRECISION,
    confidence      REAL        NOT NULL,
    source          TEXT        NOT NULL DEFAULT 'vla',
    ledger_seq      BIGINT,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS observation_object_idx ON observation (object_id);
CREATE INDEX IF NOT EXISTS observation_observed_at_idx ON observation (observed_at DESC);

-- --------------------------------------------------------------------------
-- Episodes: what the robot was asked to do and what actually happened.
--
-- This replaces the single-table navigation_tasks sqlite store in gpt_oss.py,
-- whose "find a similar task" was really "return the most recent successful
-- task" regardless of what was asked. Here similarity is an actual vector
-- search, and failures are kept as well as successes - a failed attempt is the
-- more useful memory when the same instruction comes round again.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS episode (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instruction     TEXT        NOT NULL,
    outcome         TEXT        NOT NULL CHECK (outcome IN ('success', 'failure', 'aborted', 'partial')),
    summary         TEXT,
    failure_reason  TEXT,
    start_floor     INTEGER,
    end_floor       INTEGER,
    -- The tool calls the agent actually made, in order. This is the trace an
    -- operator reads when asking "why did it do that".
    tool_calls      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    route           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    duration_s      DOUBLE PRECISION,
    embedding       vector(384),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS episode_outcome_idx ON episode (outcome);
CREATE INDEX IF NOT EXISTS episode_started_at_idx ON episode (started_at DESC);
CREATE INDEX IF NOT EXISTS episode_embedding_idx
    ON episode USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- --------------------------------------------------------------------------
-- Facts: durable free-text memory the agent writes for itself.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT        NOT NULL,
    source          TEXT        NOT NULL DEFAULT 'agent',
    floor           INTEGER,
    embedding       vector(384),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS fact_embedding_idx
    ON fact USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- --------------------------------------------------------------------------
-- Floor transitions, mirrored from floor_manager so a route can be planned
-- from the database without a live ROS graph.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS floor_transition (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_floor      INTEGER     NOT NULL,
    to_floor        INTEGER     NOT NULL,
    kind            TEXT        NOT NULL DEFAULT 'stairs',
    foot_x          DOUBLE PRECISION NOT NULL,
    foot_y          DOUBLE PRECISION NOT NULL,
    head_x          DOUBLE PRECISION NOT NULL,
    head_y          DOUBLE PRECISION NOT NULL,
    heading         DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- Climb reliability, updated from episode outcomes. A staircase the robot
    -- keeps failing on should stop being chosen when an alternative exists.
    attempts        INTEGER     NOT NULL DEFAULT 0,
    successes       INTEGER     NOT NULL DEFAULT 0,
    UNIQUE (from_floor, to_floor, kind)
);

-- --------------------------------------------------------------------------
-- Convenience view: objects with their room, ranked by how well observed.
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW object_directory AS
SELECT
    o.id,
    o.label,
    o.description,
    o.floor,
    p.name AS place,
    o.x,
    o.y,
    o.confidence,
    o.observation_count,
    o.position_sigma,
    o.last_seen
FROM semantic_object o
LEFT JOIN place p ON p.id = o.place_id
ORDER BY o.observation_count DESC, o.last_seen DESC;
