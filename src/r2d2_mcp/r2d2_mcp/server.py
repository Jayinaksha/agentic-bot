#!/usr/bin/env python3
"""
MCP server: the robot as a set of typed tools.

    python3 -m r2d2_mcp.server                  # stdio, for a local agent
    R2D2_MCP_TRANSPORT=http python3 -m r2d2_mcp.server

What this replaces
------------------
The previous planner (gpt_oss.py) put a JSON schema in the prompt, asked a model
to emit one object containing a whole plan, then parsed it with a bag of repair
heuristics - extract_json, repair_json, a regex fallback - and executed the
result open-loop. Three things were wrong with that, and tools fix all three:

  1. The model never learned whether anything worked. The plan was emitted
     before the first wheel turned, so a closed door, an unmountable step or a
     failed climb could not change it. Every tool here returns the real outcome.

  2. Parsing was a guess. A malformed brace meant a repaired plan that might
     mean something different from what the model intended. Tool calls are
     validated against a declared schema by the protocol, before anything moves.

  3. The prompt had to carry the entire capability surface, and grew every time
     the robot gained a feature. Tools are discovered, described and
     versioned by the server, so the robot's capabilities and the model's
     knowledge of them cannot drift apart.

Design of the tool surface
--------------------------
Tools are written for a reader, not a machine. Each one says what it does,
what it costs, and what it will refuse - because a model choosing between
`navigate_to_object` and `explore` needs to know that the first is seconds and
the second is minutes. Failures return an explanation, never a bare False.

Safety invariants the server enforces regardless of what is asked:
  * a precise dock is refused while localisation quality says otherwise
  * a climb is refused unless the robot is actually facing a mountable riser
  * a cross-floor goal is planned as a route, never as a single Nav2 goal
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None

from r2d2_mcp.config import RobotConfig, ServerConfig

log = logging.getLogger('r2d2.mcp')

# ROS, memory and routing are imported lazily inside _startup() and the tools
# that need them, so this module still imports on a machine with no ROS.
mcp = FastMCP('r2d2-robot') if FastMCP is not None else None

_bridge = None
_store = None
_ledger = None
_config = RobotConfig()
_tool_log: List[Dict[str, Any]] = []


def _record(tool: str, args: Dict[str, Any], result: Any) -> Any:
    """Keep a trace of what the agent did, for the episode record."""
    _tool_log.append({'tool': tool, 'args': args,
                      'result_summary': _summarise(result)})
    if _ledger is not None:
        asyncio.create_task(_ledger.append(
            'decision', {'kind': 'tool_call', 'tool': tool, 'args': args,
                         'result': _summarise(result)}))
    return result


def _summarise(result: Any) -> Any:
    if isinstance(result, dict):
        return {k: v for k, v in result.items()
                if k in ('success', 'reason', 'floor', 'count', 'error')}
    if isinstance(result, list):
        return {'count': len(result)}
    return {'value': str(result)[:120]}


def _require_bridge():
    if _bridge is None:
        raise RuntimeError(
            'the robot bridge is not running: this server was started without '
            'ROS. Start it on the machine running the navigation stack.')
    return _bridge


def _require_store():
    if _store is None:
        raise RuntimeError(
            'memory is not available: Postgres is unreachable or R2D2_MEMORY '
            'is off. Navigation tools still work; recall does not.')
    return _store


# ===========================================================================
# Situational awareness
# ===========================================================================

if mcp is not None:

    @mcp.tool()
    async def get_state() -> Dict[str, Any]:
        """Where the robot is and what condition it is in.

        Call this first, and again after anything that moved the robot. It
        returns pose, floor, localisation quality, what the terrain sensors see
        ahead, and a one-line summary of the last thing the camera looked at.

        The `localisation.quality` field matters: when it is "dead_reckoning"
        the robot has just come off a staircase and does not know where it is
        to better than a few tens of centimetres, so precise tasks will be
        refused until it relocalises.
        """
        return _record('get_state', {}, _require_bridge().state())

    @mcp.tool()
    async def look(question: str = '') -> Dict[str, Any]:
        """Take a fresh look through the camera and describe what is there.

        Runs the vision-language model on a live frame and grounds what it finds
        into map coordinates using the LiDAR. Costs one inference call and a few
        seconds, so use it when the robot has arrived somewhere new or when you
        genuinely need to know what is in front of it - not between every step.

        Objects come back with a `grounding_quality`:
          good          the LiDAR agreed on a distance; the coordinates are usable
          coarse        ranged, but the returns disagreed; treat the position as
                        approximate
          bearing_only  the object is above or below the scan plane (a light
                        switch, a mug on a table) so only its direction is known

        Pass `question` to ask something specific about the view.
        """
        result = await _require_bridge().observe(
            question or None, timeout_s=_config.observe_timeout_s)
        return _record('look', {'question': question}, result)

    # =======================================================================
    # Memory
    # =======================================================================

    @mcp.tool()
    async def find_object(query: str, floor: Optional[int] = None,
                          limit: int = 5) -> Dict[str, Any]:
        """Search everything the robot has ever seen, by meaning.

        This is a semantic search, not a keyword match: "somewhere to sit" will
        find a chair and "something cold" will find the fridge. Results carry a
        confidence and an observation count - an object seen once at low
        confidence is a guess, one seen thirty times is a fact.

        Leave `floor` unset to search the whole house. Set it when you only want
        things the robot can reach without using the stairs.
        """
        store = _require_store()
        bridge = _bridge
        hits = await store.find_objects(
            query, floor=floor, limit=limit,
            robot_xy=bridge.pose[:2] if bridge else None)
        result = {'query': query, 'count': len(hits),
                  'objects': [h.as_dict() for h in hits]}
        if not hits:
            result['note'] = (
                'Nothing matching has been seen yet. Try explore() to drive '
                'around and look, or look() if you are already somewhere new.')
        return _record('find_object', {'query': query, 'floor': floor}, result)

    @mcp.tool()
    async def list_known_objects(floor: Optional[int] = None,
                                 limit: int = 40) -> Dict[str, Any]:
        """Everything in the semantic map, best-observed first.

        Use this to answer "what is in the house" or to check whether a room has
        been visited at all. For finding one particular thing, find_object is
        cheaper and better ranked.
        """
        hits = await _require_store().list_objects(floor=floor, limit=limit)
        return _record('list_known_objects', {'floor': floor},
                       {'count': len(hits),
                        'objects': [h.as_dict() for h in hits]})

    @mcp.tool()
    async def remember(fact: str) -> Dict[str, Any]:
        """Write something down so it survives this task.

        For durable observations about the house that are not objects: "the
        study door sticks and needs a firm push", "the upstairs landing rug
        slips". These come back through recall() on later runs.
        """
        fact_id = await _require_store().remember(fact, floor=_bridge.floor if _bridge else None)
        if _ledger is not None:
            await _ledger.append('fact', {'content': fact, 'source': 'agent'})
        return _record('remember', {'fact': fact}, {'stored': True, 'id': fact_id})

    @mcp.tool()
    async def recall(query: str, limit: int = 5) -> Dict[str, Any]:
        """Look up notes and past attempts relevant to what you are doing now.

        Returns both free-text notes written by remember() and previous
        episodes - including the ones that failed. A failure is often the more
        useful memory: knowing the last attempt at this errand ended because a
        door was shut saves repeating it.
        """
        store = _require_store()
        facts = await store.recall(query, limit=limit)
        episodes = await store.recall_episodes(query, limit=3)
        return _record('recall', {'query': query},
                       {'facts': facts, 'past_attempts': episodes})

    # =======================================================================
    # Movement
    # =======================================================================

    @mcp.tool()
    async def navigate_to_coordinates(x: float, y: float, yaw: float = 0.0,
                                      floor: Optional[int] = None) -> Dict[str, Any]:
        """Drive to a point on the map, using the stairs if it is on another floor.

        Coordinates are in metres in the map frame. If `floor` differs from the
        current one, this plans and executes the whole route - drive to the
        staircase, climb it, drive on - rather than sending one goal that Nav2
        would try to satisfy inside the wrong map.

        Waits for the robot to actually get there and reports what happened. On
        failure the reason says why: a closed door, a step the wheels cannot
        mount, a drop the safety monitor refused to approach.

        Accuracy is about 20 cm. For anything tighter, follow this with
        dock_precisely().
        """
        bridge = _require_bridge()
        target_floor = bridge.floor if floor is None else int(floor)
        args = {'x': x, 'y': y, 'yaw': yaw, 'floor': target_floor}

        if target_floor == bridge.floor:
            result = await bridge.navigate_to(x, y, yaw,
                                              timeout_s=_config.nav_timeout_s)
            return _record('navigate_to_coordinates', args, result)

        result = await _execute_route(bridge, target_floor, x, y, yaw,
                                      'the requested position')
        return _record('navigate_to_coordinates', args, result)

    @mcp.tool()
    async def navigate_to_object(name: str,
                                 standoff: float = 0.8) -> Dict[str, Any]:
        """Drive to something the robot has seen before, by name or description.

        Looks the target up in the semantic map, works out a route - including
        stairs if it is on another floor - and drives there, stopping `standoff`
        metres short so the object is in view rather than under the wheels.

        Fails clearly when the thing has never been seen, which is the honest
        answer: use explore() or look() to find it first.
        """
        store = _require_store()
        bridge = _require_bridge()
        hits = await store.find_objects(name, limit=1, robot_xy=bridge.pose[:2])
        if not hits:
            return _record('navigate_to_object', {'name': name}, {
                'success': False,
                'reason': (f'"{name}" is not in the semantic map - the robot '
                           f'has never seen it. Explore the house or look '
                           f'around a room where you expect it to be.')})

        target = hits[0]
        if target.confidence < 0.3 or target.position_sigma > 0.8:
            note = (f'This match is weak (confidence {target.confidence:.2f}, '
                    f'position spread {target.position_sigma:.2f} m). Expect to '
                    f'have to look around on arrival.')
        else:
            note = None

        # Stop short along the line from the robot to the object, so it ends up
        # in the camera's view rather than pressed against it.
        dx, dy = target.x - bridge.pose[0], target.y - bridge.pose[1]
        distance = math.hypot(dx, dy)
        if distance > standoff and distance > 1e-6:
            scale = (distance - standoff) / distance
            goal_x = bridge.pose[0] + dx * scale
            goal_y = bridge.pose[1] + dy * scale
        else:
            goal_x, goal_y = bridge.pose[0], bridge.pose[1]
        goal_yaw = math.atan2(target.y - goal_y, target.x - goal_x)

        if target.floor == bridge.floor:
            result = await bridge.navigate_to(goal_x, goal_y, goal_yaw,
                                              timeout_s=_config.nav_timeout_s)
        else:
            result = await _execute_route(bridge, target.floor, goal_x, goal_y,
                                          goal_yaw, name)

        result['target'] = target.as_dict()
        if note:
            result['note'] = note
        return _record('navigate_to_object', {'name': name}, result)

    @mcp.tool()
    async def navigate_to_room(name: str) -> Dict[str, Any]:
        """Drive to a named room - kitchen, bedroom, study, landing.

        Rooms are places rather than objects: the robot aims for a sensible spot
        inside the room, not at a piece of furniture. Handles stairs when the
        room is on another floor.
        """
        store = _require_store()
        bridge = _require_bridge()
        place = await store.find_place(name)
        if place is None:
            return _record('navigate_to_room', {'name': name}, {
                'success': False,
                'reason': (f'no room called "{name}" is known. Known rooms come '
                           f'from mapping; list_known_objects() will show what '
                           f'has been visited.')})

        if place['floor'] == bridge.floor:
            result = await bridge.navigate_to(
                place['x'], place['y'], place['yaw'],
                timeout_s=_config.nav_timeout_s)
        else:
            result = await _execute_route(bridge, place['floor'], place['x'],
                                          place['y'], place['yaw'], name)
        result['room'] = place['name']
        result['floor'] = place['floor']
        return _record('navigate_to_room', {'name': name}, result)

    @mcp.tool()
    async def dock_precisely(target: str = 'surface',
                             bearing_deg: float = 0.0,
                             standoff: float = 0.35) -> Dict[str, Any]:
        """Close the last stretch to centimetre accuracy.

        Ordinary navigation stops within about 20 cm, which is not enough to
        read a label or line up with a doorway. This servos on the live laser
        scan against a local feature, so map error stops mattering:

          target="surface"  fit a flat face - a fridge door, a cabinet front,
                            the wall under a switch - and stop square to it
          target="gap"      find an opening and centre on it, for driving
                            through a doorway without clipping the frame

        `bearing_deg` is where to look, relative to straight ahead; positive is
        to the left. Refused when localisation quality is poor, because a
        precise stop against a pose that is not precise is theatre.
        """
        bridge = _require_bridge()
        args = {'target': target, 'bearing_deg': bearing_deg}

        if not bridge.health.get('precise_goals_allowed', True):
            return _record('dock_precisely', args, {
                'success': False,
                'reason': (f'localisation is currently '
                           f'"{bridge.health.get("quality", "unknown")}" with an '
                           f'estimated drift of '
                           f'{bridge.health.get("dead_reckoning_drift_estimate", "?")} m. '
                           f'Drive around to relocalise before docking.')})

        from r2d2_mcp.docking_client import run_docking
        result = await run_docking(bridge, math.radians(bearing_deg), target,
                                   standoff, timeout_s=_config.dock_timeout_s)
        return _record('dock_precisely', args, result)

    @mcp.tool()
    async def climb_stairs() -> Dict[str, Any]:
        """Go up or down the flight the robot is currently facing.

        Only works from the foot of a staircase, squared up to it - drive there
        first, or use navigate_to_coordinates with a different floor, which does
        the whole thing for you.

        The climb takes around a minute and the robot loses its wheel odometry
        throughout, so the pose estimate drifts and is re-seeded on arrival.
        Do not attempt anything precise immediately afterwards.
        """
        result = await _require_bridge().climb(timeout_s=_config.climb_timeout_s)
        return _record('climb_stairs', {}, result)

    @mcp.tool()
    async def explore(duration_s: float = 60.0) -> Dict[str, Any]:
        """Drive around the current floor looking at things.

        Use when something needs finding and the semantic map does not have it.
        Alternates short moves with camera looks, adding what it sees to memory.
        This is minutes, not seconds - prefer find_object first.
        """
        from r2d2_mcp.explore import run_exploration
        result = await run_exploration(_require_bridge(), duration_s)
        return _record('explore', {'duration_s': duration_s}, result)

    # =======================================================================
    # Speech and safety
    # =======================================================================

    @mcp.tool()
    async def say(text: str) -> Dict[str, Any]:
        """Speak out loud to whoever is nearby.

        For talking to a person in the room. Do not use it to narrate progress
        back to whoever gave the instruction - that goes in your final answer.
        """
        return _record('say', {'text': text}, _require_bridge().speak(text))

    @mcp.tool()
    async def stop() -> Dict[str, Any]:
        """Stop the robot immediately and cancel whatever it was doing."""
        bridge = _require_bridge()
        await bridge.cancel_navigation()
        return _record('stop', {}, bridge.emergency_stop())

    @mcp.tool()
    async def reset_safety_stop() -> Dict[str, Any]:
        """Clear a latched terrain safety stop.

        The safety monitor latches when the robot tips beyond its limits. Only
        call this once you know why it triggered and that the robot is on level
        ground - clearing it blind will simply latch again, or worse.
        """
        return _record('reset_safety_stop', {},
                       _require_bridge().reset_safety_stop())

    # =======================================================================
    # Resources
    # =======================================================================

    @mcp.resource('robot://state')
    def state_resource() -> str:
        """Live robot state, as JSON."""
        return json.dumps(_bridge.state() if _bridge else
                          {'error': 'bridge not running'}, indent=2)

    @mcp.resource('robot://floors')
    def floors_resource() -> str:
        """The floor graph: which floors exist, which are mapped, how they join."""
        return json.dumps(_bridge.floor_graph if _bridge else {}, indent=2)


async def _execute_route(bridge, target_floor: int, x: float, y: float,
                         yaw: float, description: str) -> Dict[str, Any]:
    """Drive a cross-floor route leg by leg, stopping at the first failure.

    Kept in one place so every cross-floor tool behaves identically, and so the
    model never has to sequence a staircase by hand.
    """
    from r2d2_navigation.route_planner import (LEG_CLIMB, LEG_DRIVE, RouteError,
                                               graph_from_payload, plan_route)

    if not bridge.floor_graph:
        return {'success': False,
                'reason': ('the floor graph is not available, so a cross-floor '
                           'route cannot be planned. Is floor_manager running?')}

    try:
        graph = graph_from_payload(bridge.floor_graph)
        legs = plan_route(graph, bridge.floor, target_floor, x, y, yaw,
                          goal_description=description)
    except RouteError as exc:
        return {'success': False, 'reason': str(exc)}

    completed = []
    for index, leg in enumerate(legs):
        if leg.kind == LEG_DRIVE:
            outcome = await bridge.navigate_to(leg.x, leg.y, leg.yaw,
                                               timeout_s=_config.nav_timeout_s)
        elif leg.kind == LEG_CLIMB:
            outcome = await bridge.climb(timeout_s=_config.climb_timeout_s)
        else:
            continue

        completed.append({'leg': leg.description,
                          'success': outcome.get('success')})
        if not outcome.get('success'):
            return {
                'success': False,
                'reason': (f'route failed at step {index + 1} of {len(legs)} '
                           f'({leg.description}): '
                           f'{outcome.get("reason", "unknown")}'),
                'completed_legs': completed,
                'floor': bridge.floor,
            }

    return {'success': True, 'completed_legs': completed,
            'floor': bridge.floor,
            'pose': {'x': round(bridge.pose[0], 2), 'y': round(bridge.pose[1], 2)}}


async def _startup():
    """Bring up the ROS bridge and memory, tolerating either being absent."""
    global _bridge, _store, _ledger

    try:
        from r2d2_mcp.robot_bridge import RobotBridge as _RB
        _bridge = _RB()
        log.info('robot bridge connected')
    except Exception as exc:                      # noqa: BLE001
        log.error('robot bridge unavailable (%s): movement tools will refuse. '
                  'Is the navigation stack running?', exc)

    if not _config.memory_enabled:
        log.info('memory disabled by R2D2_MEMORY=off')
        return

    try:
        from r2d2_memory.ledger import make_ledger as _ml
        from r2d2_memory.store import MemoryStore as _MS
        _ledger = _ml(enabled=True, servers=_config.nats_url)
        await _ledger.connect()
        _store = _MS(dsn=_config.pg_dsn)
        await _store.connect()
        log.info('memory connected')
    except Exception as exc:                      # noqa: BLE001
        log.warning('memory unavailable (%s): recall tools will refuse, '
                    'movement is unaffected', exc)
        _store = None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        # stderr: stdout is the MCP transport when running over stdio, and a
        # stray log line there corrupts the protocol stream.
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    if FastMCP is None:
        print('The MCP SDK is not installed. pip install "mcp[cli]"',
              file=sys.stderr)
        return 1

    server_config = ServerConfig()
    asyncio.run(_startup())

    log.info('serving the robot over %s transport', server_config.transport)

    if server_config.transport == 'http':
        mcp.settings.host = server_config.host
        mcp.settings.port = server_config.port
        mcp.run(transport='streamable-http')
    else:
        mcp.run(transport='stdio')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
