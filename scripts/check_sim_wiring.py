#!/usr/bin/env python3
"""
Check that every simulated sensor actually reaches ROS.

    python3 scripts/check_sim_wiring.py

A Gazebo sensor publishes on the topic its `<topic>` element names. The bridge
forwards a fixed list of gz topics into ROS. Nothing connects the two but a
string typed in both files, and nothing at runtime complains when they disagree
- the sensor publishes into the void, the bridge forwards a topic nobody emits,
and the subscribing node simply waits forever on a topic that will never carry a
message.

The consequence is not uniform. A missing camera is obvious the first time you
ask the robot to look at something. A missing ToF beam is not: `terrain_monitor`
holds `None` for that beam, reports its class as `unknown`, and `unknown` is
neither a cliff nor a riser. The robot would drive towards a staircase seeing
nothing at all, which is the one failure this sensor exists to prevent.

Three things are checked:

  1. every sensor's topic has a bridge entry
  2. every bridge entry has a producer (a sensor, or a known world-level topic)
  3. the bridged message type matches the sensor type - a gpu_lidar forwarded as
     an Image fails at runtime, not at startup

Requires xacro to expand the description. Without it the script says so and
exits 0 rather than reporting a pass it did not perform.
"""

from __future__ import annotations

import os
import sys
import xml.dom.minidom as minidom
from typing import Dict, Optional, Tuple

try:
    import yaml
except ImportError:
    sys.exit('PyYAML required: pip install pyyaml')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
URDF = os.path.join(ROOT, 'src/r2d2_description/urdf/r2d2_tristar.urdf.xacro')
BRIDGE = os.path.join(ROOT, 'src/r2d2_sim/config/house_bridge.yaml')

# gz sensor type -> the ROS message the bridge must carry it as.
EXPECTED_TYPE = {
    'gpu_lidar': 'sensor_msgs/msg/LaserScan',
    'lidar': 'sensor_msgs/msg/LaserScan',
    'ray': 'sensor_msgs/msg/LaserScan',
    'imu': 'sensor_msgs/msg/Imu',
    'camera': 'sensor_msgs/msg/Image',
    'depth_camera': 'sensor_msgs/msg/Image',
    'contact': 'ros_gz_interfaces/msg/Contacts',
}

# Topics the world publishes rather than any sensor on the robot.
WORLD_LEVEL = {
    'clock',
    'camera_info',
    'model/r2d2_tristar/pose',
    'tail_position_cmd',
}


def expand() -> Optional[str]:
    try:
        import xacro
    except ImportError:
        return None
    try:
        return xacro.process_file(URDF, mappings={'use_sim': 'true'}).toxml()
    except Exception as exc:                       # noqa: BLE001
        sys.exit(f'xacro could not expand the description: {exc}')


def sensors(xml: str) -> Dict[str, Tuple[str, Optional[str]]]:
    """Sensor name -> (gz type, declared topic)."""
    out = {}
    for sensor in minidom.parseString(xml).getElementsByTagName('sensor'):
        topic = sensor.getElementsByTagName('topic')
        value = (topic[0].firstChild.nodeValue.strip()
                 if topic and topic[0].firstChild else None)
        out[sensor.getAttribute('name')] = (sensor.getAttribute('type'), value)
    return out


def main() -> int:
    xml = expand()
    if xml is None:
        print('xacro is not installed, so the description cannot be expanded '
              'and this check cannot run. pip install xacro.')
        return 0

    found = sensors(xml)
    bridge = yaml.safe_load(open(BRIDGE))
    by_gz = {entry['gz_topic_name'].lstrip('/'): entry for entry in bridge}

    print(f'{len(found)} sensors in the description, '
          f'{len(bridge)} bridge entries\n')
    problems = 0

    # --- 1 and 3: every sensor reaches ROS, as the right type ---------------
    for name, (kind, topic) in sorted(found.items()):
        if topic is None:
            print(f'[FAIL] sensor "{name}" declares no <topic>, so it publishes '
                  f'on a generated path the bridge cannot name')
            problems += 1
            continue

        entry = by_gz.get(topic.lstrip('/'))
        if entry is None:
            print(f'[FAIL] sensor "{name}" publishes gz topic "{topic}" but no '
                  f'bridge entry forwards it, so nothing in ROS ever receives '
                  f'it and every subscriber waits forever')
            problems += 1
            continue

        expected = EXPECTED_TYPE.get(kind)
        actual = entry.get('ros_type_name')
        if expected and actual != expected:
            print(f'[FAIL] sensor "{name}" is a {kind} but the bridge carries '
                  f'"{topic}" as {actual}; it should be {expected}. A type '
                  f'mismatch fails when the first message arrives, not at '
                  f'startup')
            problems += 1
            continue

        print(f'[ ok ] {name:16s} {kind:10s} {topic:20s} -> '
              f'{entry["ros_topic_name"]}')

    # --- 2: no bridge entry forwards a topic nothing produces ---------------
    produced = {t.lstrip('/') for _, t in found.values() if t}
    print()
    for key, entry in sorted(by_gz.items()):
        if key in produced or key in WORLD_LEVEL:
            continue
        direction = entry.get('direction', 'GZ_TO_ROS')
        if direction == 'ROS_TO_GZ':
            continue                               # a command, not a sensor
        print(f'[FAIL] the bridge forwards gz topic "{key}" to '
              f'{entry["ros_topic_name"]}, but no sensor in the description '
              f'publishes it')
        problems += 1

    if not problems:
        print('every sensor is bridged, correctly typed, and nothing is '
              'forwarded that no sensor produces')
    else:
        print(f'\n{problems} problem(s) found')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
