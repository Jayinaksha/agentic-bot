"""Pytest configuration for the repository root.

The v1 ROS packages carry the standard ament lint stubs - test_copyright.py,
test_flake8.py, test_pep257.py - which import `ament_copyright` and friends.
Those exist only inside a sourced ROS environment, so on a plain interpreter
they fail at collection: three errors before a single real test runs.

That matters more than it looks. docs/learn.html tells a contributor to run the
test suite as their second command, and a wall of import errors on a clean
checkout reads as "this project is broken" rather than "these three files need
ROS". They are still collected and run by colcon test, which is where they
belong.
"""

collect_ignore_glob = [
    'src/*/test/test_copyright.py',
    'src/*/test/test_flake8.py',
    'src/*/test/test_pep257.py',
]
