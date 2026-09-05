import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'r2d2_memory'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'docker-compose.yml']),
        (os.path.join('share', package_name, 'sql'), glob('r2d2_memory/*.sql')),
    ],
    # Left out of install_requires on purpose: every one of these is optional at
    # runtime (see requirements.txt), and making them hard dependencies would
    # stop the package building on a robot that only needs to drive.
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jayinaksha',
    maintainer_email='jayinaksha_2302cm08@iitp.ac.in',
    description='Event ledger and pgvector semantic/episodic memory.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'memory_node = r2d2_memory.memory_node:main',
            'memory_projector = r2d2_memory.projector:main',
        ],
    },
)
