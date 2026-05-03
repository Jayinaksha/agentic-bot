import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'r2d2_comms'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line finds all .launch.py files in the 'launch' directory and installs them
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # This line finds all .yaml files in the 'config' directory and installs them
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noblehalogen',
    maintainer_email='your_email@example.com',
    description='Communication package for R2D2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local_image_sender = r2d2_comms.local_image_sender:main',
            'json_viewer = r2d2_comms.json_viewer:main',
            'r2d2_bridge = r2d2_comms.r2d2_bridge:main',
            'r2d2_reciever = r2d2_comms.r2d2_reciever:main',
            # Add other scripts here as you create them
        ],
    },
)
