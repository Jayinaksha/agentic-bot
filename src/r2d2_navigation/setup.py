import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'r2d2_navigation'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jayinaksha',
    maintainer_email='jayinaksha_2302cm08@iitp.ac.in',
    description='Nav2 configuration, cross-floor routing and precise docking.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'precise_docking = r2d2_navigation.precise_docking:main',
        ],
    },
)
