from setuptools import find_packages, setup

package_name = 'twist_stamper_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Converts Twist to TwistStamped.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This line makes 'twist_stamper_node' executable
            'twist_stamper_node = twist_stamper_pkg.twist_stamper_node:main',
        ],
    },
)