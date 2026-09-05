from setuptools import find_packages, setup

package_name = 'r2d2_mcp'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    # mcp and httpx are listed in requirements.txt rather than here: the ROS
    # build should not fail on a robot that only runs the navigation stack.
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jayinaksha',
    maintainer_email='jayinaksha_2302cm08@iitp.ac.in',
    description='MCP server and NVIDIA-model agent for the R2D2-Redux robot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'r2d2_mcp_server = r2d2_mcp.server:main',
            'r2d2_agent = r2d2_mcp.agent:main',
        ],
    },
)
