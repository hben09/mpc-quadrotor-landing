# ~/ros2_ws/src/joy2crsf_ros2/setup.py
from setuptools import setup

package_name = 'crsf_joystick'

setup(
    name=package_name,
    version='0.0.1',
    packages=['crsf_joystick'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'joystick_to_crsf = crsf_joystick.joystick_to_crsf:main',  # <- matches your file
            # (optional) also export 'joy_to_crsf' if you add that file later
        ],
    },
)
