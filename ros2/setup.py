from setuptools import setup

package_name = 'unicycle_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@todo.todo',
    description='ROS 2 wrapper for the unicycle controller',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'unicycle_node = unicycle_control.unicycle_node:main'
        ],
    },
)
