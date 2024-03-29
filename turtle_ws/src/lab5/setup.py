from setuptools import find_packages, setup

package_name = 'lab5'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mjmck34',
    maintainer_email='mjmckenna34@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'print_fixed_odometry = lab5.print_fixed_odometry:main',
            'rotation_script = lab5.rotation_script:main',
            'get_object_range = lab5.get_object_range:main',
            'go_to_goal = lab5.go_to_goal:main',
            'new_script = lab5.new:main',
        ],
    },
)
