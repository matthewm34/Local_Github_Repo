from setuptools import find_packages, setup

package_name = 'lab3'

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
            'camera_debugger = lab3.camera_debugger:main',
            'find_object = lab3.find_object:main',
            'get_object_range = lab3.get_object_range:main',
        ],
    },
)
