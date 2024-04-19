from setuptools import find_packages, setup

package_name = 'lab_final'

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
            'image_utils = lab_final.image_utils:main',
            'rotation_script = lab_final.rotation_script:main',
            'sign_detect = lab_final.sign_detect:main',
            'lidar_detect = lab_final.lidar_detect:main',
            'go_to_goal = lab_final.go_to_goal:main',
        ],
    },
)
