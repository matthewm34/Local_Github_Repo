# Cool kids only

### how to build a package

go to root of workspace
'''cd turtle_ws'''

build all packages with:
'''colcon build'''

or only build for specific package with:
'''colcon build --package-select my_package'''

to use package and executable:
'''source /opt/ros/humble/setup.bash'''

inside turtle_ws:
'''. install/setup.bash'''

to run a node from package:
'''ros2 run my_package my_node'''