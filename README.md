# larcc_interface

```
cd catkin_ws/src
git clone https://github.com/afonsocastro/larcc_interface.git
git clone https://github.com/ros-industrial/ur_msgs.git
cd catkin_ws
catkin_make

roslaunch ur_robot_driver ur10e_bringup.launch robot_ip:=192.168.56.2

*connect the cable and open the external control program on the teach pendant*

roslaunch ur10e_moveit_config ur10e_moveit_planning_execution.launch
roslaunch ur10e_moveit_config moveit_rviz.launch config:=true

```