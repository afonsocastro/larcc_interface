# larcc_interface


## Table of Contents

1. [Configuration](#configuration)
2. [Installation](#installation)
3. [Controlling UR10e through MoveIt with RViz](#controlling-ur10e-through-moveit-with-rviz)
4. [Real-time UR10e following a tracked object](#real-time-ur10e-following-a-tracked-object)


## Configuration
This repository was built to work with:
* UR10e manipulator (Universal Robot 10 e-series)
* Ubuntu 20.04.3 LTS
* ROS Noetic 

## Installation
1. [On UR10e controller](#on-ur10e-controller)
2. [On computer](#on-computer)

### On UR10e controller
For working on a real robot you need to install the [externalcontrol-1.0.5.urcap](https://github.com/afonsocastro/larcc_interface/blob/master/resources/externalcontrol-1.0.5.urcap) which can be found inside the resources folder of this repository.

Using a USB pen drive, follow:
1. Format the flash drive
2. Download and save the externalcontrol-1.0.5.urcap on the USB pen drive
3. Insert the USB drive on UR10e controller (the controller has two USB ports)

![controller-ports](docs/controller_ports.png)

4. Turn on the Teach Pendant
 
![tp1](docs/es_01_welcome.png)

5. Click on *Menu* (top right corner) + *System* + *URCaps* + Select *External Control* and press "+"

![tp2](docs/es_05_urcaps_installed.png)

6. Configure the remote host's IP to ```192.168.56.1```

![tp3](docs/es_07_installation_excontrol.png)

7. Click on *Menu* (top right corner) + *System* + *Network*
8. Configure:
   1. Network method : Static Address
   2. IP address: ```192.168.56.2```
   3. Subnet mask: ```255.255.255.0
   4. Default gateway: ```192.168.56.2```

9. Click on *Apply*

![tp4](docs/tp1.jpg)


### On computer
First, it is required to have MoveIt installed in your system:

```
sudo apt install ros-noetic-moveit
```

Besides MoveIt, there are other packages that need to be installed:

```
sudo apt-get install ros-noetic-industrial-robot-status-interface
sudo apt-get install ros-noetic-scaled-controllers
sudo apt-get install ros-noetic-pass-through-controllers
sudo apt-get install ros-noetic-ur-client-library
sudo apt-get install ros-noetic-velocity-controllers
sudo apt-get install ros-noetic-force-torque-sensor-controller
```
(**Note:** At this moment, if you do not have a catkin workspace, you should now create one, by following the steps described [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace))

After all these installations, on your catkin workspace you need to clone this repository and the ```ur_msgs``` (http://wiki.ros.org/ur_msgs) package:

```
cd catkin_ws/src
git clone https://github.com/afonsocastro/larcc_interface.git
git clone https://github.com/ros-industrial/ur_msgs.git
```

Finally, lets compile our catkin workspace:
```
cd ~/catkin_ws
catkin_make
```

## Controlling UR10e through MoveIt with RViz

roslaunch ur_robot_driver ur10e_bringup.launch robot_ip:=192.168.56.2

*connect the cable and open the external control program on the teach pendant*

roslaunch ur10e_moveit_config ur10e_moveit_planning_execution.launch
roslaunch ur10e_moveit_config moveit_rviz.launch config:=true

## Real-time UR10e following a tracked object










