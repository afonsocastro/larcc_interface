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
   3. Subnet mask: ```255.255.255.0```
   4. Default gateway: ```192.168.56.2```

9. Click on *Apply*

![tp4](docs/tp1.jpg)

10. Disable EtherNet/IP fieldbus:

Installation > Fieldbus > EtherNet/IP > Disable

![tp5](docs/tp_ethernet_fieldbus.png)

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

Now compile your catkin workspace:
```
cd ~/catkin_ws
catkin_make
```


Finally, to establish the communication between the robot and the computer, it is required to **connect an Ethernet cable from the UR10e controller to the computer**.
After you connect the cable, you need to configure the IPv4 like this:

![tp6](docs/ip.jpeg)

## Controlling UR10e through MoveIt with RViz
Just need to follow these next 4 steps to remotely control the real UR10e robot, connected via the Ethernet cable to your computer.

1. ```roslaunch ur_robot_driver ur10e_bringup.launch robot_ip:=192.168.56.2 ```
2. Run the external control program on the teach pendant:

   Click on *Program* + *URCaps* + *External Control* + Press "play"

![tp7](docs/tp2.jpg)

At this point, you should get the message "_Robot connected to reverse interface. Ready to receive control commands._" printed out on your terminal window.

3. ``` roslaunch ur10e_moveit_config ur10e_moveit_planning_execution.launch```

At this point, you should get the green message "_You can start planning now!_" printed out on your terminal window, just like this:

![tp8](docs/you_can_start_planning.png)

4. ``` roslaunch ur10e_moveit_config moveit_rviz.launch config:=true```


Now you can control the real robot, by simply moving the manipulator marker on RViz and then asking the robot to move to that goal (using the Motion Planning Panel).
MoveIt will plan the trajetory.

![tp9](docs/UR10e_moving_moveit.mp4)


## Real-time UR10e following a tracked object

[//]: # (todo: create this section)
(Under construction)










