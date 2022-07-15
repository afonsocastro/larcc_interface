#!/bin/bash

echo "Running 1st app..."
$HOME/path_to_1st/app &     # this runs an app from user's home directory
                            # the ampersand makes the app `fork`
                            # it means it'll start in the background
                            # and the script will continue to execute
echo "Moving gripper"
roscd use_cases/src/gripper_control_test
python3 gripper_close.pyter