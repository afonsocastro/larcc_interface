#pragma once

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <std_srvs/Trigger.h>

#include <ur_dashboard_msgs/SetModeAction.h>
#include <ur_dashboard_msgs/RobotMode.h>
#include <ur_dashboard_msgs/SafetyMode.h>

#include <ur_client_library/ur/ur_driver.h>  // Para urcl::RobotMode, urcl::SafetyMode, robotModeString, safetyModeString

namespace ur_driver
{

class RobotStateHelper
{
public:
    explicit RobotStateHelper(const ros::NodeHandle& nh);

private:
    // Callbacks ROS
    void robotModeCallback(const ur_dashboard_msgs::RobotMode& msg);
    void safetyModeCallback(const ur_dashboard_msgs::SafetyMode& msg);

    // Atualização do estado do robô
    void updateRobotState();
    void doTransition();

    // Action callbacks
    void setModeGoalCallback();
    void setModePreemptCallback();

    // Serviço seguro para acionar dashboard
    bool safeDashboardTrigger(ros::ServiceClient* srv_client);

    // Inicializa o action server quando possível
    void startActionServer();

    ros::NodeHandle nh_;
    bool is_started_;

    urcl::RobotMode robot_mode_;
    urcl::SafetyMode safety_mode_;

    ros::Subscriber robot_mode_sub_;
    ros::Subscriber safety_mode_sub_;

    ros::ServiceClient unlock_protective_stop_srv_;
    ros::ServiceClient restart_safety_srv_;
    ros::ServiceClient power_on_srv_;
    ros::ServiceClient power_off_srv_;
    ros::ServiceClient brake_release_srv_;
    ros::ServiceClient stop_program_srv_;
    ros::ServiceClient play_program_srv_;

    actionlib::SimpleActionServer<ur_dashboard_msgs::SetModeAction> set_mode_as_;
    ur_dashboard_msgs::SetModeFeedback feedback_;
    ur_dashboard_msgs::SetModeResult result_;
    ur_dashboard_msgs::SetModeGoalConstPtr goal_;
};

}  // namespace ur_driver