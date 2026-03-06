#include <ur_robot_driver/robot_state_helper.h>
#include <std_srvs/Trigger.h>
#include <thread>
#include <chrono>

namespace ur_driver
{

RobotStateHelper::RobotStateHelper(const ros::NodeHandle& nh)
  : nh_(nh)
  , is_started_(false)
  , robot_mode_(urcl::RobotMode::UNKNOWN)
  , safety_mode_(urcl::SafetyMode::UNDEFINED_SAFETY_MODE)
  , set_mode_as_(nh_, "set_mode", false)
{
    robot_mode_sub_ = nh_.subscribe("robot_mode", 1, &RobotStateHelper::robotModeCallback, this);
    safety_mode_sub_ = nh_.subscribe("safety_mode", 1, &RobotStateHelper::safetyModeCallback, this);

    unlock_protective_stop_srv_ = nh_.serviceClient<std_srvs::Trigger>("dashboard/unlock_protective_stop");
    restart_safety_srv_ = nh_.serviceClient<std_srvs::Trigger>("dashboard/restart_safety");
    power_on_srv_ = nh_.serviceClient<std_srvs::Trigger>("dashboard/power_on");
    power_off_srv_ = nh_.serviceClient<std_srvs::Trigger>("dashboard/power_off");
    brake_release_srv_ = nh_.serviceClient<std_srvs::Trigger>("dashboard/brake_release");
    stop_program_srv_ = nh_.serviceClient<std_srvs::Trigger>("dashboard/stop");
    play_program_srv_ = nh_.serviceClient<std_srvs::Trigger>("dashboard/play");

    play_program_srv_.waitForExistence();
    set_mode_as_.registerGoalCallback(std::bind(&RobotStateHelper::setModeGoalCallback, this));
    set_mode_as_.registerPreemptCallback(std::bind(&RobotStateHelper::setModePreemptCallback, this));
}

void RobotStateHelper::robotModeCallback(const ur_dashboard_msgs::RobotMode& msg)
{
    urcl::RobotMode new_mode = static_cast<urcl::RobotMode>(msg.mode);
    if (robot_mode_ != new_mode)
    {
        robot_mode_ = new_mode;
        ROS_INFO_STREAM("Robot mode is now " << urcl::robotModeString(robot_mode_));
        updateRobotState();
        if (!is_started_) startActionServer();
    }
}

void RobotStateHelper::safetyModeCallback(const ur_dashboard_msgs::SafetyMode& msg)
{
    urcl::SafetyMode new_mode = static_cast<urcl::SafetyMode>(msg.mode);
    if (safety_mode_ != new_mode)
    {
        safety_mode_ = new_mode;
        ROS_INFO_STREAM("Robot's safety mode is now " << urcl::safetyModeString(safety_mode_));
        updateRobotState();
        if (!is_started_) startActionServer();
    }
}

void RobotStateHelper::updateRobotState()
{
    if (!set_mode_as_.isActive() || !goal_) return;

    feedback_.current_robot_mode =
        static_cast<ur_dashboard_msgs::SetModeFeedback::_current_robot_mode_type>(robot_mode_);
    feedback_.current_safety_mode =
        static_cast<ur_dashboard_msgs::SetModeFeedback::_current_safety_mode_type>(safety_mode_);
    set_mode_as_.publishFeedback(feedback_);

    urcl::RobotMode target_mode = static_cast<urcl::RobotMode>(goal_->target_robot_mode.mode);

    if (robot_mode_ < target_mode || safety_mode_ > urcl::SafetyMode::REDUCED)
    {
        doTransition();
    }
    else if (robot_mode_ == target_mode)
    {
        result_.success = true;
        result_.message = "Reached target robot mode.";
        if (robot_mode_ == urcl::RobotMode::RUNNING && goal_->play_program)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            safeDashboardTrigger(&play_program_srv_);
        }
        if (set_mode_as_.isActive()) set_mode_as_.setSucceeded(result_);
    }
    else
    {
        result_.success = false;
        result_.message = "Robot reached higher mode than requested.";
        set_mode_as_.setAborted(result_);
    }
}

void RobotStateHelper::doTransition()
{
    urcl::RobotMode target_mode = static_cast<urcl::RobotMode>(goal_->target_robot_mode.mode);

    if (target_mode < robot_mode_)
    {
        safeDashboardTrigger(&power_off_srv_);
    }
    else
    {
        switch (safety_mode_)
        {
            case urcl::SafetyMode::PROTECTIVE_STOP: safeDashboardTrigger(&unlock_protective_stop_srv_); break;
            case urcl::SafetyMode::SYSTEM_EMERGENCY_STOP:
            case urcl::SafetyMode::ROBOT_EMERGENCY_STOP:
                ROS_WARN_STREAM("Please release the EM-Stop to proceed."); break;
            case urcl::SafetyMode::VIOLATION:
            case urcl::SafetyMode::FAULT: safeDashboardTrigger(&restart_safety_srv_); break;
            default:
                switch (robot_mode_)
                {
                    case urcl::RobotMode::CONFIRM_SAFETY: ROS_WARN_STREAM("Confirm safety on teach pendant."); break;
                    case urcl::RobotMode::BOOTING: ROS_INFO_STREAM("Robot is booting."); break;
                    case urcl::RobotMode::POWER_OFF: safeDashboardTrigger(&power_on_srv_); break;
                    case urcl::RobotMode::POWER_ON: ROS_INFO_STREAM("Robot powering on."); break;
                    case urcl::RobotMode::IDLE: safeDashboardTrigger(&brake_release_srv_); break;
                    case urcl::RobotMode::BACKDRIVE: ROS_INFO_STREAM("Backdrive mode."); break;
                    case urcl::RobotMode::RUNNING: ROS_INFO_STREAM("Operational mode reached."); break;
                    default: ROS_WARN_STREAM("Unhandled robot mode."); break;
                }
        }
    }
}

void RobotStateHelper::setModeGoalCallback()
{
    goal_ = set_mode_as_.acceptNewGoal();
    urcl::RobotMode target_mode = static_cast<urcl::RobotMode>(goal_->target_robot_mode.mode);

    switch (target_mode)
    {
        case urcl::RobotMode::POWER_OFF:
        case urcl::RobotMode::IDLE:
        case urcl::RobotMode::RUNNING:
            if (robot_mode_ != target_mode || safety_mode_ > urcl::SafetyMode::REDUCED)
            {
                if (goal_->stop_program) safeDashboardTrigger(&stop_program_srv_);
                doTransition();
            }
            else updateRobotState();
            break;
        default:
            result_.message = "Requested illegal or unsupported mode.";
            result_.success = false;
            set_mode_as_.setAborted(result_);
            break;
    }
}

void RobotStateHelper::setModePreemptCallback()
{
    ROS_INFO_STREAM("Current goal got preempted.");
    set_mode_as_.setPreempted();
}

bool RobotStateHelper::safeDashboardTrigger(ros::ServiceClient* srv_client)
{
    assert(srv_client != nullptr);
    std_srvs::Trigger srv;
    srv_client->call(srv);
    ROS_INFO_STREAM(srv.response.message);
    if (!srv.response.success)
    {
        result_.success = false;
        result_.message = "Dashboard service failed: " + srv.response.message;
        set_mode_as_.setAborted(result_);
    }
    return srv.response.success;
}

void RobotStateHelper::startActionServer()
{
    if (robot_mode_ != urcl::RobotMode::UNKNOWN && safety_mode_ != urcl::SafetyMode::UNDEFINED_SAFETY_MODE)
    {
        set_mode_as_.start();
        is_started_ = true;
    }
}

}  // namespace ur_driver