from dataclasses import dataclass


@dataclass(frozen=True)
class GripperStatusDTO:

    is_activated: bool

    going_to_position_request: bool

    activation_in_progress: bool
    activation_completed: bool
    is_in_automatic_release: bool

    object_detected: bool
    fingers_in_motion_towards_requested_position: bool
    fingers_stopped_opening: bool
    fingers_stopped_closing: bool
    fingers_are_at_requested_position: bool

    fault_action_delayed: bool
    fault_missing_activation_bit: bool
    fault_max_temperature_exceeded: bool
    fault_no_communication_1_second: bool
    fault_under_minimum_voltage: bool
    fault_internal: bool
    fault_activation: bool
    fault_overcurrent_triggered: bool
    fault_automatic_release_completed: bool

    requested_position: int
    actual_position: int
    actual_force_motor_current: int

    def __repr__(self):
        rep = '-------------------------------------\n' + \
              'GRIPPER STATUS\n' + \
              '-------------------------------------\n' + \
              'is_activated: ' + str(self.is_activated) + '\n' + \
              'going_to_position_request: ' + str(self.going_to_position_request) + '\n' + \
              'activation_in_progress: ' + str(self.activation_in_progress) + '\n' + \
              'activation_completed: ' + str(self.activation_completed) + '\n' + \
              'is_in_automatic_release: ' + str(self.is_in_automatic_release) + '\n' + \
              'fingers_in_motion_towards_requested_position: ' + str(self.fingers_in_motion_towards_requested_position) + '\n' + \
              'fingers_stopped_opening: ' + str(self.fingers_stopped_opening) + '\n' + \
              'fingers_stopped_closing: ' + str(self.fingers_stopped_closing) + '\n' + \
              'fingers_are_at_requested_position: ' + str(self.fingers_are_at_requested_position) + '\n' + \
              'fault_action_delayed: ' + str(self.fault_action_delayed) + '\n' + \
              'fault_missing_activation_bit: ' + str(self.fault_missing_activation_bit) + '\n' + \
              'fault_max_temperature_exceeded: ' + str(self.fault_max_temperature_exceeded) + '\n' + \
              'fault_no_communication_1_second: ' + str(self.fault_no_communication_1_second) + '\n' + \
              'fault_under_minimum_voltage: ' + str(self.fault_under_minimum_voltage) + '\n' + \
              'fault_internal: ' + str(self.fault_internal) + '\n' + \
              'fault_activation: ' + str(self.fault_activation) + '\n' + \
              'fault_overcurrent_triggered: ' + str(self.fault_overcurrent_triggered) + '\n' + \
              'fault_automatic_release_completed: ' + str(self.fault_automatic_release_completed) + '\n' + \
              'requested_position: ' + str(self.requested_position) + '\n' + \
              'actual_position: ' + str(self.actual_position) + '\n' + \
              'actual_force_motor_current: ' + str(self.actual_force_motor_current) + '\n'

        return rep


