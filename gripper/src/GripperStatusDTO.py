from dataclasses import dataclass
from colorama import Fore


@dataclass(frozen=True)
class GripperStatusDTO:
    is_activated: bool

    going_to_position_request: bool

    activation_in_progress: bool
    activation_completed: bool
    is_in_automatic_release_or_reset: bool

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
    fault_automatic_release_in_progress: bool

    requested_position: float
    actual_position: float
    actual_force_motor_current: float

    def __repr__(self):
        rep = '-------------------------------------\n' + \
              'GRIPPER STATUS\n' + \
              '-------------------------------------\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'is_activated: ' + Fore.YELLOW + str(self.is_activated) + '\n' + Fore.RESET + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'going_to_position_request: ' + Fore.YELLOW + str(self.going_to_position_request) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'activation_in_progress: ' + Fore.YELLOW + str(self.activation_in_progress) + '\n' + \
              Fore.LIGHTBLUE_EX + 'activation_completed: ' + Fore.YELLOW + str(self.activation_completed) + '\n' + \
              Fore.LIGHTBLUE_EX + 'is_in_automatic_release_or_reset: ' + Fore.YELLOW + str(self.is_in_automatic_release_or_reset) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'object_detected: ' + Fore.YELLOW + str(self.object_detected) + '\n' + \
              Fore.LIGHTBLUE_EX + 'fingers_in_motion_towards_requested_position: ' + Fore.YELLOW + str(self.fingers_in_motion_towards_requested_position) + '\n' + \
              Fore.LIGHTBLUE_EX + 'fingers_stopped_opening: ' + Fore.YELLOW + str(self.fingers_stopped_opening) + '\n' + \
              Fore.LIGHTBLUE_EX + 'fingers_stopped_closing: ' + Fore.YELLOW + str(self.fingers_stopped_closing) + '\n' + \
              Fore.LIGHTBLUE_EX + 'fingers_are_at_requested_position: ' + Fore.YELLOW + str(self.fingers_are_at_requested_position) + '\n' + \
              '\n' + \
              Fore.RED + 'FAULTS:' '\n' + \
              '\n' + \
              Fore.RED + '  Prior Faults:' '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_action_delayed: ' + Fore.YELLOW + str(self.fault_action_delayed) + '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_missing_activation_bit: ' + Fore.YELLOW + str(self.fault_missing_activation_bit) + '\n' + \
              '\n' + \
              Fore.RED + '  Minor Faults:' '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_max_temperature_exceeded: ' + Fore.YELLOW + str(self.fault_max_temperature_exceeded) + '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_no_communication_1_second: ' + Fore.YELLOW + str(self.fault_no_communication_1_second) + '\n' + \
              '\n' + \
              Fore.RED + '  Major Faults:' '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_under_minimum_voltage: ' + Fore.YELLOW + str(self.fault_under_minimum_voltage) + '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_automatic_release_in_progress: ' + Fore.YELLOW + str(self.fault_automatic_release_in_progress) + '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_internal: ' + Fore.YELLOW + str(self.fault_internal) + '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_activation: ' + Fore.YELLOW + str(self.fault_activation) + '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_overcurrent_triggered: ' + Fore.YELLOW + str(self.fault_overcurrent_triggered) + '\n' + \
              Fore.LIGHTMAGENTA_EX + '     fault_automatic_release_completed: ' + Fore.YELLOW + str(self.fault_automatic_release_completed) + '\n' + \
              '\n' + \
              Fore.LIGHTGREEN_EX + 'requested_position: ' + Fore.WHITE + str(self.requested_position) + ' mm\n' + \
              '\n' + \
              Fore.LIGHTGREEN_EX + 'actual_position: ' + Fore.WHITE + str(self.actual_position) + ' mm\n' + \
              '\n' + \
              Fore.LIGHTGREEN_EX + 'actual_force_motor_current: ' + Fore.WHITE + str(self.actual_force_motor_current) + ' mA\n' + \
              '\n'
        return rep
