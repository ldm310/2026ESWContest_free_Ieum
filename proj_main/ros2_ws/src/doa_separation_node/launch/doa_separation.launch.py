from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = {
    "device_name": "ArrayUAC10",
    "device_channels": "6",
    "pulse_source": "alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input",
    "torch_device": "cuda",
    "activity_threshold": "0.3",
    "publish_audio": "true",
    "frame_id": "mic_array",
}


def generate_launch_description() -> LaunchDescription:
    declarations = [DeclareLaunchArgument(name, default_value=value)
                    for name, value in ARGUMENTS.items()]
    parameters = {name: LaunchConfiguration(name) for name in ARGUMENTS}
    return LaunchDescription(declarations + [
        Node(package="doa_separation_node", executable="doa_separation",
             name="doa_separation", output="screen", parameters=[parameters]),
    ])
