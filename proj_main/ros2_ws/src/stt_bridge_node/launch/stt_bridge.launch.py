from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = {
    "model_size": "small",
    "language": "ko",
    "track_timeout_sec": "2.0",
    "publish_partial": "true",
}


def generate_launch_description() -> LaunchDescription:
    declarations = [DeclareLaunchArgument(k, default_value=v) for k, v in ARGUMENTS.items()]
    parameters = {k: LaunchConfiguration(k) for k in ARGUMENTS}
    return LaunchDescription(declarations + [
        Node(package="stt_bridge_node", executable="stt_bridge",
             name="stt_bridge", output="screen", parameters=[parameters]),
    ])
