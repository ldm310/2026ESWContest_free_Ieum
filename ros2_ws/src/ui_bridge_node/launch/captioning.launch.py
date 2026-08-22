from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = {
    "activity_threshold": "0.45",
    "coast_chunks": "10",
    "model_size": "tiny",
    "silence_threshold": "0.004",
    "audio_gain": "1.0",
    "port": "8765",
    "open_browser": "false",
    "bypass": "false",
    "bypass_channel": "0",
    "camera": "true",
    "show_window": "false",
}


def generate_launch_description() -> LaunchDescription:
    declarations = [DeclareLaunchArgument(k, default_value=v) for k, v in ARGUMENTS.items()]
    config = {k: LaunchConfiguration(k) for k in ARGUMENTS}
    return LaunchDescription(declarations + [
        Node(package="doa_separation_node", executable="doa_separation",
             name="doa_separation", output="screen",
             parameters=[{"activity_threshold": config["activity_threshold"],
                          "coast_chunks": config["coast_chunks"],
                          "bypass": config["bypass"],
                          "bypass_channel": config["bypass_channel"]}]),
        Node(package="stt_bridge_node", executable="stt_bridge",
             name="stt_bridge", output="screen",
             parameters=[{"model_size": config["model_size"],
                          "silence_threshold": config["silence_threshold"],
                          "audio_gain": config["audio_gain"]}]),
        Node(package="camera_node", executable="camera",
             name="camera", output="screen",
             condition=IfCondition(config["camera"]),
             parameters=[{"show_window": config["show_window"]}]),
        Node(package="ui_bridge_node", executable="ui_bridge",
             name="ui_bridge", output="screen",
             parameters=[{"port": config["port"],
                          "open_browser": config["open_browser"]}]),
    ])
