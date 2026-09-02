from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = {
    "activity_threshold": "0.30",

    "coast_chunks": "32",
    "confirm_chunks": "2",

    "model_size": "small",
    "stt_device": "cuda",
    "stt_compute_type": "float16",

    "partial_interval": "86400.0",
    "silence_threshold": "0.050",
    "audio_gain": "3.0",
    "port": "8765",
    "open_browser": "false",

    "raw_audio": "false",
    "camera": "true",
    "yaw_offset_deg": "0.0",
    "speaking_threshold": "0.0",
    "debug_ui": "true",
    "debug_port": "8770",
}


def generate_launch_description() -> LaunchDescription:
    declarations = [DeclareLaunchArgument(k, default_value=v) for k, v in ARGUMENTS.items()]
    config = {k: LaunchConfiguration(k) for k in ARGUMENTS}
    return LaunchDescription(declarations + [
        Node(package="doa_separation_node", executable="doa_separation",
             name="doa_separation", output="screen",
             parameters=[{"activity_threshold": config["activity_threshold"],
                          "coast_chunks": config["coast_chunks"],
                          "confirm_chunks": config["confirm_chunks"],
                          "raw_audio": config["raw_audio"]}]),
        Node(package="stt_bridge_node", executable="stt_bridge",
             name="stt_bridge", output="screen",
             parameters=[{"model_size": config["model_size"],
                          "device": config["stt_device"],
                          "compute_type": config["stt_compute_type"],
                          "partial_interval": config["partial_interval"],
                          "silence_threshold": config["silence_threshold"],
                          "audio_gain": config["audio_gain"]}]),
        Node(package="camera_node", executable="camera",
             name="camera", output="screen",
             condition=IfCondition(config["camera"]),
             parameters=[{"yaw_offset_deg": config["yaw_offset_deg"],
                          "speaking_threshold": config["speaking_threshold"]}]),
        Node(package="doa_debug_node", executable="doa_debug",
             name="doa_debug", output="screen",
             condition=IfCondition(config["debug_ui"]),
             parameters=[{"port": config["debug_port"]}]),
        Node(package="ui_bridge_node", executable="ui_bridge",
             name="ui_bridge", output="screen",
             parameters=[{"port": config["port"],
                          "open_browser": config["open_browser"],
                          "speaking_threshold": config["speaking_threshold"]}]),
    ])
