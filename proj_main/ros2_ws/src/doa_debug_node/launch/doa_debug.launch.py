from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = {

    "host": "127.0.0.1",
    "port": "8770",
}


def generate_launch_description() -> LaunchDescription:
    declarations = [DeclareLaunchArgument(k, default_value=v) for k, v in ARGUMENTS.items()]
    config = {k: LaunchConfiguration(k) for k in ARGUMENTS}
    return LaunchDescription(declarations + [
        Node(package="doa_debug_node", executable="doa_debug",
             name="doa_debug", output="screen",
             parameters=[{"host": config["host"],
                          "port": ParameterValue(config["port"], value_type=int)}]),
    ])
