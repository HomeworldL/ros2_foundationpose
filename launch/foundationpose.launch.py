from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    params = PathJoinSubstitution(
        [
            FindPackageShare("ros2_foundationpose"),
            "config",
            "foundationpose_config.yaml",
        ]
    )
    
    params_display = PathJoinSubstitution(
        [
            FindPackageShare("ros2_foundationpose"),
            "config",
            "pose_display.yaml",
        ]
    )
    
    return LaunchDescription(
        [
            Node(
                package="ros2_foundationpose",
                executable="foundationpose_node.py",
                name="foundationpose_node",
                parameters=[params],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },
            ),
            Node(
                package="ros2_foundationpose",
                executable="pose_display.py",
                name="pose_display_node",
                parameters=[params_display],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },
            )
        ]
    )
