from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()
    ld.add_action(
        Node(
            package="LLM",
            executable="audio_listener",
            output="screen",
        )
    )

    ld.add_action(
        Node(
            package="LLM",
            executable="wake_word_detector",
            output="screen",
        )
    )
    
    return ld