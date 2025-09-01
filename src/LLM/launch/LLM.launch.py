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
            executable="audio_publisher",
            output="screen",
        )
    )

    ld.add_action(
        Node(
            package="LLM",
            executable="wake_word",
            output="screen",
        )
    )

    ld.add_action(
        Node(
            package="LLM",
            executable="stt",
            output="screen",
        )
    )

    ld.add_action(
        Node(
            package="LLM",
            executable="llm_main",
            output="screen",
        )
    )

    ld.add_action(
        Node(
            package="LLM",
            executable="tts",
            output="screen",
        )
    )
    
    return ld