# LLM.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    common = dict(package="llm", output="screen")

    return LaunchDescription([
        Node(**common, executable="audio_listener", name="audio_listener"),
        Node(**common, executable="audio_publisher", name="audio_sink",
             parameters=[{"rate": 24000, "device": "cpu"}]),

        Node(**common, executable="wake_word", name="wake_word"),

        Node(**common, executable="stt", name="silero_stt",
             parameters=[{"language": "es", "device": "cpu"}]),

        Node(**common, executable="llm_main", name="octopy_agent"),

        Node(**common, executable="tts", name="silero_tts",
             parameters=[{"rate": 24000, "device": "cpu", "voice": "es_2"}]),
    ])
