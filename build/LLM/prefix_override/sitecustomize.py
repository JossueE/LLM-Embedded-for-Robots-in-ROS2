import sys
if sys.prefix == '/home/snorlix/ROS2/Octopy/.venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/snorlix/ROS2/Octopy/install/LLM'
