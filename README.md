# LLM Embedded for Robots in ROS2 🤖🦾

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**LLM Embedded for Robots in ROS2** is an intelligent agent that integrates a **local Large Language Model** (e.g., LLaMA.cpp) with **tool-calling** to control and query robot functions using natural language.  
It is optimized to run **on embedded hardware**, such as industrial PCs or lightweight laptops running Linux, without relying on cloud services.

---

## 🚀 Features

- **ROS2 Integration**: Subscribes and publishes to `/octopy/ask` and `/octopy/answer` topics.
- **Tool-calling**: The model can trigger specific functions:
  - `get_current_pose()` → Current robot pose from `/amcl_pose`.
  - `lookup_named_pose(name)` → Predefined static poses.
  - `kb_lookup(query)` → Queries the local *knowledge base* (JSON).
- **Local model support**: Compatible with [llama.cpp](https://github.com/ggerganov/llama.cpp) `.gguf` models.
- **Knowledge preloading**: Loads the local knowledge base at startup for instant responses.
- **Low resource usage**: Optimized for Ryzen 5 or Intel i5 CPUs, with optional GPU acceleration.

---

## 📂 Project Structure

```plaintext
.
├── agent_node.py        # Main ROS2 node (subscription + tool-calling)
├── rack.json            # Local knowledge base
├── requirements.txt     # Python dependencies
├── launch/              # ROS2 launch files
└── README.md            # This file

