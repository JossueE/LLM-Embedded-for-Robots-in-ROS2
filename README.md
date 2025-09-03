# LLM Embedded for Robots in ROS2 🤖🦾

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


**LLM-Embedded-for-Robots-in-ROS2** is a ROS2 framework that fuses a local Large Language Model with a full offline speech pipeline so robots can understand and respond to natural language without the cloud.

---

## 📚 Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing) 
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🛠️ Installation
> [!IMPORTANT]
> Ensure ROS2 Humble and Python ≥3.10 are installed before continuing.

### Prerequisites
- Ubuntu 22.04
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- Python 3.10.12 
- Git, CMake, colcon
- (Optional) NVIDIA CUDA for GPU acceleration
- (Suggestion) Use Virtual Enfoment


### Setup
```bash
$ sudo dnf install portaudio-devel python3-devel
```

```bash
# Clone the repository
$ git clone https://github.com/JossueE/LLM-Embedded-for-Robots-in-ROS2.git
$ cd LLM-Embedded-for-Robots-in-ROS2
```
```bash
# Create a virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate
$ python -m pip install -U pip
```
```bash
# Install dependencies
(.venv) $ pip install -r requirements.txt
```
```bash
# Build ROS2 packages
(.venv) $ colcon build
(.venv) $ source install/setup.bash
```
---

## ⚡ Quick Start
Run the example launch file to start the wake-word → STT → LLM → TTS pipeline:
```bash
(.venv) $ ros2 launch LLM LLM.launch.py
```
The fisrt time, the models will be donwloaded, so it could take a little bit. Don't say anithing until you see.
```bash
[llm_main-4] [INFO] [xxxxxxxxxx.xxxxxxxxx] [octopy_agent]: Octopy listo ✅  Publica en /transcript
```
and
```bash
[stt-3] [INFO] [xxxxxxxxxx.xxxxxxxxxx] [silero_stt_node]: Silero listo 🔊 SR=16000ch=1 device=cpu lang=es
[stt-3] Transcribe cuando /flag_wake_word cae de True a False.

[tts-6] [INFO] [1756768229.614419483] [silero_tts_node]: Silero TTS listo 🔊 rate=24000 device=cpu lang=es speaker=v3_es
```

### Test just a Node
You could use the follow example or try to speak in the microphone
```bash
# Run the LLM agent Node
$ home/<your user>/LLM-Embedded-for-Robots-in-ROS2/.venv/bin/python3 /home/<your user>/LLM-Embedded-for-Robots-in-ROS2/install/LLM/lib/LLM/llm_main
```
> [!TIP]
> By this way you are sure that you call your virtual env

Mic → Wake Word → STT → LLM/Tools → TTS → Speaker
```

## 📂 Project Structure
```text
LLM-Embedded-for-Robots-in-ROS2/
├── src/
|   ├──config
|   |  ├──kb.json
|   |  └── poses.json
│   └── LLM/                 # ROS2 package: agent, STT, wake word, TTS
│       ├──llm_utils
│       │  ├──llm_client.py 
│       │  ├──llm_intentions.py 
│       │  ├──llm_router.py 
│       │  └──llm_tools.py 
│       ├──audio_listener.py 
│       ├──llm_main.py 
│       ├──speech_to_text.py 
│       └──wake_word_detector
├── models.yml               # Auto-downloaded model list
├── requirements.txt
└── README.md
```
## 🧪 Usage
### ROS Topics
- `/audio` – raw audio input
- `/flag_wake_word` – wake word detection flag
- `/transcript` – speech-to-text output
- `/octopy/ask` – text questions to the LLM agent
- `/octopy/answer` – agent replies
- `/battery_state` – battery feedback for tools
- `/amcl_pose` – pose feedback for tools

## ⚙️ Configuration
> [!WARNING]
> Large models require significant disk space; check `models.yml` for sizes.

### Performance
```bash
# Limit CPU threads used by llama.cpp
$ export OCTOPY_THREADS=4
```
Other optional variables: `OCTOPY_CTX`, `OCTOPY_N_BATCH`, `OCTOPY_N_GPU_LAYERS`.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

---

## 📄 License   
This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Vosk](https://alphacephei.com/vosk/)
- [Silero Models](https://github.com/snakers4/silero-models)
- [Qwen Models](https://huggingface.co/Qwen)
- The ROS2 community