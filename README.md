# LLM Embedded for Robots in ROS2 🤖🦾

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


**LLM-Embedded-for-Robots-in-ROS2** is a ROS2 framework that fuses a local Large Language Model with a full offline speech pipeline so robots can understand and respond to natural language without the cloud.

---
## 📝 Flowchart
[See the full Diagram (SVG)](<Diagrama de Flujo.svg>)

<p align="center">
  <img src="Diagrama de Flujo.svg" alt="Diagrama de Flujo" />
</p>


---

## 📚 Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing) 
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

<h2 id="installation">🛠️ Installation</h2>
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
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential portaudio19-dev
```

```bash
# Clone the repository
git clone https://github.com/JossueE/LLM-Embedded-for-Robots-in-ROS2.git
cd LLM-Embedded-for-Robots-in-ROS2
```
```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```
> [!TIP]
> Whit (.venv) Active

```bash
# Install dependencies
pip install -r requirements.txt
```
```bash
# Build ROS2 packages
colcon build
source install/setup.bash
```
---

<h2 id="quick-start">⚡ Quick Start</h2>
Run the example launch file to start the wake-word → STT → LLM → TTS pipeline:
```bash
ros2 launch LLM LLM.launch.py
```
The fisrt time, the models will be donwloaded, so it could take a little bit. Don't say anithing until you see and don't stop the process.

```bash
[llm_main-4] [INFO] [xxxxxxxxxx.xxxxxxxxx] [octopy_agent]: Octopy listo ✅  Publica en /transcript
```
and
```bash

[stt-3] [INFO] [xxxxxxxxxx.xxxxxxxxxx] [silero_stt_node]: Silero listo 🔊 SR=16000ch=1 device=cpu lang=es
[stt-3] Transcribe cuando /flag_wake_word cae de True a False.

[tts-6] [INFO] [xxxxxxxxxx.xxxxxxxxxx] [silero_tts_node]: Silero TTS listo 🔊 rate=24000 device=cpu lang=es speaker=v3_es

[python-2] [INFO] [xxxxxxxxxx.xxxxxxxxxx] [audio_sink]: AudioSink ▶️ rate=24000 Hz, ch=1, fpb=256, device_index=None

```

### Test just a Node
You could use the follow example or try to speak in the microphone
> [!TIP]
> By this way you are sure that you call your virtual env

```bash
# Run the LLM agent Node
home/<your user>/LLM-Embedded-for-Robots-in-ROS2/.venv/bin/python3 /home/<your user>/LLM-Embedded-for-Robots-in-ROS2/install/LLM/lib/LLM/llm_main
```

Mic → Wake Word → STT → LLM/Tools → TTS → Speaker

<h2 id="configuration">⚙️ Configuration</h2>
> [!WARNING]
> LLMs and audio models can be large. Ensure you have enough **disk space** and **RAM/VRAM** for your chosen settings.

All runtime settings live in **`config/config.py`**. They are plain Python constants—edit the file and restart your nodes to apply changes.

> [!TIP]
> If you build with `colcon build --symlink-install`, Python edits are picked up without rebuilding. Otherwise, rebuild and `source install/setup.bash`.

### Importing settings in your code
```python
# if you re-export in config/__init__.py
from .config import AUDIO_LISTENER_SAMPLE_RATE, DEFAULT_MODEL_FILENAME

# otherwise
from .llm_utils.config  import AUDIO_LISTENER_SAMPLE_RATE, DEFAULT_MODEL_FILENAME

```
<h2 id="Project Structure">📂 Project Structure</h2>

```text
LLM-Embedded-for-Robots-in-ROS2/
├── src/
|   ├──data
|   |  ├──kb.json
|   |  └── poses.json
│   └── LLM/                 # ROS2 package: agent, STT, wake word, TTS
│       ├──llm_utils
│       │  ├──data.py 
│       │  ├──llm_client.py 
│       │  ├──llm_intentions.py 
│       │  ├──llm_router.py 
│       │  └──llm_tools.py 
│       ├──audio_publisher.py 
│       ├──audio_listener.py 
│       ├──llm_main.py 
│       ├──speech_to_text.py 
│       ├──text_to_speech.py 
│       └──wake_word_detector.py
├── models.yml               # Auto-downloaded model list
├── requirements.txt
└── README.md
```
<h2 id="usage">🧪 Usage</h2>
### ROS Topics
- `/audio` – raw audio input
- `/flag_wake_word` – wake word detection flag
- `/transcript` – speech-to-text output
- `/octopy/ask` – text questions to the LLM agent
- `/octopy/answer` – agent replies
- `/battery_state` – battery feedback for tools
- `/amcl_pose` – pose feedback for tools

<h2 id="contributing">🤝 Contributing</h2>
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

---

<h2 id="license">📄 License</h2>
This project is licensed under the [MIT License](LICENSE).

---

<h2 id="acknowledgements">🙏 Acknowledgements</h2>
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Vosk](https://alphacephei.com/vosk/)
- [Silero Models](https://github.com/snakers4/silero-models)
- [Qwen Models](https://huggingface.co/Qwen)
- The ROS2 community