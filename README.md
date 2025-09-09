# LLM Embedded for Robots in ROS2 🤖🦾

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


**LLM-Embedded-for-Robots-in-ROS2** is a ROS2 framework that fuses a local Large Language Model with a full offline speech pipeline so robots can understand and respond to natural language without the cloud.

---
## 📝 Flowchart
<div style="overflow:auto; border:1px solid #eaecef; padding:6px;">
  <object data="Diagrama de Flujo.svg" type="image/svg+xml" width="4400">
    <!-- Fallback para visores viejos -->
    <img src="Diagrama de Flujo.svg" alt="Flowchart" />
  </object>
</div>

<p align="right">
  <a href="https://lucid.app/lucidchart/50ed3019-62f3-460d-a3e3-071d72727e35/view">Open SVG in real size</a>
</p>

---

## 📚 Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
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
sudo apt install -y python3-dev python3-venv build-essential portaudio19-dev curl unzip
sudo snap install yq
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
Run the downloader to fetch any missing models:
```bash
# Install all the models for the LLM Integration
bash "$(ros2 pkg prefix LLM)/share/LLM/scripts/download_models.sh"
```
The script installs everything into your cache (`~/.cache/octopy`, or `$OCTOPY_CACHE` if set).
You’re done when you see:
```bash
"OK. Modelos listos en: $CACHE_DIR ✅ "
```
---

<h2 id="configuration">⚙️ Configuration</h2>

> [!WARNING] 
> LLMs and audio models can be large. Ensure you have enough **disk space** and **RAM/VRAM** for your chosen settings.

All runtime settings live in **`config/config.py`**. They are plain Python constants—edit the file and restart your nodes to apply changes.

### 📦 Model catalog (`models.yml`)

Define which models Octybot uses (LLM, STT, TTS, wake-word) along with their URLs and sample rates.

> ⚠️ **Important:** The **`name`** of every model in `models.yml` must match **exactly** the name you use in `config.py` **and** the name documented in this README (same text and file extension).

### 🔗 Required matching with `config.py`
Use the **same strings** from `models.yml` in your Python config:

> [!TIP]
> If you build with `colcon build --symlink-install`, Python edits are picked up without rebuilding. Otherwise, rebuild and `source install/setup.bash`.

### Importing settings in your code
```python
# if you re-export in config/__init__.py
from .config import AUDIO_LISTENER_SAMPLE_RATE, DEFAULT_MODEL_FILENAME

# otherwise
from .llm_utils.config  import AUDIO_LISTENER_SAMPLE_RATE, DEFAULT_MODEL_FILENAME

```

<h2 id="quick-start">⚡ Quick Start</h2>

### Launch the full pipeline (Wake-Word → STT → LLM → TTS)

Start everything with:

```bash
ros2 launch llm llm.launch.py
```
This launch uses your default microphone. You’ll know the nodes are ready when you see logs like:

```bash
[llm_main-4] [INFO] [xxxxxxxxxx.xxxxxxxxx] [octopy_agent]: Octopy listo ✅  Publica en /transcript

[python-2] [INFO] [xxxxxxxxxx.xxxxxxxxxx] [audio_sink]: AudioSink ▶️ rate=24000 Hz, ch=1, fpb=256, device_index=None

[stt-3] [INFO] [xxxxxxxxxx.xxxxxxxxxx] [silero_stt_node]: Silero listo 🔊 SR=16000ch=1 device=cpu lang=es
[stt-3] Transcribe cuando /flag_wake_word cae de True a False.

[tts-6] [INFO] [xxxxxxxxxx.xxxxxxxxxx] [silero_tts_node]: Silero TTS listo 🔊 rate=24000 device=cpu lang=es speaker=v3_es

```
Now say `ok robot` — the system will start listening and run the pipeline.

<h2 id="usage">🧪 Usage</h2>

### Agent intents (`handle(data, tipo)`)
| `tipo`     | What it does | Input `data` | Output shape |
|---|---|---|---|
| `rag` | Returns `data` as-is (external RAG already resolved). |  Pre-composed **string** from your RAG - `kb.json`| `str` |
| `general` | Free-form Q&A via `llm.answer_general`. | Question | `str` |
| `battery` | Reads `%` via `tool_get_batt()`. | Reads battery status from `/battery_state` (message type: `sensor_msgs/msg/BatteryState`).| `str` like `Mi batería es: 84.0%` (or no-reading msg) |
| `pose` | Reads AMCL pose via `tool_get_pose()`. | Reads battery status from `/amcl_pose` (message type: `sensor_msgs/msg/PoseWithCovarianceStamped`). | `str` (no pose) **or** JSON `{"x","y","yaw_deg","frame"}` |
| `navigate` | Navigate to a named place or generate a short motion. Tries `tool_nav(data)` first (KB/`poses.json`): if found, replies **"Voy"** (execute) or **"Por allá"** (indicate/simulate). If not found, falls back to `llm.plan_motion(data)` → `_clamp_motion(...)` → `natural_move_llm(...)`. | Pre-composed string from your RAG - poses.json and `str` Natural-language place or motion command (e.g., `ve a la enfermería`, `gira 90° y avanza 0.5 m`). | Usually `str`. On fallback may return a **tuple**: `(mensaje, '{"yaw": <deg>, "distance": <m>}' )`. |

> If you consume the agent’s reply topic, handle both cases for `navigate`: always speak/log the **string**; optionally route the **JSON** telemetry if present.

---

### Set your topics (fill these in)

#### Core agent topics
| Purpose | Default | **Your topic** |
|---|---|---|
| Input (String **in**) | `/transcript` | `________________________` |
| Answer (String **out**) | `/answer` | `________________________` |

#### Extra topics required by intents
| Intent (`tipo`) | Needs… | Default | **Your topic** |
|---|---|---|---|
| `battery` | Battery state | `/battery_state` | `________________________` |
| `pose` | AMCL pose | `/amcl_pose` | `________________________` |
| `navigate` | Nav stack topics/actions you use (e.g., Nav2 goal action) | *(your Nav2 setup)* | `________________________` |

#### Voice pipeline (optional, if you use WW→STT→TTS)
| Purpose | Default | **Your topic** |
|---|---|---|
| Raw audio (Int16MultiArray **in**) | `/audio` | `________________________` |
| Raw audio (Int16MultiArray **out**) | `/audio_publisher` | `________________________` |
| Wake-word flag (Bool **out**) | `/flag_wake_word` | `________________________` |


<h2 id="Project Structure">📂 Project Structure</h2>

```text
LLM-Embedded-for-Robots-in-ROS2/
├── src/
|   ├──config
|   |  └──models.yml
|   ├──scripts
|   |  └──download_models.sh
|   ├──launch
|   |  └──llm.launch.py
|   ├──data
|   |  ├──kb.json
|   |  └──poses.json
│   └── llm/                 # ROS2 package: agent, STT, wake word, TTS
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