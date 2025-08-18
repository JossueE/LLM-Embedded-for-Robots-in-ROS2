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

## 📋 Requirements
**OS & Tools**

- Ubuntu 22.04
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- Python 3.10.12 
- Git, CMake, colcon
- (Optional) NVIDIA CUDA for GPU acceleration
- (Suggestion) Use Virtual Enviroment

**Models**
- A `.gguf` LLM compatible with `llama.cpp`
- The project is working with `Llama-3.2-3B-Instruct-Q4_0` 
  - **Download:** [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main)
[!IMPORTANT] If you want to change the model, remember to use which one who can Support Tool Calls and Functions
- *(Optional)* A **faster-whisper** model if you use the STT node

**System packages (recommended)**
```bash
sudo apt update
sudo apt install -y python3-rosdep python3-colcon-common-extensions \
                    build-essential cmake git git-lfs
# For Speech to Text:
sudo apt install -y portaudio19-dev
```
---

## ⚙️ Installation 

**1. Clone**
```bash
git clone https://github.com/JossueE/LLM-Embedded-for-Robots-in-ROS2.git
cd LLM-Embedded-for-Robots-in-ROS2
```

**2. (Recommended) Virtual Enviroment**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```
**3. Install Dependencies**
```bash
pip install -r requirements.txt 
pending...
```
**4. Set Your LLM**
```bash
pending...
```
or paste in the terminal
```bash
export OCTOPY_MODEL=/path/to/your/model.gguf

```
**5. Build**

```bash
#To make colcon build you have to stay in LLM-Embedded-for-Robots-in-ROS2

colcon build
source install/setup.bash
```
---

**1. Start the agent node**
[!NOTE] Remember to Start Your Virutual Enviroment 
```bash
ros2 run LLM agent
```
you're supposed to see:
```bash
[INFO] [xxxxxxxxxx.xxxxxxxxx] [octopy_agent]: Octopy listo ✅  Publica en /octopy/ask
```
a common mistake:
```bash
#If you see
Package 'LLM' not found
```
paste:
```bash
source install/setup.bash 
ros2 run LLM agent
```
**2. Test the Node**
While having the last terminal open, start a new one. 
[!IMPORTANT] Everytime that we open a New Terminal, we will source
```bash
#We source the terminal
source install/setup.bash
```
then paste:
```bash
#We are going to check if the LLM is working
ros2 topic pub -1 /octopy/ask std_msgs/String "data: '¿Quién eres?'"
```
in the last terminal we have to see:
```bash
[INFO] [xxxxxxxxxx.xxxxxxxx] [octopy_agent]: ANS: Soy Octybot, tu amigo robot.
```

---
## 🧪 Examples
**A) Typical queries to the agent**
- Robot status
```bash
ros2 topic pub /octopy/ask std_msgs/String "data: '¿Cuál es tu nivel de Batería?'"
```
- Current pose (tool-call get_current_pose)
```bash
ros2 topic pub /octopy/ask std_msgs/String "data: '¿Dime cuál es tu Pose?'"
```
- Go to a named pose (tool-call lookup_named_pose)
```bash
ros2 topic pub /octopy/ask std_msgs/String "data: 'Ve a la enfermería'"
```
- Knowledge base direct answers (kb.json)\
  If a message matches a __trigger__, the agent responds from the KB without invoking the LLM.

**B) Edit the knowledge base (kb.json)**
Minimal structure:

```bash
{
  "knowledge": [
    {
      "triggers": ["where is the charging station", 
                   "go to charger", 
                   "ve a estación de carga"],
      "answer": "The charging station is at X=1.2, Y=-0.5."
    }
  ]
}
```
Add entries and restart the agent to preload the KB.

**C) Add a new Tool Call**
__1. Implement the function in the agent node (e.g., `get_battery_status()`).__

```bash
def _add_your_new_function(self):
    """
    Example of a function implementation (to be completed by the user).

    NOTES:
    - Define the logic of your function.
    - Decide if your function needs input arguments.
    - Return the result that makes sense for your use case 
      (e.g., integer, boolean, object, or None).
    - You can raise exceptions if needed to handle errors.
    """
```
__2. Register it in the tools dictionary__
 - It defines which functions the agent can call.
 - Add a new dictionary entry to the list returned by `_tools_spec()`.
 - Follow the same structure:
 Check the official documentation [FUNCTION CALLING 🔽✏️](https://llama-cpp-python.readthedocs.io/en/latest/#function-calling)


```bash
{
    "type": "function",
    "function": {
        "name": "<function_name>",
        "description": "<short explanation of what it does>",
        "parameters": {
            "type": "object",
            "properties": {
                "<arg1>": {"type": "string"},
                "<arg2>": {"type": "boolean"},
                # ...
            },
            "required": ["<arg1>", "..."]  # optional if no required params
        }
    }
}
```
Example:
```bash
def _tools_spec(self) -> List[Dict[str, Any]]:
    return [
        # existing tools...
        {
            "type": "function",
            "function": {
                "name": "new_tool_name",
                "description": "What this tool does.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"}
                    },
                    "required": ["param1"]
                }
            }
        }
    ]
```

If you want to see more about [CHECK THIS LINK 🔽✏️](https://medium.com/@jimsweb/designing-custom-functions-for-llms-with-locally-hosted-llama-cpp-4d8ed3f9226a)

__3.Mention it in the LLM system prompt (so the model knows it exists).__

The `system_prompt` parameter defines the instructions given to the model at startup.
It tells the assistant who it is (Octopy) and how it should behave, including when to call tools like `get_battery`, `get_current_pose`, or `nav_to_place`.

``` bash
#This can be changed

self.declare_parameter(
    'system_prompt',
    os.environ.get(
        'OCTOPY_SYSTEM_PROMPT',
        "You are Octopy, an assistant for a ROS2 robot. Always use tools when they apply.\n"
        "Rules:\n"
        "- Battery: call get_battery and answer 'My battery is: XX.X%'.\n"
        "- Current pose: call get_current_pose and return ONLY JSON {x,y,yaw_deg,frame}.\n"
        "- Go to place ('go to X', 'navigate to X'): call nav_to_place simulate=false and answer 'Going'.\n"
        "- Pointing ('where is X', 'point to X'): call nav_to_place simulate=true and answer 'Over there'.\n"
        "- Here add the <instruction> and the <new_function> the you need for your system. \n"
        "- Questions outside robot/KB: answer 'I can only respond about the robot and my local knowledge base.'\n"
        "Always respond in Spanish, concise (<=120 words)."
    )
)
```
__4. Add the `tool_call` to the Dispacher.__\
It routes the request to the correct internal method and returns that method’s result.

``` bash
    def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_current_pose":
            return self._tool_get_current_pose()
        if name == "lookup_named_pose":
            return self._tool_lookup_named_pose(args.get("name", ""))
        if name == "kb_lookup":
            return self._tool_kb_lookup(args.get("q", ""))
        if name == "get_battery":
            return self._tool_get_battery()
        if name == "nav_to_place":
            return self._tool_nav_to_place(args.get("text", ""), bool(args.get("simulate", False)))
        if name == "<functio_name>":
            return self.<new_function>(define the arguments if you need)))
        return {"error": "tool_desconocida", "name": name}

)
```
