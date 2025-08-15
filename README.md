# LLM Embedded for Robots in ROS2 🤖🦾

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**LLM Embedded for Robots in ROS2** es un agente inteligente que integra un **Large Language Model** local (ej. LLaMA.cpp) con **tool-calling** para controlar y consultar funciones de un robot usando lenguaje natural.  
Está optimizado para correr **en hardware embebido**, como PCs industriales o laptops ligeras con Linux, sin depender de la nube.

---

## 🚀 Características

- **Integración ROS2**: Suscripción y publicación en tópicos `/octopy/ask` y `/octopy/answer`.
- **Tool-calling**: El modelo puede invocar funciones específicas:
  - `get_current_pose()` → Pose actual del robot desde `/amcl_pose`.
  - `lookup_named_pose(name)` → Poses estáticas predefinidas.
  - `kb_lookup(query)` → Consulta en un *knowledge base* local (JSON).
- **Soporte para modelos locales**: Compatible con [llama.cpp](https://github.com/ggerganov/llama.cpp) y variantes `.gguf`.
- **Precarga de conocimiento**: Carga inicial de la base de datos local para respuestas instantáneas.
- **Bajo consumo**: Optimizado para CPUs Ryzen 5 e Intel i5, con opción de GPU.

---

## 📂 Estructura del proyecto

```plaintext
.
├── agent_node.py        # Nodo ROS2 principal (suscripción y tool-calling)
├── rack.json            # Base de conocimiento local
├── requirements.txt     # Dependencias de Python
├── launch/              # Archivos de lanzamiento ROS2
└── README.md            # Este archivo
