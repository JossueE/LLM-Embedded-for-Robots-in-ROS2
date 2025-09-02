# LLM_venv.launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
from ament_index_python.packages import get_package_prefix
from pathlib import Path
import os, sys

def _find_site_packages() -> str | None:
    """Find the installed site-packages for the LLM package inside install/."""
    try:
        prefix = Path(get_package_prefix("LLM")).resolve()   # .../install/LLM
    except Exception:
        return None
    lib = prefix / "lib"
    # pick the newest pythonX.Y/site-packages that exists
    candidates = sorted([p for p in lib.glob("python*/site-packages") if p.is_dir()], reverse=True)
    return str(candidates[0]) if candidates else None

def _find_venv_python() -> str:
    """Prefer the active venv; else look for workspace .venv; else fall back to current Python."""
    # 1) If the user activated a venv before launching
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and (Path(venv) / "bin" / "python").exists():
        return str(Path(venv) / "bin" / "python")

    # 2) Try to discover workspace .venv based on install prefix
    try:
        install_prefix = Path(get_package_prefix("LLM")).resolve()  # .../install/LLM
        ws_root = install_prefix.parent.parent                      # .../<workspace>
        cand = ws_root / ".venv" / "bin" / "python"
        if cand.exists():
            return str(cand)
    except Exception:
        pass

    # 3) Last resort: whatever Python is running the launch system
    return sys.executable

def _py_mod(mod: str, env_override: dict) -> ExecuteProcess:
    """Run a module with the chosen Python and environment."""
    python = _find_venv_python()
    return ExecuteProcess(
        cmd=[python, "-m", mod],
        env=env_override,
        output="screen",
    )

def generate_launch_description() -> LaunchDescription:
    # Base environment (inherit the user’s, then ensure our PYTHONPATH includes the installed package)
    env = os.environ.copy()
    site = _find_site_packages()
    if site:
        env["PYTHONPATH"] = (site + os.pathsep + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else site

    py = _find_venv_python()

    return LaunchDescription([
        LogInfo(msg=f"[LLM.launch] Using Python: {py}"),
        LogInfo(msg=f"[LLM.launch] PYTHONPATH (prepended): {env.get('PYTHONPATH','<empty>')}"),

        _py_mod("LLM.audio_listener", env),
        _py_mod("LLM.audio_publisher", env),
        _py_mod("LLM.wake_word_detector", env),
        _py_mod("LLM.speech_to_text", env),
        _py_mod("LLM.llm_main", env),
        _py_mod("LLM.text_to_speech", env),
    ])