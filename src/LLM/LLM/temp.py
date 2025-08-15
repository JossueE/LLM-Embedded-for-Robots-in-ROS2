#~/llama.cpp/LLM/Llama-3.2-3B-Instruct-Q4_0.gguf
import os, json, site
site.addsitedir(os.path.expanduser('~/ROS2/Octopy/.venv/lib/python3.10/site-packages'))
from llama_cpp import Llama, __version__ as LLAMA_CPP_VER

print("llama_cpp:", LLAMA_CPP_VER)

llm = Llama(

    model_path=os.path.expanduser(os.getenv("OCTOPY_MODEL","~/llama.cpp/LLM/Llama-3.2-3B-Instruct-Q4_0.gguf")),
    n_ctx=2048, n_threads=4, n_gpu_layers=0,
    chat_format="llama-3",           # <— COMODÍN
)
resp = llm.create_chat_completion(
      messages = [
        {
          "role": "system",
          "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

        },
        {
          "role": "user",
          "content": "Extract Jason is 25 years old"
        }
      ],
      tools=[{
        "type": "function",
        "function": {
          "name": "UserDetail",
          "parameters": {
            "type": "object",
            "title": "UserDetail",
            "properties": {
              "name": {
                "title": "Name",
                "type": "string"
              },
              "age": {
                "title": "Age",
                "type": "integer"
              }
            },
            "required": [ "name", "age" ]
          }
        }
      }],
      tool_choice={
        "type": "function",
        "function": {
          "name": "UserDetail"
        }
      }
)


# --- VALIDACIÓN DE TOOL CALLING ---
print("\n[RAW RESPONSE]")
print(json.dumps(resp, indent=2, ensure_ascii=False))

msg = (resp.get("choices") or [{}])[0].get("message", {})

# Soporta tanto el esquema nuevo (tool_calls) como el viejo (function_call)
tool_calls = msg.get("tool_calls")
if not tool_calls and msg.get("function_call"):
    # Normaliza a lista para tratarlo igual
    fc = msg["function_call"]
    tool_calls = [{"type": "function", "function": {"name": fc.get("name"), "arguments": fc.get("arguments")}}]

if not tool_calls:
    print("\n❌ VALIDACIÓN: el modelo NO devolvió tool calls (puede que el chat_format o la versión no soporten tools).")
else:
    print(f"\n✅ VALIDACIÓN: el modelo devolvió {len(tool_calls)} tool_call(s).")
    valid_names = {t["function"]["name"] for t in tools}
    for i, tc in enumerate(tool_calls, 1):
        fn = (tc.get("function") or {})
        name = fn.get("name")
        args_raw = fn.get("arguments", "")
        print(f"  · Tool #{i}: name={name!r}")

        # 1) nombre válido
        if name not in valid_names:
            print(f"    ⚠️ Nombre de tool no registrado: {name!r} (esperado uno de {valid_names})")
        else:
            print("    ✓ Nombre coincide con tools registradas")

        # 2) argumentos parseables
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            print(f"    ✓ Arguments parseados: {args}")
        except Exception as e:
            print(f"    ❌ Error parseando arguments como JSON: {e}\n    arguments_raw={args_raw!r}")
            args = None

        # 3) chequeo rápido del esquema esperado de esta tool
        if name == "UserDetail" and isinstance(args, dict):
            missing = [k for k in ("name","age") if k not in args]
            if missing:
                print(f"    ❌ Faltan campos requeridos: {missing}")
            else:
                print("    ✓ Campos requeridos presentes (name, age)")
                # Ejemplo: “simulación” de ejecutar la tool
                print(f"    → Simulación de ejecución: User(name={args['name']!r}, age={args['age']})")