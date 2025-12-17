from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_SYSTEM_V1_PATH = _PROMPTS_DIR / "system_v1.txt"

with _SYSTEM_V1_PATH.open("r", encoding="utf-8") as f:
    system_prompt = f.read()

PROMPT_VERSION = "system_v1"
