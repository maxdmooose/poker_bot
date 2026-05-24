# evolution/file_ops.py

import os
from pathlib import Path
from typing import Dict, List

from config import BOTS_DIR, HOF_DIR, LOG_DIR, SPECIES


def ensure_dirs():
    Path(BOTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(HOF_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    for sp in SPECIES:
        Path(os.path.join(BOTS_DIR, sp)).mkdir(parents=True, exist_ok=True)


def write_file(path: str, content: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def copy_file(src: str, dst: str):
    from shutil import copyfile
    Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
    copyfile(src, dst)


def list_bots_for_species(state: dict, species: str) -> List[Dict]:
    return state["bots"].get(species, [])


def add_bot_to_state(state: dict, bot: Dict):
    sp = bot["species"]
    state["bots"].setdefault(sp, []).append(bot)
