# evolution/hall_of_fame.py

import os
from pathlib import Path
from typing import Dict

from config import HOF_DIR, HOF_EV_THRESHOLD_BB100
from file_ops import copy_file


def maybe_add_to_hof(bot: Dict):
    ev = bot.get("ev", None)
    if ev is None or ev < HOF_EV_THRESHOLD_BB100:
        return
    src = bot["path"]
    name = os.path.basename(src)
    dst = os.path.join(HOF_DIR, name)
    copy_file(src, dst)
