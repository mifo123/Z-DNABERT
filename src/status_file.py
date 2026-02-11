# src/status_file.py
import json
import time
import os
import pathlib
from typing import Dict, Any


def update_status_file(status_path: pathlib.Path, patch: Dict[str, Any]) -> None:
    """
    Bezpečne (atomicky) aktualizuje JSON status súbor.

    - ak súbor neexistuje -> vytvorí ho z `patch`
    - ak existuje        -> načíta, urobí update(patch), zapíše späť
    - vždy doplní `updated_at` (UNIX timestamp, sekundy)
    """
    status_path = pathlib.Path(status_path)
    data: Dict[str, Any] = {}

    if status_path.exists():
        try:
            with status_path.open("r") as f:
                data = json.load(f)
        except Exception:
            # ak je status rozbitý, začneme odznova
            data = {}

    data.update(patch)
    data["updated_at"] = time.time()

    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    os.replace(tmp, status_path)
