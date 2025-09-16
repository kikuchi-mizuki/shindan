import json
import os
import threading
from typing import Dict, List

STORE_PATH = os.path.join(os.path.dirname(__file__), "normalize_store.json")
_lock = threading.Lock()

# In-memory stores (loaded at import)
MISREAD: Dict[str, str] = {"ロ腔": "口腔"}
SYN: Dict[str, str] = {}
ATC_CACHE: Dict[str, List[str]] = {}
TAGS: Dict[str, List[str]] = {}


def _load_store():
    if not os.path.exists(STORE_PATH):
        return
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        MISREAD.update(data.get("misread", {}))
        SYN.update(data.get("synonym", {}))
        ATC_CACHE.update(data.get("atc", {}))
        TAGS.update(data.get("tag", {}))
    except Exception:
        # ignore corrupted store; continue with defaults
        pass


def _save_store():
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    data = {
        "misread": MISREAD,
        "synonym": SYN,
        "atc": ATC_CACHE,
        "tag": TAGS,
    }
    tmp = STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STORE_PATH)


def save_to_db(kind: str, key: str, value):
    with _lock:
        if kind == "misread":
            MISREAD[key] = value
        elif kind == "synonym":
            SYN[key] = value
        elif kind == "atc":
            if isinstance(value, list):
                ATC_CACHE[key] = value
            else:
                ATC_CACHE[key] = [v for v in str(value).split(",") if v]
        elif kind == "tag":
            if isinstance(value, list):
                TAGS[key] = value
            else:
                TAGS[key] = [v for v in str(value).split(",") if v]
        _save_store()


def fix_misread(s: str) -> str:
    t = s
    for bad, good in MISREAD.items():
        t = t.replace(bad, good)
    return t


def learn_misread(bad: str, good: str):
    save_to_db("misread", bad, good)


def learn_synonym(alias: str, generic: str):
    save_to_db("synonym", alias, generic)


def cache_atc(generic: str, atc_list: List[str]):
    save_to_db("atc", generic, atc_list)


def learn_tag(generic: str, *tags: str):
    cur = set(TAGS.get(generic, []))
    cur.update(tags)
    save_to_db("tag", generic, sorted(cur))


def get_synonym(alias: str) -> str:
    return SYN.get(alias, alias)


def get_tags(generic: str) -> List[str]:
    return TAGS.get(generic, [])


# Load at import
_load_store()


