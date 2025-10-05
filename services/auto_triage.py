"""
RP行ごとの自動トリアージ（自動確定/要確認/要再撮影）
最小実装: レイアウト特化抽出 + 正規化 + 一貫性チェック + スコアリング
"""
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CONFUSION_PAIRS = [
    (re.compile(r"アスパラ\-?CA"), re.compile(r"(カリウム|K|マグネシウム|Mg)")),
    (re.compile(r"タケキャブ|ボノプラザン"), re.compile(r"ラベプラゾール")),
]

DOSE_FORM_UNITS = {
    "テープ": ["枚"],
    "錠": ["錠"],
    "カプセル": ["カプセル", "包"],
}

def split_rp_lines(text: str) -> List[str]:
    """処方内容のRP行を抽出 (RP001/RP002 ...)"""
    lines = []
    for m in re.finditer(r"(?m)^\s*RP\d+.*$", text):
        lines.append(m.group(0))
    return lines or [text]

def extract_attrs(line: str) -> Dict[str, Any]:
    """用量・単位等の簡易抽出"""
    strength = None
    m = re.search(r"(\d+)\s*mg", line)
    if m:
        strength = f"{m.group(1)}mg"
    size = None
    m = re.search(r"(\d+)\s*cm\s*×\s*(\d+)\s*cm", line)
    if m:
        size = f"{m.group(1)}cm×{m.group(2)}cm"
    qty = None
    m = re.search(r"(\d+)\s*(錠|枚)", line)
    if m:
        qty = f"{m.group(1)}{m.group(2)}"
    days = None
    m = re.search(r"(\d+)\s*日分", line)
    if m:
        days = f"{m.group(1)}日分"
    dose_form = "テープ" if "テープ" in line or "パップ" in line else ("錠" if "錠" in line else None)
    return {"strength": strength, "size": size, "qty": qty, "days": days, "dose_form": dose_form}

def dict_match_score(name: str, normalized: str) -> float:
    return 1.0 if name == normalized else 0.8 if normalized else 0.0

def unit_form_is_consistent(attrs: Dict[str, Any]) -> bool:
    f = attrs.get("dose_form")
    if not f:
        return True
    units = DOSE_FORM_UNITS.get(f, [])
    if attrs.get("qty") is None:
        return True
    return any(u in attrs["qty"] for u in units)

def in_confusion_pairs(name: str, line: str) -> bool:
    for a, b in CONFUSION_PAIRS:
        if a.search(line) and b.search(line):
            logger.info(f"Confusion pair flagged: {line}")
            return True
    return False

def score_line(name_conf: float, name: str, normalized: str, attrs: Dict[str, Any], line: str) -> float:
    checks = [
        dict_match_score(name, normalized) >= 0.8,
        unit_form_is_consistent(attrs),
        not in_confusion_pairs(name, line),
    ]
    rule_consistency = sum(checks) / len(checks)
    return 0.5 * name_conf + 0.3 * dict_match_score(name, normalized) + 0.2 * rule_consistency

def triage(text: str, ocr_confidence: float = 0.8, normalizer=None) -> List[Dict[str, Any]]:
    """RP行ごとにスコアリングして triage 結果を返す"""
    if normalizer is None:
        from .advanced_normalizer import AdvancedNormalizer
        normalizer = AdvancedNormalizer()

    results = []
    for line in split_rp_lines(text):
        # 粗い薬剤名抽出（名寄せは normalizer に委譲）
        rest = re.sub(r"^\s*RP\d+\s*[:：]?\s*", "", line)
        m = re.search(r"([ぁ-んァ-ヶ一-龥A-Za-z0-9・ー\-]+?)(錠|テープ|カプセル|包)", rest)
        raw_name = (m.group(1) + m.group(2)) if m else rest.strip().split()[0]
        norm = normalizer.normalize_drug_name(raw_name)
        normalized = norm.get("normalized_name", raw_name)
        attrs = extract_attrs(line)
        s = score_line(ocr_confidence, raw_name, normalized, attrs, line)
        status = "auto_confirm" if s >= 0.85 else ("needs_review" if s >= 0.6 else "fail")
        results.append({
            "raw": line,
            "name": normalized,
            "attrs": attrs,
            "score": round(s, 3),
            "status": status,
        })
    return results


