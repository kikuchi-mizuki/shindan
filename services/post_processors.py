from typing import List, Any, Tuple, Optional


def fix_picosulfate_form(drug: dict) -> dict:
    """ピコスルファートNa（ラキソベロン）の剤形・用量をOCRの表記揺れから補正する。
    - raw/brandに「液」を含む → 0.75mg/mL, 剤形=液
    - raw/brandに「錠」を含む → 2.5mg（既存があれば優先）, 剤形=錠
    """
    raw = f"{drug.get('raw') or ''}{drug.get('brand') or ''}"
    name = f"{drug.get('generic') or ''}{drug.get('brand') or ''}"
    if ("ピコスルファート" in name) or ("ラキソベロン" in raw):
        if "液" in raw:
            drug["strength"] = "0.75mg/mL"
            drug["dose_form"] = "液"
        elif "錠" in raw:
            drug["strength"] = drug.get("strength") or "2.5mg"
            drug["dose_form"] = "錠"
    return drug

def fix_dosage_forms(drug: dict) -> dict:
    """特定薬剤の剤形を修正（µg単位の薬剤はカプセル優先）"""
    generic = drug.get('generic', '')
    strength = drug.get('strength', '')
    
    # ナルフラフィン塩酸塩（µg単位）→ カプセル
    if "ナルフラフィン" in generic and "µg" in strength:
        drug["dose"] = "1カプセル"
        drug["dose_form"] = "カプセル"
    
    # リナクロチド（µg単位）→ カプセル、用法を朝食前に修正
    elif "リナクロチド" in generic and "µg" in strength:
        drug["dose"] = "1カプセル"
        drug["dose_form"] = "カプセル"
        drug["freq"] = "朝食前"
        drug["frequency"] = "朝食前"
    
    return drug


