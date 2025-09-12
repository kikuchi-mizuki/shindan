from typing import Dict


def fix_picosulfate_form(drug: Dict) -> Dict:
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


