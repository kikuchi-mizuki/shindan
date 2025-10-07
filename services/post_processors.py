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
    """特定薬剤の剤形を修正（ナルフラフィン塩酸塩のみカプセルに修正）"""
    generic = drug.get('generic', '')
    strength = drug.get('strength', '')
    
    # ナルフラフィン塩酸塩（µg単位）→ カプセル
    if "ナルフラフィン" in generic:
        drug["dose"] = "1カプセル"
        drug["dose_form"] = "カプセル"
        # strengthの単位も修正（2.5μg/錠 → 2.5μg/カプセル）
        if strength:
            drug["strength"] = strength.replace("/錠", "/カプセル").replace("錠", "カプセル")
    
    # リナクロチドは日本では錠剤が正しいので修正しない
    
    return drug

def fix_frequency_normalization(drug: dict) -> dict:
    """頻度の正規化（朝夕→朝・夕、眠前→就寝前）"""
    freq = drug.get('freq', '')
    if freq:
        # 朝夕 → 朝・夕
        if '朝夕' in freq and '朝・夕' not in freq:
            drug['freq'] = freq.replace('朝夕', '朝・夕')
        # 眠前 → 就寝前
        elif '眠前' in freq:
            drug['freq'] = freq.replace('眠前', '就寝前')
    
    return drug

def fix_tramadol_display(drug: dict) -> dict:
    """トラマドール配合の表示を修正（1回1錠、1日3回）"""
    generic = drug.get('generic', '')
    if 'トラマドール' in generic and 'アセトアミノフェン' in generic:
        # 用量を「1回1錠、1日3回」に修正
        drug['dose'] = '1回1錠'
        drug['freq'] = '1日3回'
        # strengthを「配合錠」に修正
        if 'strength' in drug and drug['strength'] == '不明':
            drug['strength'] = '配合錠'
    
    return drug

def fix_entresto_dosage(drug: dict) -> dict:
    """エンレスト（サクビトリル/バルサルタン）の用量を修正（2錠→1錠）"""
    generic = drug.get('generic', '')
    if 'サクビトリル' in generic and 'バルサルタン' in generic:
        # 2錠 → 1錠に修正
        if drug.get('dose') == '2錠':
            drug['dose'] = '1錠'
    
    return drug


