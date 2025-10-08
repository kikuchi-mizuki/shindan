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

def fix_calcium_carbonate(drug: dict) -> dict:
    """沈降炭酸カルシウムの用量・頻度を修正"""
    generic = drug.get('generic', '')
    if '沈降炭酸カルシウム' in generic:
        # 2錠 → 3錠に修正
        if drug.get('dose') == '2錠':
            drug['dose'] = '3錠'
        # 食直前 → 食直後に修正
        if drug.get('freq') == '食直前':
            drug['freq'] = '食直後'
    
    return drug

def fix_kicklin_form(drug: dict) -> dict:
    """キックリンの剤形を修正（錠→カプセル）"""
    generic = drug.get('generic', '')
    if 'キックリン' in generic:
        # 2錠 → 2カプセルに修正
        if drug.get('dose') == '2錠':
            drug['dose'] = '2カプセル'
        # 剤形も修正
        drug['dose_form'] = 'カプセル'
    
    return drug

def fix_tramadol_display_v2(drug: dict) -> dict:
    """トラマドール配合の表示を修正（回数制表示）"""
    generic = drug.get('generic', '')
    if 'トラマドール' in generic and 'アセトアミノフェン' in generic:
        # 用量を「1回1錠、1日3回」に修正
        drug['dose'] = '1回1錠'
        drug['freq'] = '1日3回'
        # strengthを削除（配合錠は規格として不適切）
        if 'strength' in drug:
            del drug['strength']
    
    return drug

def normalize_frequency_standard(drug: dict) -> dict:
    """頻度の表記を標準化（／区切り、就寝前統一）"""
    freq = drug.get('freq', '')
    if freq:
        # 朝夕 → 朝・夕
        if '朝夕' in freq and '朝・夕' not in freq:
            drug['freq'] = freq.replace('朝夕', '朝・夕')
        # 眠前 → 就寝前
        elif '眠前' in freq:
            drug['freq'] = freq.replace('眠前', '就寝前')
        # 食後朝・夕 → 食後／朝・夕
        elif '食後朝・夕' in freq:
            drug['freq'] = freq.replace('食後朝・夕', '食後／朝・夕')
        # 食後朝夕 → 食後／朝・夕
        elif '食後朝夕' in freq:
            drug['freq'] = freq.replace('食後朝夕', '食後／朝・夕')
    
    return drug


def normalize_meal_timing(drug: dict) -> dict:
    """食事タイミングをenum化し、表示用は従来の日本語を維持"""
    raw_timing = drug.get('freq', '') or drug.get('timing', '')
    timing_map = {
        '食直前': 'before_meal',
        '食前': 'before_meal',
        '食後': 'after_meal',
        '食直後': 'after_meal',
    }
    if raw_timing:
        key = None
        for k in timing_map.keys():
            if k in raw_timing:
                key = k
                break
        if key:
            drug['timing_norm'] = timing_map[key]
    return drug


def fix_aspirin_classification(drug: dict) -> dict:
    """アスピリン腸溶は用途分類で『抗血小板薬』に寄せる（NSAIDsを外す）"""
    name = drug.get('generic', '') or drug.get('name', '')
    strength = (drug.get('strength') or '').replace('／', '/').replace(' ', '')
    if name.startswith('アスピリン'):
        # 低用量抗血小板の代表用量 81/100mg を優先
        if '81mg' in strength or '100mg' in strength or '腸溶' in name:
            drug['final_classification'] = '抗血小板薬'
    return drug


def extract_component_strengths(drug: dict) -> dict:
    """配合剤の成分強度を抽出して component_strengths に格納する"""
    import re
    text_sources = [
        drug.get('raw', ''),
        drug.get('original_name', ''),
        drug.get('name', ''),
        drug.get('strength', ''),
    ]
    joined = ' '.join([t for t in text_sources if t])
    m = re.search(r'(\d+\.?\d*\s*(?:mg|μg))\s*/\s*(\d+\.?\d*\s*(?:mg|μg))', joined)
    if m:
        drug['component_strengths'] = [m.group(1), m.group(2)]
    return drug


