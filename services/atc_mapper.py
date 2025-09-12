"""
ATCコードを日本語分類ラベルにマッピング
"""
import logging

logger = logging.getLogger(__name__)

# ATCコードの大分類（1文字目）
ATC_TOP_JP = {
    "A": "消化器系・代謝",
    "B": "血液・造血器",
    "C": "循環器系",
    "D": "皮膚科用",
    "G": "泌尿・生殖器・性ホルモン",
    "H": "全身用ホルモン（性ホルモン除く）",
    "J": "全身用抗感染薬",
    "L": "抗腫瘍・免疫調節",
    "M": "筋骨格系",
    "N": "神経系",
    "P": "寄生虫",
    "R": "呼吸器系",
    "S": "感覚器",
    "V": "その他",
}

# ATCコードの詳細分類（部分一致）
ATC_PARTIAL_JP = {
    # 消化器系・代謝 (A)
    "A06": "下剤",
    "A06AB": "下剤（刺激性）",
    "A06AB08": "下剤（刺激性：ピコスルファート）",
    "A06AX04": "便秘症治療薬（GC-C作動薬）",
    "A06AX16": "便秘症治療薬（胆汁酸輸送阻害薬）",
    "A02AC": "制酸薬（カルシウム）",
    "A02AB": "制酸薬（アルミニウム）",
    "A02AD": "制酸薬（マグネシウム）",
    
    # 筋骨格系 (M)
    "M04AA": "高尿酸血症治療薬（キサンチン酸化酵素阻害薬）",
    "M02AA": "NSAIDs外用",
    "M01AB": "NSAIDs（ジクロフェナク系）",
    "M01AE": "NSAIDs（プロピオン酸系）",
    
    # 呼吸器系 (R)
    "R03DC": "抗アレルギー薬（LTRA）",
    "R06AX": "抗ヒスタミン薬",
    
    # 全身用ホルモン (H)
    "H05BX": "副甲状腺機能亢進症治療薬（Ca受容体作動薬）",
    "H05AA": "副甲状腺ホルモン",
    
    # 神経系 (N)
    "N05CF": "睡眠薬（非ベンゾジアゼピン系）",
    "N05CM": "睡眠薬（オレキシン受容体拮抗薬）",
    "N05CH": "睡眠薬（メラトニン受容体作動薬）",
    "N06AB": "抗うつ薬（SSRI）",
    "N06AX": "抗うつ薬（その他）",
    
    # 皮膚科用 (D)
    "D07": "副腎皮質ホルモン外用",
    "D11": "皮膚科用その他",
    
    # 循環器系 (C)
    "C09AA": "ACE阻害薬",
    "C09CA": "ARB",
    "C07AB": "β遮断薬",
    "C08CA": "カルシウム拮抗薬",
    
    # 血液・造血器 (B)
    "B01AC": "抗血小板薬",
    "B01AA": "抗凝固薬",
}

def atc_to_jp(atc_codes):
    """
    ATCコードリストを日本語分類ラベルに変換
    
    Args:
        atc_codes: ATCコードのリスト
        
    Returns:
        str: 日本語分類ラベル or None
    """
    if not atc_codes:
        return None
    
    # 最も詳細なコードから順に短縮してマッチング
    leaf = sorted(atc_codes, key=len)[-1]  # 例: A06AB06
    
    # 詳細から大分類まで順番にチェック
    for length in [len(leaf), 5, 3, 1]:
        if length <= len(leaf):
            code = leaf[:length]
            if code in ATC_PARTIAL_JP:
                logger.info(f"ATC mapping: {code} -> {ATC_PARTIAL_JP[code]}")
                return ATC_PARTIAL_JP[code]
    
    # 大分類のみ
    first_char = leaf[0] if leaf else ""
    if first_char in ATC_TOP_JP:
        logger.info(f"ATC top-level mapping: {first_char} -> {ATC_TOP_JP[first_char]}")
        return ATC_TOP_JP[first_char]
    
    logger.warning(f"No ATC mapping found for: {atc_codes}")
    return None

def get_atc_classification_info(atc_codes):
    """
    ATCコードの詳細情報を取得
    
    Args:
        atc_codes: ATCコードのリスト
        
    Returns:
        dict: 分類情報
    """
    if not atc_codes:
        return {"classification": None, "atc_codes": []}
    
    classification = atc_to_jp(atc_codes)
    
    return {
        "classification": classification,
        "atc_codes": atc_codes,
        "primary_atc": sorted(atc_codes, key=len)[-1] if atc_codes else None
    }
