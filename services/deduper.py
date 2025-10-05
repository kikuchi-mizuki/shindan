"""
薬剤重複統合ユーティリティ
同一薬の重複カウントを防ぐための後処理
"""
import re
import logging
from typing import List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# 剤形キーワード（外用は区別したい）
EXTERNAL_TOKENS = ["ゲル", "貼付", "外用", "軟膏", "クリーム"]
FORM_TOKENS = ["錠", "OD錠", "口腔内崩壊錠", "カプセル", "顆粒", "細粒", "散", "液", "シロップ", "内用"] + EXTERNAL_TOKENS

def _zenhan(s: str) -> str:
    """全角半角統一"""
    # 全角数字・英字を半角に変換
    s = re.sub(r'[０-９]', lambda m: chr(ord(m.group(0)) - 0xFEE0), s)
    s = re.sub(r'[Ａ-Ｚａ-ｚ]', lambda m: chr(ord(m.group(0)) - 0xFEE0), s)
    s = s.replace("―", "ー").replace("ｰ", "ー")
    s = re.sub(r"[ 　\t]", "", s)
    return s

def _route_tag(text: str) -> str:
    """投与経路の判定"""
    return "外用" if any(tok in text for tok in EXTERNAL_TOKENS) else "内用"

def normalize_generic(name: str) -> str:
    """商品名→一般名の正規化（簡易版）"""
    # 基本的な同義語マッピング
    synonyms = {
        "オルケディア": "エボカルセト",
        "リンゼス": "リナクロチド", 
        "ラキソベロン": "ピコスルファートナトリウム",
        "グーフィス": "エロビキシバット",
        "芍薬甘草湯": "芍薬甘草湯",
        "ツムラ芍薬甘草湯": "芍薬甘草湯",
        "ツムラ芍薬甘草湯エキス顆粒": "芍薬甘草湯",
        "ファモチジン": "ファモチジン",
        "ファモチジンロ腔内崩壊": "ファモチジン",  # OCR誤読対応
        "ファモチジン口腔内崩壊": "ファモチジン",
        "アスピリン腸溶": "アスピリン",
        "アスピリン": "アスピリン",
        # 電解質/ミネラル製剤（取り違え対策）
        "アスパラ-CA錠200": "L-アスパラギン酸カルシウム",
        "アスパラK錠": "L-アスパラギン酸カリウム・L-アスパラギン酸マグネシウム",
        # 外用NSAIDs
        "ロキソニンテープ": "ロキソプロフェンナトリウム外用テープ",
        # 眠剤ブランド→一般名（同一成分で重複統合）
        "ベルソムラ": "スボレキサント",
        "デエビゴ": "レンボレキサント",
        "デビゴ": "レンボレキサント",
    }
    
    # 剤形を除去
    base_name = name
    for form in FORM_TOKENS:
        base_name = base_name.replace(form, "")
    
    # 同義語チェック
    for brand, generic in synonyms.items():
        if brand in base_name:
            return generic
    
    return base_name

def canonical_key(drug: dict[str, Any]) -> str:
    """
    同一薬かどうかを判定するためのキー（改善版）
    例: 「センノシド」「センノシド錠」→ 同じキー
        「ジクロフェナク（内服）」と「ジクロフェナクゲル（外用）」→ 別キー
        配合錠と構成単剤は別キー
    """
    name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
    route = drug.get("route") or _route_tag(drug.get("raw", "") + name)

    # 剤形を除去して一般名を取得
    base = _form_agnostic_name(name)
    base = _zenhan(base)
    base = normalize_generic(base)  # 商品名→一般名の辞書正規化

    return f"{base}::{route}"

def _form_agnostic_name(name: str) -> str:
    """剤形を除去した薬剤名を取得"""
    # 剤形キーワードを除去
    form_tokens = ["錠", "OD錠", "口腔内崩壊", "腸溶", "徐放", "CR", "カプセル", "cap", "包", "顆粒", "散", "液", "ゲル", "軟膏", "クリーム", "貼付", "テープ"]
    
    base = name
    for token in form_tokens:
        base = base.replace(token, "")
    
    return base.strip()

def dedupe(drugs: List[dict[str, Any]]) -> Tuple[List[dict[str, Any]], int]:
    """
    薬剤リストの重複統合
    
    Args:
        drugs: ai_extractor などが返す {raw,generic,brand,strength,confidence,...}
        
    Returns:
        Tuple[List[Dict], int]: (重複統合後リスト, 除外件数)
    """
    try:
        if not drugs:
            return [], 0
        
        best = {}
        
        def score(d: dict[str, Any]) -> float:
            """薬剤の品質スコア（一般名あり>用量情報あり>信頼度）"""
            score_val = 0.0
            if d.get("generic"):
                score_val += 1.0
            if d.get("strength"):
                score_val += 1.0
            if d.get("confidence"):
                score_val += float(d.get("confidence", 0))
            return score_val
        
        for d in drugs:
            k = canonical_key(d)
            if k not in best or score(d) > score(best[k]):
                best[k] = d
        
        merged = list(best.values())
        removed = len(drugs) - len(merged)
        
        logger.info(f"Deduplication: {len(drugs)} -> {len(merged)} drugs (removed {removed} duplicates)")
        
        return merged, removed
        
    except Exception as e:
        logger.error(f"Deduplication error: {e}")
        return drugs, 0

# 配合錠と単剤の二重取りを防ぐ
COMBO_SEP = "・"

def is_combo(name: str) -> bool:
    """配合錠かどうかを判定"""
    return COMBO_SEP in name

def collapse_combos(drugs: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """配合錠がある場合、構成単剤を除外（改善版）"""
    if not drugs:
        return drugs
    
    names = [d.get("generic") or d.get("brand") or d.get("raw", "") for d in drugs]
    combo_parts = set()
    
    # 配合錠の構成成分を抽出
    for n in names:
        if is_combo(n):
            for p in n.split(COMBO_SEP):
                # 剤形を除去して一般名のみで比較
                normalized_part = _form_agnostic_name(p.strip())
                combo_parts.add(normalized_part)
    
    # 配合錠がある場合、構成単剤を除外
    out = []
    for d in drugs:
        n = (d.get("generic") or d.get("brand") or d.get("raw", ""))
        normalized_name = _form_agnostic_name(n)
        
        # 配合錠があるブロックで、構成単剤を別件として拾っていたら除外
        if normalized_name in combo_parts and any(is_combo(x) for x in names):
            logger.info(f"Excluding single component '{n}' due to combo drug presence")
            continue
        out.append(d)
    
    logger.info(f"Combo collapse: {len(drugs)} -> {len(out)} drugs")
    return out

# 剤形違いの同一薬を統合
FORM_NOISE = ["錠", "OD錠", "口腔内崩壊", "腸溶", "徐放", "CR", "カプセル", "cap", "包", "顆粒", "散", "液", "ゲル", "軟膏", "クリーム", "貼付", "テープ"]

def strip_form(s: str) -> str:
    """剤形キーワードを除去"""
    for t in FORM_NOISE:
        s = s.replace(t, "")
    return s

def canonical_key_form_agnostic(name: str) -> str:
    """剤形を無視した正規化キー"""
    # OCR誤読の修正
    fixed_name = name.replace("ロ腔", "口腔")
    # 剤形除去
    stripped = strip_form(fixed_name)
    # 正規化
    normalized = normalize_generic(stripped)
    return normalized.replace(" ", "")

def dedupe_with_form_agnostic(drugs: List[dict[str, Any]]) -> Tuple[List[dict[str, Any]], int]:
    """剤形無視の重複統合も併用"""
    if not drugs:
        return [], 0
    
    # 通常の重複統合
    merged, removed1 = dedupe(drugs)
    
    # 剤形無視の重複統合
    form_agnostic_groups = {}
    for d in merged:
        name = d.get("generic") or d.get("brand") or d.get("raw", "")
        key = canonical_key_form_agnostic(name)
        if key not in form_agnostic_groups:
            form_agnostic_groups[key] = []
        form_agnostic_groups[key].append(d)
    
    # 各グループから最良のものを選択
    final_drugs = []
    for group in form_agnostic_groups.values():
        if len(group) > 1:
            # 複数ある場合は最良のものを選択
            best = max(group, key=lambda d: (
                bool(d.get("generic")),
                bool(d.get("strength")),
                float(d.get("confidence", 0))
            ))
            final_drugs.append(best)
            logger.info(f"Form-agnostic dedup: selected '{best.get('generic', '')}' from {len(group)} variants")
        else:
            final_drugs.append(group[0])
    
    removed2 = len(merged) - len(final_drugs)
    total_removed = removed1 + removed2
    
    logger.info(f"Enhanced deduplication: {len(drugs)} -> {len(final_drugs)} drugs (removed {total_removed} duplicates)")
    return final_drugs, total_removed

# ノイズ削除
NOISE_PATTERNS = [
    r"^[ン・\s]*配合$",
    r"^ロ腔内崩壊$",
    r"^[ン・\s]*$",
    r"^配合$"
]

def remove_noise(drugs: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """OCRノイズを除去"""
    if not drugs:
        return drugs
    
    filtered = []
    for d in drugs:
        name = d.get("generic") or d.get("brand") or d.get("raw", "")
        
        # ノイズパターンにマッチするかチェック
        is_noise = False
        for pattern in NOISE_PATTERNS:
            if re.match(pattern, name.strip()):
                is_noise = True
                logger.info(f"Removing noise: '{name}' (matched pattern: {pattern})")
                break
        
        if not is_noise:
            filtered.append(d)
    
    logger.info(f"Noise removal: {len(drugs)} -> {len(filtered)} drugs")
    return filtered

def get_dedup_summary(original_count: int, deduped_count: int, removed_count: int) -> str:
    """重複統合のサマリーメッセージを生成"""
    if removed_count > 0:
        return f"✅ {deduped_count}剤検出しました（重複 {removed_count} 件を自動統合）"
    else:
        return f"✅ {deduped_count}剤検出しました"
