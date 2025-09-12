"""
薬剤重複統合ユーティリティ
同一薬の重複カウントを防ぐための後処理
"""
import re
import logging
from typing import List, Dict, Tuple, Any

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
        "ツムラ芍薬甘草湯エキス顆粒": "芍薬甘草湯"
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

def canonical_key(drug: Dict[str, Any]) -> str:
    """
    同一薬かどうかを判定するためのキー
    例: 「センノシド」「センノシド錠」→ 同じキー
        「ジクロフェナク（内服）」と「ジクロフェナクゲル（外用）」→ 別キー
    """
    name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
    route = drug.get("route") or _route_tag(drug.get("raw", "") + name)

    base = name
    for tok in FORM_TOKENS:
        base = base.replace(tok, "")
    base = _zenhan(base)
    base = normalize_generic(base)  # 商品名→一般名の辞書正規化

    return f"{base}::{route}"

def dedupe(drugs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
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
        
        def score(d: Dict[str, Any]) -> float:
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

def get_dedup_summary(original_count: int, deduped_count: int, removed_count: int) -> str:
    """重複統合のサマリーメッセージを生成"""
    if removed_count > 0:
        return f"✅ {deduped_count}剤検出しました（重複 {removed_count} 件を自動統合）"
    else:
        return f"✅ {deduped_count}剤検出しました"
