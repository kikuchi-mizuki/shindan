"""
薬剤分類サービス
辞書 → KEGG/ATC → AIヒントの順でフォールバック
"""
import logging
from typing import Dict, List, Optional, Any
from .drug_classes import CLASS_BY_GENERIC

logger = logging.getLogger(__name__)

def normalize_generic(generic_name: str) -> str:
    """一般名を正規化"""
    if not generic_name:
        return ""
    
    # 基本的な正規化
    normalized = generic_name.strip()
    
    # よくある表記揺れを修正
    replacements = {
        "酒石酸塩": "",
        "塩酸塩": "",
        "ナトリウム": "",
        "シュウ酸塩": "",
        "エキス顆粒": "",
        "錠": "",
        "カプセル": "",
        "ゲル": "",
        "液": "",
        "顆粒": "",
        "散": "",
        "軟膏": "",
        "クリーム": "",
        "貼付剤": "",
        "点眼液": "",
        "点鼻液": "",
        "吸入液": "",
        "注射剤": "",
        "注射液": "",
        "（医療用）": "",
        "（外用）": "",
        "（内服）": "",
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized.strip()

def add_dosage_form_tag(generic: str, raw_text: str) -> str:
    """剤形で分類が変わるケースに対応"""
    if not raw_text:
        return generic
    
    # 外用など剤形で分類が変わるケース
    if any(keyword in raw_text for keyword in ["ゲル", "外用", "貼付", "軟膏", "クリーム"]):
        if "ジクロフェナク" in generic:
            return "ジクロフェナクナトリウム（外用）"
        elif "インドメタシン" in generic:
            return "インドメタシン（外用）"
        elif "ケトプロフェン" in generic:
            return "ケトプロフェン（外用）"
    
    return generic

class Classifier:
    """薬剤分類サービス"""
    
    def __init__(self, kegg_service=None):
        self.kegg_service = kegg_service  # Noneでも動く
        logger.info("Classifier initialized")
    
    def classify_one(self, drug: Dict[str, Any]) -> Optional[str]:
        """
        単一薬剤の分類を取得
        
        Args:
            drug: ai_extractorが返した1件 {"raw","generic","class_hint",...}
            
        Returns:
            日本語の分類ラベル or None
        """
        try:
            # 一般名を取得・正規化
            generic = drug.get("generic", "")
            if not generic:
                generic = normalize_generic(drug.get("raw", ""))
            
            # 剤形タグを追加
            generic_with_form = add_dosage_form_tag(generic, drug.get("raw", ""))
            
            # 1) ローカル辞書で検索
            classification = CLASS_BY_GENERIC.get(generic_with_form)
            if classification:
                logger.info(f"Found classification in local dict: {generic_with_form} -> {classification}")
                return classification
            
            # 元の一般名でも検索
            classification = CLASS_BY_GENERIC.get(generic)
            if classification:
                logger.info(f"Found classification in local dict (original): {generic} -> {classification}")
                return classification
            
            # 2) KEGG/ATC由来（実装済みの場合）
            if self.kegg_service:
                try:
                    kegg_info = self.kegg_service.safe_find_kegg_info(generic)
                    if kegg_info and kegg_info.get('category'):
                        classification = kegg_info['category']
                        logger.info(f"Found classification in KEGG: {generic} -> {classification}")
                        return classification
                except Exception as e:
                    logger.warning(f"KEGG classification failed: {e}")
            
            # 3) AIのclass_hint
            hint = drug.get("class_hint")
            if hint:
                classification = f"{hint}（AI推定）"
                logger.info(f"Using AI hint: {generic} -> {classification}")
                return classification
            
            logger.warning(f"No classification found for: {generic}")
            return None
            
        except Exception as e:
            logger.error(f"Classification error for drug {drug}: {e}")
            return None
    
    def classify_many(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        複数薬剤の分類を一括処理
        
        Args:
            drugs: ai_extractorが返した薬剤リスト
            
        Returns:
            分類情報を追加した薬剤リスト
        """
        try:
            result = []
            for drug in drugs:
                classification = self.classify_one(drug)
                drug["class_jp"] = classification or "分類未設定"
                result.append(drug)
            
            logger.info(f"Classified {len(result)} drugs")
            return result
            
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            # エラーの場合は元のリストをそのまま返す
            for drug in drugs:
                drug["class_jp"] = "分類エラー"
            return drugs
    
    def get_classification_stats(self, drugs: List[Dict[str, Any]]) -> Dict[str, int]:
        """分類統計を取得"""
        stats = {
            "total": len(drugs),
            "classified": 0,
            "ai_estimated": 0,
            "unclassified": 0
        }
        
        for drug in drugs:
            class_jp = drug.get("class_jp", "")
            if class_jp == "分類未設定":
                stats["unclassified"] += 1
            elif "（AI推定）" in class_jp:
                stats["ai_estimated"] += 1
            else:
                stats["classified"] += 1
        
        return stats
