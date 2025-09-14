"""
KEGG/ATCベースの薬剤分類器
重複統合後の薬剤リストに対して分類を実行
"""
import logging
from typing import List, Dict, Any, Tuple, Optional

from .kegg_client import KEGGClient
from .atc_mapper import atc_to_jp
from .drug_classes import CLASS_BY_GENERIC

logger = logging.getLogger(__name__)

# ローカルのフォールバック（KEGGでATCが引けない時）
FALLBACK_CLASS = {
    "ナルフラフィン塩酸塩": "かゆみ治療薬（κオピオイド受容体作動薬）",
    "芍薬甘草湯": "漢方エキス（筋痙攣・疼痛）",
    "ツムラ芍薬甘草湯": "漢方エキス（筋痙攣・疼痛）",
    "ツムラ芍薬甘草湯エキス顆粒": "漢方エキス（筋痙攣・疼痛）",
    # 新たに追加された薬剤のフォールバック
    "エンレスト": "心不全治療薬（ARNI）",
    "アスピリン": "NSAIDs（抗血小板薬）",
    "アスピリン腸溶": "NSAIDs（抗血小板薬）",
    "キックリン": "鎮痛薬（オピオイド＋アセトアミノフェン配合）",
    "トラマドール・アセトアミノフェン": "鎮痛薬（オピオイド＋アセトアミノフェン配合）",
    "テルミサルタン": "ARB（アンジオテンシン受容体拮抗薬）",
    "アムロジピン": "カルシウム拮抗薬",
    "テルミサルタン・アムロジピン": "ARB＋カルシウム拮抗薬配合",
    "ファモチジン": "H2ブロッカー",
    "ファモチジンロ腔内崩壊": "H2ブロッカー",
    "センナ": "下剤（刺激性）",
    "センナ・センナ実": "下剤（刺激性）",
    "センナ・センナ実配合": "下剤（刺激性）",
    "ン配合": "配合薬（詳細不明）",
}

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
        # 新たに追加された同義語
        "エンレスト": "エンレスト",
        "キックリン": "キックリン",
        "トラマドール・アセトアミノフェン": "トラマドール・アセトアミノフェン",
        "テルミサルタン・アムロジピン": "テルミサルタン・アムロジピン",
        "ファモチジンロ腔内崩壊": "ファモチジン",
        "センナ・センナ実配合": "センナ・センナ実",
    }
    
    # 同義語チェック
    for brand, generic in synonyms.items():
        if brand in name:
            return generic
    
    return name

class KeggClassifier:
    """KEGG/ATCベースの薬剤分類器"""
    
    def __init__(self):
        self.kegg = KEGGClient()
        logger.info("KeggClassifier initialized")

    def classify_one(self, drug: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        単一薬剤の分類を実行
        
        Args:
            drug: 薬剤情報辞書
            
        Returns:
            Tuple[分類ラベル, KEGG情報]
        """
        try:
            # 一般名を取得
            g = drug.get("generic") or normalize_generic(drug.get("raw", ""))
            if not g:
                logger.warning(f"No generic name found for drug: {drug}")
                return None, None
            
            logger.info(f"Classifying drug: {g}")
            
            # 1) ローカル辞書をチェック
            if g in CLASS_BY_GENERIC:
                classification = CLASS_BY_GENERIC[g]
                logger.info(f"Local dictionary match: {g} -> {classification}")
                return classification, None
            
            # 2) KEGG/ATCで分類（外用ジクロフェナクは強制的に外用ATCへ寄せる）
            info = self.kegg.best_kegg_and_atc(g)
            # 外用判定トークン
            raw_plus_brand = f"{drug.get('raw','')}{drug.get('brand','')}"
            if "ジクロフェナク" in g and any(k in raw_plus_brand for k in ["ゲル","外用","塗布","貼付","軟膏","クリーム"]):
                atc_codes = ["M02AA15"]
                classification = atc_to_jp(atc_codes)
                return classification, {"kegg_id": (info or {}).get("kegg_id"), "atc": atc_codes}
            if info:
                atc_codes = info.get("atc", [])
                if atc_codes:
                    classification = atc_to_jp(atc_codes)
                    if classification:
                        logger.info(f"KEGG/ATC classification: {g} -> {classification}")
                        return classification, info
            
            # 3) フォールバック辞書
            if g in FALLBACK_CLASS:
                classification = FALLBACK_CLASS[g]
                logger.info(f"Fallback classification: {g} -> {classification}")
                return classification, None
            
            # 4) AIヒント（信頼できる場合のみ）
            ai_hint = drug.get("class_hint")
            if ai_hint and self._is_reliable_ai_hint(ai_hint):
                classification = f"{ai_hint}（AI推定）"
                logger.info(f"AI hint classification: {g} -> {classification}")
                return classification, None
            
            logger.warning(f"No classification found for: {g}")
            return None, None
            
        except Exception as e:
            logger.error(f"Classification error for drug {drug}: {e}")
            return None, None

    def _is_reliable_ai_hint(self, hint: str) -> bool:
        """AIヒントの信頼性をチェック"""
        if not hint:
            return False
        
        # 信頼できるキーワード
        reliable_keywords = [
            "NSAIDs", "下剤", "便秘薬", "抗アレルギー", "睡眠薬", 
            "抗うつ薬", "制酸薬", "高尿酸血症", "副甲状腺", "漢方"
        ]
        
        return any(keyword in hint for keyword in reliable_keywords)

    def classify_many(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        複数薬剤の分類を実行
        
        Args:
            drugs: 薬剤情報辞書のリスト
            
        Returns:
            分類情報が追加された薬剤リスト
        """
        logger.info(f"Classifying {len(drugs)} drugs")
        
        classified_drugs = []
        for i, drug in enumerate(drugs):
            try:
                classification, kegg_info = self.classify_one(drug)
                
                # 薬剤情報を更新
                updated_drug = drug.copy()
                updated_drug["final_classification"] = classification or "分類未設定"
                
                # KEGG情報を追加
                if kegg_info:
                    updated_drug["kegg_id"] = kegg_info.get("kegg_id")
                    updated_drug["kegg_category"] = kegg_info.get("label")
                    updated_drug["atc_codes"] = kegg_info.get("atc", [])
                
                classified_drugs.append(updated_drug)
                
                logger.info(f"Drug {i+1}/{len(drugs)}: {drug.get('generic', 'Unknown')} -> {updated_drug['final_classification']}")
                
            except Exception as e:
                logger.error(f"Error classifying drug {i+1}: {e}")
                # エラー時は元の薬剤情報を保持
                updated_drug = drug.copy()
                updated_drug["final_classification"] = "分類未設定"
                classified_drugs.append(updated_drug)
        
        # 分類結果の統計
        classified_count = sum(1 for d in classified_drugs if d["final_classification"] != "分類未設定")
        logger.info(f"Classification complete: {classified_count}/{len(drugs)} drugs classified")
        
        return classified_drugs

    def get_classification_stats(self, drugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分類結果の統計情報を取得"""
        total = len(drugs)
        classified = sum(1 for d in drugs if d.get("final_classification") != "分類未設定")
        local_dict = sum(1 for d in drugs if d.get("generic") in CLASS_BY_GENERIC)
        kegg_atc = sum(1 for d in drugs if d.get("atc_codes"))
        fallback = sum(1 for d in drugs if d.get("generic") in FALLBACK_CLASS)
        
        return {
            "total_drugs": total,
            "classified": classified,
            "classification_rate": classified / total if total > 0 else 0,
            "local_dict_matches": local_dict,
            "kegg_atc_matches": kegg_atc,
            "fallback_matches": fallback
        }
