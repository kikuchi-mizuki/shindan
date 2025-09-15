"""
薬剤表示フォーマッター
剤形情報を保持した表示形式の統一
"""
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DisplayFormatter:
    """薬剤表示フォーマッター"""
    
    def __init__(self):
        # 剤形表示の正規化
        self.form_display_map = {
            "腸溶": "腸溶",
            "口腔内崩壊": "口腔内崩壊",
            "OD錠": "口腔内崩壊",
            "徐放": "徐放",
            "CR": "徐放",
            "ゲル": "ゲル",
            "軟膏": "軟膏",
            "クリーム": "クリーム",
            "貼付": "貼付",
            "テープ": "テープ",
        }
        
        # 薬剤名の表示形式ルール
        self.display_rules = {
            "アスピリン": self._format_aspirin,
            "ファモチジン": self._format_famotidine,
            "センナ": self._format_senna,
        }
    
    def format_drug_display(self, drug: Dict[str, Any]) -> str:
        """薬剤の表示形式を統一"""
        try:
            generic_name = drug.get("generic", "")
            brand_name = drug.get("brand", "")
            raw_name = drug.get("raw", "")
            
            # 表示用の薬剤名を決定
            display_name = generic_name or brand_name or raw_name
            if not display_name:
                return "不明"
            
            # 特殊な表示ルールを適用
            for drug_key, formatter in self.display_rules.items():
                if drug_key in display_name:
                    return formatter(drug, display_name)
            
            # デフォルト表示
            return self._default_format(drug, display_name)
            
        except Exception as e:
            logger.error(f"Display formatting error: {e}")
            return drug.get("generic", drug.get("brand", drug.get("raw", "不明")))
    
    def _format_aspirin(self, drug: Dict[str, Any], display_name: str) -> str:
        """アスピリンの表示形式"""
        raw_name = drug.get("raw", "")
        
        # 腸溶の情報がある場合は表示に含める
        if "腸溶" in raw_name:
            return "アスピリン腸溶"
        elif "アスピリン" in display_name:
            return "アスピリン"
        
        return display_name
    
    def _format_famotidine(self, drug: Dict[str, Any], display_name: str) -> str:
        """ファモチジンの表示形式"""
        raw_name = drug.get("raw", "")
        
        # 口腔内崩壊の情報がある場合は表示に含める
        if "口腔内崩壊" in raw_name or "OD錠" in raw_name:
            return "ファモチジン口腔内崩壊"
        elif "ファモチジン" in display_name:
            return "ファモチジン"
        
        return display_name
    
    def _format_senna(self, drug: Dict[str, Any], display_name: str) -> str:
        """センナの表示形式"""
        raw_name = drug.get("raw", "")
        
        # センナ実配合の場合は詳細表示
        if "センナ実" in raw_name or "センナ実配合" in raw_name:
            return "センナ実配合"
        elif "センナ" in display_name:
            return "センナ"
        
        return display_name
    
    def _default_format(self, drug: Dict[str, Any], display_name: str) -> str:
        """デフォルトの表示形式"""
        # 基本的には一般名をそのまま使用
        return display_name
    
    def format_drug_list(self, drugs: List[Dict[str, Any]]) -> List[str]:
        """薬剤リストの表示形式を統一"""
        formatted_drugs = []
        
        for drug in drugs:
            formatted_name = self.format_drug_display(drug)
            formatted_drugs.append(formatted_name)
        
        return formatted_drugs
    
    def get_display_summary(self, drugs: List[Dict[str, Any]]) -> str:
        """薬剤リストの表示サマリーを生成"""
        if not drugs:
            return "薬剤が検出されませんでした。"
        
        formatted_names = self.format_drug_list(drugs)
        
        # 重複除去
        unique_names = list(dict.fromkeys(formatted_names))
        
        if len(unique_names) == 1:
            return f"1剤検出: {unique_names[0]}"
        elif len(unique_names) <= 5:
            return f"{len(unique_names)}剤検出: {', '.join(unique_names)}"
        else:
            return f"{len(unique_names)}剤検出: {', '.join(unique_names[:3])} 他{len(unique_names)-3}剤"
