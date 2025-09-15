"""
薬剤相互作用判定エンジン
タグベースの相互作用検出システム
"""
import logging
import yaml
from typing import List, Dict, Any, Tuple
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

class InteractionEngine:
    """薬剤相互作用判定エンジン"""
    
    def __init__(self, rules_file: str = "services/interaction_rules.yaml"):
        self.rules_file = rules_file
        self.rules = self._load_rules()
        logger.info(f"InteractionEngine initialized with {len(self.rules)} rules")
    
    def _load_rules(self) -> List[Dict[str, Any]]:
        """相互作用ルールを読み込み"""
        try:
            rules_path = Path(self.rules_file)
            if not rules_path.exists():
                logger.warning(f"Rules file not found: {self.rules_file}")
                return []
            
            with open(rules_path, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f)
            
            logger.info(f"Loaded {len(rules)} interaction rules")
            return rules
            
        except Exception as e:
            logger.error(f"Failed to load interaction rules: {e}")
            return []
    
    def collect_tags(self, drugs: List[Dict[str, Any]]) -> Counter:
        """薬剤リストから相互作用タグを収集"""
        from services.drug_normalization_service import DrugNormalizationService
        
        normalizer = DrugNormalizationService()
        tag_counter = Counter()
        
        for drug in drugs:
            generic_name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
            if not generic_name:
                continue
            
            # 相互作用タグを取得
            tags = normalizer.get_interaction_tags(generic_name)
            tag_counter.update(tags)
            
            logger.debug(f"Drug '{generic_name}' -> tags: {tags}")
        
        logger.info(f"Collected tags: {dict(tag_counter)}")
        return tag_counter
    
    def evaluate_rules(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """相互作用ルールを評価"""
        if not drugs or not self.rules:
            return []
        
        tag_counter = self.collect_tags(drugs)
        triggered_rules = []
        
        for rule in self.rules:
            if self._check_rule(rule, tag_counter):
                triggered_rules.append(rule)
                logger.info(f"Rule triggered: {rule['name']}")
        
        return triggered_rules
    
    def _check_rule(self, rule: Dict[str, Any], tag_counter: Counter) -> bool:
        """個別ルールの条件をチェック"""
        need_tags = rule.get("need_tags")
        if not need_tags:
            return False
        
        # タグの種類
        required_tags = need_tags.get("any_of", [])
        if not required_tags:
            return False
        
        # 最小カウント
        min_count = need_tags.get("min_count", 2)
        
        # 該当タグの合計数を計算
        total_count = sum(tag_counter.get(tag, 0) for tag in required_tags)
        
        return total_count >= min_count
    
    def format_interactions(self, triggered_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """相互作用結果をフォーマット"""
        if not triggered_rules:
            return {
                "has_interactions": False,
                "major_interactions": [],
                "moderate_interactions": [],
                "summary": "相互作用は検出されませんでした。"
            }
        
        # 重大度別に分類
        major_interactions = [r for r in triggered_rules if r.get("severity") == "major"]
        moderate_interactions = [r for r in triggered_rules if r.get("severity") == "moderate"]
        
        # サマリー生成
        summary_parts = []
        if major_interactions:
            summary_parts.append(f"重大な相互作用: {len(major_interactions)}件")
        if moderate_interactions:
            summary_parts.append(f"注意すべき相互作用: {len(moderate_interactions)}件")
        
        summary = "、".join(summary_parts) + "が検出されました。"
        
        return {
            "has_interactions": True,
            "major_interactions": major_interactions,
            "moderate_interactions": moderate_interactions,
            "summary": summary,
            "total_count": len(triggered_rules)
        }
    
    def check_drug_interactions(self, drugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """薬剤相互作用の総合チェック"""
        try:
            if not drugs:
                return {
                    "has_interactions": False,
                    "major_interactions": [],
                    "moderate_interactions": [],
                    "summary": "薬剤が検出されませんでした。"
                }
            
            # ルール評価
            triggered_rules = self.evaluate_rules(drugs)
            
            # 結果フォーマット
            result = self.format_interactions(triggered_rules)
            
            logger.info(f"Interaction check completed: {result['summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Interaction check failed: {e}")
            return {
                "has_interactions": False,
                "major_interactions": [],
                "moderate_interactions": [],
                "summary": "相互作用チェック中にエラーが発生しました。",
                "error": str(e)
            }
