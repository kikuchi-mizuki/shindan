"""
ルールエンジン - YAMLルールベースの相互作用判定
誤検知ほぼゼロ／取りこぼしほぼゼロで、根拠が示せる診断
"""
import yaml
import logging
from typing import List, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RuleEngine:
    """ルールベースの相互作用判定エンジン"""
    
    def __init__(self, rules_file: str = "services/interaction_rules.yaml"):
        """初期化"""
        self.rules_file = rules_file
        self.rules = []
        self._load_rules()
    
    def _load_rules(self):
        """YAMLルールファイルを読み込み"""
        try:
            rules_path = Path(self.rules_file)
            if not rules_path.exists():
                logger.warning(f"Rules file not found: {self.rules_file}")
                return
            
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules = yaml.safe_load(f) or []
            
            logger.info(f"Loaded {len(self.rules)} interaction rules")
            
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self.rules = []
    
    def judge(self, drugs: List[str]) -> List[dict[str, Any]]:
        """
        薬剤リストに対してルールベースの相互作用判定を実行
        
        Args:
            drugs: 薬剤名のリスト（一般名）
            
        Returns:
            マッチしたルールのリスト
        """
        if not drugs or not self.rules:
            return []
        
        # 薬剤名をセットに変換（高速検索のため）
        drug_set = set(drugs)
        hits = []
        
        for rule in self.rules:
            try:
                if self._evaluate_rule(rule, drug_set):
                    # ルールにマッチした場合、詳細情報を追加
                    rule_result = rule.copy()
                    rule_result['matched_drugs'] = self._get_matched_drugs(rule, drug_set)
                    rule_result['rule_id'] = rule.get('id', 'unknown')
                    hits.append(rule_result)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Rule engine found {len(hits)} interactions for {len(drugs)} drugs")
        return hits
    
    def _evaluate_rule(self, rule: dict[str, Any], drug_set: Set[str]) -> bool:
        """個別ルールの評価"""
        rule_id = rule.get('id', 'unknown')
        
        # requires: すべての薬剤が必要
        if "requires" in rule:
            required_drugs = rule["requires"]
            if all(drug in drug_set for drug in required_drugs):
                logger.debug(f"Rule {rule_id}: requires match")
                return True
        
        # requires_any + with_any: いずれかが必要 + いずれかが必要
        elif "requires_any" in rule and "with_any" in rule:
            requires_any = rule["requires_any"]
            with_any = rule["with_any"]
            if (any(drug in drug_set for drug in requires_any) and 
                any(drug in drug_set for drug in with_any)):
                logger.debug(f"Rule {rule_id}: requires_any + with_any match")
                return True
        
        # match_any: 閾値以上の薬剤がマッチ
        elif "match_any" in rule:
            match_drugs = rule["match_any"]
            threshold = int(rule.get("threshold", 2))
            matched_count = len(drug_set & set(match_drugs))
            if matched_count >= threshold:
                logger.debug(f"Rule {rule_id}: match_any threshold met ({matched_count}/{threshold})")
                return True
        
        return False
    
    def _get_matched_drugs(self, rule: dict[str, Any], drug_set: Set[str]) -> List[str]:
        """マッチした薬剤名を取得"""
        matched = []
        
        if "requires" in rule:
            matched.extend([drug for drug in rule["requires"] if drug in drug_set])
        
        if "requires_any" in rule:
            matched.extend([drug for drug in rule["requires_any"] if drug in drug_set])
        
        if "with_any" in rule:
            matched.extend([drug for drug in rule["with_any"] if drug in drug_set])
        
        if "match_any" in rule:
            matched.extend([drug for drug in rule["match_any"] if drug in drug_set])
        
        return list(set(matched))  # 重複除去
    
    def get_rule_by_id(self, rule_id: str) -> Optional[dict[str, Any]]:
        """ルールIDでルールを取得"""
        for rule in self.rules:
            if rule.get('id') == rule_id:
                return rule
        return None
    
    def get_rules_by_severity(self, severity: str) -> List[dict[str, Any]]:
        """重要度でルールをフィルタ"""
        return [rule for rule in self.rules if rule.get('severity') == severity]
    
    def validate_rule(self, rule: dict[str, Any]) -> bool:
        """ルールの妥当性を検証"""
        required_fields = ['id', 'name', 'severity', 'advice']
        
        # 必須フィールドのチェック
        for field in required_fields:
            if field not in rule:
                logger.error(f"Rule missing required field: {field}")
                return False
        
        # ルール条件のチェック
        has_condition = any(field in rule for field in ['requires', 'requires_any', 'match_any'])
        if not has_condition:
            logger.error(f"Rule {rule.get('id')} has no condition")
            return False
        
        return True
    
    def add_rule(self, rule: dict[str, Any]) -> bool:
        """新しいルールを追加"""
        if not self.validate_rule(rule):
            return False
        
        # 重複チェック
        if any(existing_rule.get('id') == rule.get('id') for existing_rule in self.rules):
            logger.error(f"Rule with id {rule.get('id')} already exists")
            return False
        
        self.rules.append(rule)
        logger.info(f"Added new rule: {rule.get('id')}")
        return True
    
    def save_rules(self):
        """ルールをYAMLファイルに保存"""
        try:
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.rules, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved {len(self.rules)} rules to {self.rules_file}")
        except Exception as e:
            logger.error(f"Failed to save rules: {e}")
    
    def get_statistics(self) -> dict[str, Any]:
        """ルール統計情報を取得"""
        severity_counts = {}
        for rule in self.rules:
            severity = rule.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_rules': len(self.rules),
            'severity_distribution': severity_counts,
            'rules_file': self.rules_file
        }
