"""
薬剤相互作用判定エンジン
タグベースの相互作用検出システム
"""
import logging
import yaml
from typing import List, Any, Tuple, Optional
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

class InteractionEngine:
    """薬剤相互作用判定エンジン"""
    
    def __init__(self, rules_file: str = "services/interaction_rules.yaml"):
        self.rules_file = rules_file
        self.rules = self._load_rules()
        
        # 対象薬特定サービスを初期化
        try:
            from .interaction_target_resolver import InteractionTargetResolver
            self.target_resolver = InteractionTargetResolver()
            logger.info("InteractionTargetResolver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize InteractionTargetResolver: {e}")
            self.target_resolver = None
        
        logger.info(f"InteractionEngine initialized with {len(self.rules)} rules")
    
    def _load_rules(self) -> List[dict[str, Any]]:
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
    
    def collect_tags_per_drug(self, drugs: List[dict[str, Any]]) -> dict[str, set[str]]:
        """薬剤ごとのタグを収集（薬名をキーとして）"""
        from services.drug_normalization_service import DrugNormalizationService
        
        normalizer = DrugNormalizationService()
        per_drug_tags = {}
        
        for drug in drugs:
            name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
            if not name:
                continue
            
            # 相互作用タグを取得（配合名分解対応）
            tags = normalizer.tags_for_drug(drug)
            per_drug_tags[name] = tags
            
            logger.debug(f"Drug '{name}' -> tags: {tags}")
        
        logger.info(f"Collected tags per drug: {[(k, list(v)) for k, v in per_drug_tags.items()]}")
        return per_drug_tags
    
    def collect_tags(self, drugs: List[dict[str, Any]]) -> Counter:
        """薬剤リストから相互作用タグを収集（後方互換性のため）"""
        per_drug_tags = self.collect_tags_per_drug(drugs)
        tag_counter = Counter()
        
        for tags in per_drug_tags.values():
            tag_counter.update(tags)
        
        logger.info(f"Collected tags: {dict(tag_counter)}")
        return tag_counter
    
    def evaluate_rules(self, drugs: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """相互作用ルールを評価（targets付き）"""
        if not drugs or not self.rules:
            return []
        
        per_drug_tags = self.collect_tags_per_drug(drugs)
        tag_counter = self.collect_tags(drugs)
        triggered_rules = []
        
        for rule in self.rules:
            if self._check_rule(rule, tag_counter, drugs):
                # targetsを特定
                targets = self._get_targets_for_rule(rule, drugs, per_drug_tags)
                
                # ルールにtargetsを追加
                rule_with_targets = rule.copy()
                rule_with_targets['targets'] = targets
                
                triggered_rules.append(rule_with_targets)
                logger.info(f"Rule triggered: {rule['name']} -> targets: {targets}")
        
        return triggered_rules
    
    def _check_rule(self, rule: dict[str, Any], tag_counter: Counter, drugs: List[dict[str, Any]] = None) -> bool:
        """個別ルールの条件をチェック
        サポート:
          - need_tags: { any_of: [...], min_count: N }
          - requires_tags_all: [ ... ]
          - match_any + threshold: 名前に含まれる候補の一致数で評価
        """
        # requires_tags_all（全部のタグが1回以上）
        requires_all = rule.get("requires_tags_all")
        if requires_all:
            if not all(tag_counter.get(tag, 0) > 0 for tag in requires_all):
                return False
            return True

        # need_tags（任意集合の合算 >= min_count）
        need_tags = rule.get("need_tags")
        if need_tags:
            required_tags = need_tags.get("any_of", [])
            if not required_tags:
                return False
            min_count = need_tags.get("min_count", 2)
            total_count = sum(tag_counter.get(tag, 0) for tag in required_tags)
            if total_count < min_count:
                return False
            # 条件を満たしたが、追加条件がある場合は続行

        # match_any + threshold
        match_any = rule.get("match_any")
        if match_any:
            threshold = int(rule.get("threshold", 2))
            names = []
            if drugs:
                for d in drugs:
                    names.append((d.get("generic") or d.get("brand") or d.get("raw") or ""))
            cnt = 0
            for target in match_any:
                if any(target in name for name in names):
                    cnt += 1
            return cnt >= threshold

        # どれにも該当しないがneed_tagsは満たしているケース
        return bool(need_tags)
    
    def _get_targets_for_rule(self, rule: dict[str, Any], drugs: List[dict[str, Any]], per_drug_tags: dict[str, set[str]]) -> List[str]:
        """ルールに該当する薬剤名を特定"""
        targets = []
        
        # need_tagsの場合
        need_tags = rule.get("need_tags")
        if need_tags:
            kinds = need_tags.get("any_of", [])
            for drug in drugs:
                name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
                if name in per_drug_tags and per_drug_tags[name] & set(kinds):
                    display_name = self._get_display_name(drug)
                    if display_name not in targets:
                        targets.append(display_name)
        
        # requires_tags_allの場合（必要なタグを持つ薬剤をすべて追加）
        requires_all = rule.get("requires_tags_all")
        if requires_all:
            for drug in drugs:
                name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
                if name in per_drug_tags:
                    # この薬剤が持つタグのうち、required_tags_allに含まれるものがあれば追加
                    drug_tags = per_drug_tags[name]
                    if any(tag in drug_tags for tag in requires_all):
                        display_name = self._get_display_name(drug)
                        if display_name not in targets:
                            targets.append(display_name)
        
        # match_anyの場合
        match_any = rule.get("match_any")
        if match_any:
            for drug in drugs:
                name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
                if any(target in name for target in match_any):
                    display_name = self._get_display_name(drug)
                    if display_name not in targets:
                        targets.append(display_name)
        
        return targets
    
    def _get_display_name(self, drug: dict[str, Any]) -> str:
        """表示用の薬剤名を取得"""
        return drug.get("display") or drug.get("brand") or drug.get("generic") or drug.get("raw", "")
    
    def format_interactions(self, triggered_rules: List[dict[str, Any]], drugs: List[dict[str, Any]] = None) -> dict[str, Any]:
        """相互作用結果をフォーマット（対象薬の特定付き）"""
        if not triggered_rules:
            return {
                "has_interactions": False,
                "major_interactions": [],
                "moderate_interactions": [],
                "summary": "相互作用は検出されませんでした。"
            }
        
        # 対象薬の特定を追加（最小パッチ適用）
        if self.target_resolver:
            resolved_targets = self.target_resolver.resolve_targets(drugs)
            
            for rule in triggered_rules:
                rule_id = rule.get('id', '')
                
                # ルールIDに基づいて対象薬を設定
                if rule_id == 'raas_contraindicated' and 'raas_contraindicated' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['raas_contraindicated']
                elif rule_id == 'raas_double_block_avoid' and 'raas_overlap' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['raas_overlap']
                elif rule_id == 'pde5_nitrate_contraindicated' and 'pde5_nitrate' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['pde5_nitrate']
                elif rule_id == 'gastric_acid_suppression_duplicate' and 'acid_dup' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['acid_dup']
                elif rule_id == 'antihypertensive_multi_drug' and 'poly_antihypertensive' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['poly_antihypertensive']
                else:
                    # フォールバック: 既存のロジック
                    if 'targets' in rule:
                        rule['target_drugs'] = ", ".join(rule['targets']) if rule['targets'] else "対象薬の特定に失敗"
                    else:
                        rule['target_drugs'] = self._identify_target_drugs(rule)
        else:
            # フォールバック: 既存のロジック
            for rule in triggered_rules:
                if 'targets' in rule:
                    rule['target_drugs'] = ", ".join(rule['targets']) if rule['targets'] else "対象薬の特定に失敗"
                elif 'matched_drugs' not in rule:
                    rule['target_drugs'] = self._identify_target_drugs(rule)
        
        # 重大度別に分類
        major_interactions = [r for r in triggered_rules if r.get("severity") == "major"]
        moderate_interactions = [r for r in triggered_rules if r.get("severity") in ["moderate", "minor"]]
        
        # サマリー生成
        summary_parts = []
        if major_interactions:
            summary_parts.append(f"重大な相互作用: {len(major_interactions)}件")
        if moderate_interactions:
            summary_parts.append(f"注意すべき相互作用: {len(moderate_interactions)}件")
        
        summary = "、".join(summary_parts) + "が検出されました。"
        
        # 注意のみでも相互作用ありとして扱う
        has_interactions = len(major_interactions) > 0 or len(moderate_interactions) > 0
        
        return {
            "has_interactions": has_interactions,
            "major_interactions": major_interactions,
            "moderate_interactions": moderate_interactions,
            "summary": summary,
            "total_count": len(triggered_rules)
        }
    
    def _identify_target_drugs(self, rule: dict[str, Any]) -> str:
        """ルールに該当した薬剤名を特定"""
        try:
            matched_drugs = rule.get('matched_drugs', [])
            if not matched_drugs:
                return "対象薬の特定に失敗"
            
            # 薬剤名を結合
            target_names = []
            for drug in matched_drugs:
                display_name = self._get_display_name(drug)
                if display_name:
                    target_names.append(display_name)
            
            return ", ".join(target_names) if target_names else "対象薬の特定に失敗"
            
        except Exception as e:
            logger.error(f"Target drug identification failed: {e}")
            return "対象薬の特定に失敗"
    
    def check_drug_interactions(self, drugs: List[dict[str, Any]]) -> dict[str, Any]:
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
            result = self.format_interactions(triggered_rules, drugs)
            
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
