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
    """薬剤相互作用判定エンジン（3層ハイブリッド診断）"""
    
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
        
        # KEGGクライアントを初期化
        try:
            from .kegg_client import KEGGClient
            self.kegg_client = KEGGClient()
            logger.info("KEGGClient initialized for DDI")
        except Exception as e:
            logger.error(f"Failed to initialize KEGGClient: {e}")
            self.kegg_client = None
        
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
        """表示用の薬剤名を取得（一般名で統一）"""
        # 表示用の正規化（一般名で統一）
        GENERIC_DISPLAY = {
            "エンレスト": "サクビトリル/バルサルタン",
            "テラムロAP": "テルミサルタン/アムロジピン",
            "タケキャブ": "ボノプラザン",
            "ランソプラゾールOD": "ランソプラゾール",
        }
        
        # 一般名を優先
        name = drug.get("generic") or drug.get("brand") or drug.get("raw", "")
        
        # 用量情報を除去（5mg、100mg等）
        import re
        name = re.sub(r'\s*\d+(\.\d+)?\s*mg\b', '', name, flags=re.IGNORECASE)
        # メーカー情報を除去（「トーワ」、「サンド」等）
        name = re.sub(r'「.*?」', '', name)
        # 余分な空白を整理
        name = re.sub(r'\s+', '', name).strip()
        
        # 一般名に変換
        return GENERIC_DISPLAY.get(name, name)
    
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
                    # 仕上げ修正：モニタリングの定型文を追加
                    if rule.get('advice'):
                        rule['advice'] += "開始/変更後1–2週でCr/eGFR/K、血圧を再評価。"
                elif rule_id == 'raas_double_block_avoid' and 'raas_overlap' in resolved_targets:
                    # RAAS重複の場合は特定の順序で表示
                    raas_overlap_targets = resolved_targets['raas_overlap']
                    # 3剤すべてが含まれている場合は推奨フォーマット順で表示
                    if "エナラプリル" in raas_overlap_targets and "サクビトリル/バルサルタン" in raas_overlap_targets and "テルミサルタン/アムロジピン" in raas_overlap_targets:
                        rule['target_drugs'] = "エナラプリル、サクビトリル/バルサルタン、テルミサルタン/アムロジピン"
                    else:
                        rule['target_drugs'] = raas_overlap_targets
                    # 仕上げ修正：理由の一行補足とモニタリングの定型文を追加
                    if rule.get('advice'):
                        advice = rule['advice']
                        # ARNIが含まれていればARB含有の旨を補足
                        if "サクビトリル/バルサルタン" in raas_overlap_targets:
                            advice += "（注：ARNIはARB成分〈バルサルタン〉を含むため）"
                        advice += "開始/変更後1–2週でCr/eGFR/K、血圧を再評価。"
                        rule['advice'] = advice
                elif rule_id == 'pde5_nitrate_contraindicated' and 'pde5_nitrate' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['pde5_nitrate']
                elif rule_id == 'gastric_acid_suppression_duplicate' and 'acid_dup' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['acid_dup']
                elif rule_id == 'antihypertensive_multi_drug' and 'poly_antihypertensive' in resolved_targets:
                    rule['target_drugs'] = resolved_targets['poly_antihypertensive']
                else:
                    # フォールバック: 既存のロジック
                    if 'targets' in rule:
                        rule['target_drugs'] = "、".join(rule['targets']) if rule['targets'] else "対象薬の特定に失敗"
                    else:
                        rule['target_drugs'] = self._identify_target_drugs(rule)
        else:
            # フォールバック: 既存のロジック
            for rule in triggered_rules:
                if 'targets' in rule:
                    rule['target_drugs'] = "、".join(rule['targets']) if rule['targets'] else "対象薬の特定に失敗"
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
            
            return "、".join(target_names) if target_names else "対象薬の特定に失敗"
            
        except Exception as e:
            logger.error(f"Target drug identification failed: {e}")
            return "対象薬の特定に失敗"
    
    def check_drug_interactions(self, drugs: List[dict[str, Any]]) -> dict[str, Any]:
        """薬剤相互作用の総合チェック（3層ハイブリッド診断）"""
        try:
            if not drugs:
                return {
                    "has_interactions": False,
                    "major_interactions": [],
                    "moderate_interactions": [],
                    "summary": "薬剤が検出されませんでした。"
                }
            
            major_interactions = []
            moderate_interactions = []
            
            # === 第1層：手動ルール（高速・高精度）===
            logger.info("Layer 1: Manual rules check")
            if self.target_resolver:
                findings = self.target_resolver.build_report(drugs)
                
                for finding in findings:
                    severity = finding.get("severity", "")
                    logger.info(f"Manual rule: {finding.get('title', '')} (severity: {severity})")
                    
                    interaction = {
                        "name": finding.get("title", ""),
                        "target_drugs": finding.get("targets", ""),
                        "advice": finding.get("action", ""),
                        "severity": severity,
                        "source": "手動ルール"
                    }
                    
                    if severity == "重大":
                        major_interactions.append(interaction)
                    else:
                        moderate_interactions.append(interaction)
            
            # === 第2層：KEGG DDI（公式データベース）===
            logger.info("Layer 2: KEGG DDI check")
            if self.kegg_client:
                kegg_interactions = self._check_kegg_ddi(drugs)
                
                for interaction in kegg_interactions:
                    # 重複チェック（手動ルールで既に検出されているものは除外）
                    is_duplicate = False
                    for existing in major_interactions + moderate_interactions:
                        if self._is_similar_interaction(existing, interaction):
                            is_duplicate = True
                            logger.info(f"KEGG DDI duplicate skipped: {interaction.get('name', '')}")
                            break
                    
                    if not is_duplicate:
                        logger.info(f"KEGG DDI found: {interaction.get('name', '')} (severity: {interaction.get('severity', '')})")
                        if interaction.get("severity") == "重大":
                            major_interactions.append(interaction)
                        else:
                            moderate_interactions.append(interaction)
            
            # === 第3層：AI補完（未知パターン検出）===
            # 注：第1層・第2層で検出が少ない場合のみ実行（コスト管理）
            total_found = len(major_interactions) + len(moderate_interactions)
            if total_found == 0 and len(drugs) >= 3:
                logger.info("Layer 3: AI補完 check (no interactions found in Layer 1-2)")
                ai_interactions = self._check_ai_interactions(drugs)
                
                for interaction in ai_interactions:
                    logger.info(f"AI found: {interaction.get('name', '')} (severity: {interaction.get('severity', '')})")
                    if interaction.get("severity") == "重大":
                        major_interactions.append(interaction)
                    else:
                        moderate_interactions.append(interaction)
            else:
                logger.info(f"Layer 3: AI補完 skipped ({total_found} interactions found in Layer 1-2)")
            
            has_interactions = len(major_interactions) > 0 or len(moderate_interactions) > 0
            summary = f"重大な相互作用: {len(major_interactions)}件、注意すべき相互作用: {len(moderate_interactions)}件" if has_interactions else "相互作用は検出されませんでした。"
            
            return {
                "has_interactions": has_interactions,
                "major_interactions": major_interactions,
                "moderate_interactions": moderate_interactions,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Interaction check failed: {e}")
            return {
                "has_interactions": False,
                "major_interactions": [],
                "moderate_interactions": [],
                "summary": "相互作用チェック中にエラーが発生しました。",
                "error": str(e)
            }
    
    def _check_kegg_ddi(self, drugs: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """KEGG DDIで薬剤間相互作用をチェック"""
        interactions = []
        
        try:
            # 各薬剤のKEGG IDを取得
            drug_kegg_ids = []
            for drug in drugs:
                generic = drug.get('generic', drug.get('name', drug.get('raw', '')))
                
                # KEGG検索
                kegg_info = self.kegg_client.best_kegg_and_atc(generic)
                if kegg_info and kegg_info.get('kegg_id'):
                    drug_kegg_ids.append({
                        'name': generic,
                        'kegg_id': kegg_info['kegg_id']
                    })
            
            logger.info(f"Found KEGG IDs for {len(drug_kegg_ids)} drugs")
            
            # 薬剤ペアごとにDDIをチェック
            for i in range(len(drug_kegg_ids)):
                for j in range(i + 1, len(drug_kegg_ids)):
                    drug1 = drug_kegg_ids[i]
                    drug2 = drug_kegg_ids[j]
                    
                    ddi_results = self.kegg_client.get_drug_interactions(
                        drug1['kegg_id'], 
                        drug2['kegg_id']
                    )
                    
                    for ddi in ddi_results:
                        interactions.append({
                            'name': f"{drug1['name']} × {drug2['name']}",
                            'target_drugs': f"{drug1['name']}、{drug2['name']}",
                            'advice': ddi.get('description', '相互作用が報告されています。'),
                            'severity': ddi.get('severity', '併用注意'),
                            'source': 'KEGG'
                        })
            
            logger.info(f"KEGG DDI check completed: {len(interactions)} interactions found")
            return interactions
            
        except Exception as e:
            logger.error(f"KEGG DDI check failed: {e}")
            return []
    
    def _check_ai_interactions(self, drugs: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """AI（LLM）で未知の相互作用をチェック"""
        interactions = []
        
        try:
            import os
            from openai import OpenAI
            
            # OpenAI APIキーの確認
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found, skipping AI check")
                return []
            
            client = OpenAI(api_key=api_key)
            
            # 薬剤名リストを作成
            drug_names = [d.get('generic', d.get('name', d.get('raw', ''))) for d in drugs]
            drug_list_str = "、".join(drug_names)
            
            prompt = f"""以下の薬剤の組み合わせについて、重要な薬物相互作用を分析してください。

薬剤リスト：{drug_list_str}

以下の形式でJSON配列として回答してください（相互作用がない場合は空配列）：
[
  {{
    "drug1": "薬剤名1",
    "drug2": "薬剤名2",
    "severity": "重大" または "併用注意",
    "description": "相互作用の内容と対応方法"
  }}
]

注意点：
- 臨床的に重要な相互作用のみを報告してください
- 禁忌・重大な相互作用は「重大」、その他は「併用注意」としてください
- 対応方法も具体的に記載してください"""
            
            logger.info("Calling OpenAI API for AI interaction check")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "あなたは薬剤師です。薬物相互作用の分析を行ってください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # レスポンスをパース
            import json
            import re
            
            content = response.choices[0].message.content.strip()
            
            # JSON部分を抽出（コードブロックで囲まれている場合に対応）
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                content = json_match.group(1)
            elif '```' in content:
                content = re.sub(r'```[^\n]*\n', '', content).replace('```', '')
            
            ai_results = json.loads(content)
            
            for result in ai_results:
                interactions.append({
                    'name': f"{result.get('drug1', '')} × {result.get('drug2', '')}",
                    'target_drugs': f"{result.get('drug1', '')}、{result.get('drug2', '')}",
                    'advice': result.get('description', ''),
                    'severity': result.get('severity', '併用注意'),
                    'source': 'AI'
                })
            
            logger.info(f"AI check completed: {len(interactions)} interactions found")
            return interactions
            
        except Exception as e:
            logger.error(f"AI interaction check failed: {e}")
            return []
    
    def _is_similar_interaction(self, interaction1: dict[str, Any], interaction2: dict[str, Any]) -> bool:
        """2つの相互作用が類似しているかチェック（重複除去用）"""
        try:
            # 対象薬名で比較
            targets1 = set(interaction1.get('target_drugs', '').split('、'))
            targets2 = set(interaction2.get('target_drugs', '').split('、'))
            
            # 対象薬が完全一致または大部分が一致していれば類似とみなす
            common = targets1 & targets2
            if len(common) >= 2 or (len(targets1) <= 2 and len(common) == len(targets1)):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Similarity check failed: {e}")
            return False
