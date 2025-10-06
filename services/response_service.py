import logging
import json
from typing import List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ResponseService:
    def __init__(self):
        pass
    
    def generate_response(self, drug_info: dict[str, Any]) -> str:
        """薬剤情報からLINE Bot用の応答メッセージを生成（詳細版）"""
        try:
            logger.info(f"Generating response for drug_info keys: {list(drug_info.keys())}")
            
            # 監査用トレース：matched_rules と matched_drugs のJSONログ出力
            self._log_audit_trace(drug_info)
            
            response_parts = []
            
            # ヘッダー
            response_parts.append("🏥 薬剤相互作用診断システム")
            response_parts.append("━━━━━━━━━━━━━━")
            
            # 診断結果では薬剤情報を省略（最初の検出で既に表示済み）
            
            # AI分析結果の確認
            ai_analysis = drug_info.get('ai_analysis', {})
            logger.info(f"AI analysis keys: {list(ai_analysis.keys()) if ai_analysis else 'None'}")
            
            # AI分析結果が空の場合のフォールバック（簡潔表示）
            if not ai_analysis or (not ai_analysis.get('patient_safety_alerts') and not ai_analysis.get('risk_summary')):
                # 相互作用チェック結果のみ表示（余計な前置きは出さない）
                interactions = drug_info.get('interactions') or []
                if interactions:
                    for interaction in interactions:
                        # 形式A: 従来の {drug1, drug2, risk, description, mechanism}
                        if 'drug1' in interaction and 'drug2' in interaction:
                            risk_emoji = self._get_risk_emoji(interaction.get('risk', 'medium'))
                            response_parts.append(f"{risk_emoji} {interaction.get('drug1')} + {interaction.get('drug2')}")
                            response_parts.append(f"リスク: {interaction.get('description', '相互作用あり')}")
                            if interaction.get('mechanism'):
                                response_parts.append(f"機序: {interaction.get('mechanism')}")
                            response_parts.append("")
                            continue

                        # 形式B: ルールエンジンの {id, name, severity, advice, targets}
                        name = interaction.get('name') or interaction.get('id', '相互作用注意')
                        severity = interaction.get('severity', 'moderate')
                        risk_emoji = self._get_severity_label(severity)
                        targets = interaction.get('target_drugs') or interaction.get('targets') or interaction.get('matched_drugs') or []
                        advice = interaction.get('advice') or interaction.get('description')

                        # 相互作用の表示（先頭の全角スペースを削除）
                        response_parts.append(f"{risk_emoji}：{name}")
                        response_parts.append(f"・対象：{self._format_targets(targets).strip()}")
                        if advice:
                            response_parts.append(f"・対応：{advice.strip()}")
                        response_parts.append("")
                else:
                    # 相互作用がない場合は簡潔に通知
                    response_parts.append("✅ 現在の薬剤組み合わせでは重大な相互作用は検出されませんでした。")
                    response_parts.append("")
            else:
                # AI分析結果が正常な場合の詳細表示
                
                # 1. 併用禁忌の詳細表示（重複除去）
                critical_risks = ai_analysis.get('risk_summary', {}).get('critical_risk', [])
                contraindicated_risks = [risk for risk in ai_analysis.get('detected_risks', []) if risk.get('risk_level') == 'contraindicated']
                
                # interactionsからも禁忌を検出
                contraindicated_interactions = [interaction for interaction in drug_info.get('interactions', []) if interaction.get('risk') == 'contraindicated']
                for interaction in contraindicated_interactions:
                    contraindicated_risks.append({
                        'involved_drugs': [interaction.get('drug1', ''), interaction.get('drug2', '')],
                        'description': interaction.get('description', '禁忌相互作用'),
                        'clinical_impact': '血中濃度上昇、過度の眠気、転倒リスク',
                        'recommendation': '医師・薬剤師に相談してください'
                    })
                
                # 重複を除去
                seen_combinations = set()
                unique_critical_risks = []
                unique_high_risks = []
                unique_medium_risks = []
                
                for risk in critical_risks + contraindicated_risks:
                    involved_drugs = tuple(sorted(risk.get('involved_drugs', [])))
                    if involved_drugs not in seen_combinations:
                        seen_combinations.add(involved_drugs)
                        unique_critical_risks.append(risk)
                
                if unique_critical_risks:
                    response_parts.append("🚨 併用禁忌 (重大リスク)")
                    response_parts.append("")
                    
                    for risk in unique_critical_risks:
                        response_parts.append(f"✅ 対象の薬: {', '.join(risk.get('involved_drugs', []))}")
                        response_parts.append(f"✅ 理由: {risk.get('description', '')}")
                        response_parts.append(f"✅ 考えられる症状: {risk.get('clinical_impact', '')}")
                        response_parts.append(f"✅ 推奨事項: {risk.get('recommendation', '')}")
                        response_parts.append("")
                        response_parts.append("━━━━━━━━━━━━━━")
                        response_parts.append("")
                
                # 2. 同効薬の重複の詳細表示（重複除去）
                high_risks = ai_analysis.get('risk_summary', {}).get('high_risk', [])
                if high_risks:
                    # 重複を除去
                    seen_combinations = set()
                    
                    for risk in high_risks:
                        involved_drugs = tuple(sorted(risk.get('involved_drugs', [])))
                        if involved_drugs not in seen_combinations:
                            seen_combinations.add(involved_drugs)
                            unique_high_risks.append(risk)
                    
                    if unique_high_risks:
                        response_parts.append("⚠️ 同効薬の重複 (注意リスク)")
                        response_parts.append("")
                        for risk in unique_high_risks:
                            response_parts.append(f"✅ 対象の薬: {', '.join(risk.get('involved_drugs', []))}")
                            if risk.get('involved_categories'):
                                response_parts.append(f"✅ 薬効分類: {', '.join(risk.get('involved_categories', []))}")
                            response_parts.append(f"✅ 理由: {risk.get('description', '')}")
                            response_parts.append(f"✅ 考えられる症状: {risk.get('clinical_impact', '')}")
                            response_parts.append(f"✅ 推奨事項: {risk.get('recommendation', '')}")
                            response_parts.append("")
                
                # 3. 併用注意の詳細表示（重複除去）
                medium_risks = ai_analysis.get('risk_summary', {}).get('medium_risk', [])
                if medium_risks:
                    # 重複を除去
                    seen_combinations = set()
                    
                    for risk in medium_risks:
                        involved_drugs = tuple(sorted(risk.get('involved_drugs', [])))
                        if involved_drugs not in seen_combinations:
                            seen_combinations.add(involved_drugs)
                            unique_medium_risks.append(risk)
                    
                    if unique_medium_risks:
                        response_parts.append("📋 併用注意 (軽微リスク)")
                        response_parts.append("")
                        for risk in unique_medium_risks:
                            response_parts.append(f"✅ 対象の薬: {', '.join(risk.get('involved_drugs', []))}")
                            response_parts.append(f"✅ 理由: {risk.get('description', '')}")
                            response_parts.append(f"✅ 考えられる症状: {risk.get('clinical_impact', '')}")
                            response_parts.append(f"✅ 推奨事項: {risk.get('recommendation', '')}")
                            response_parts.append("")
                
                # 4. 患者プロファイル分析
                if ai_analysis.get('detailed_analysis', {}).get('patient_profile'):
                    profile = ai_analysis['detailed_analysis']['patient_profile']
                    response_parts.append("👤 患者プロファイル分析")
                    if profile.get('likely_conditions'):
                        response_parts.append(f"推定疾患: {', '.join(profile['likely_conditions'])}")
                    if profile.get('polypharmacy_risk') != 'low':
                        response_parts.append(f"多剤併用リスク: {profile['polypharmacy_risk']}")
                    response_parts.append("")
                
                # 5. 問題がない場合の表示
                if not unique_critical_risks and not high_risks and not unique_medium_risks:
                    response_parts.append("✅ 診断結果")
                    response_parts.append("現在の薬剤組み合わせでは、")
                    response_parts.append("重大な相互作用や注意すべきリスクは")
                    response_parts.append("検出されませんでした。")
                    response_parts.append("")
                    response_parts.append("💡 一般的な注意事項:")
                    response_parts.append("・定期的な健康チェックを継続してください")
                    response_parts.append("・体調に変化があれば医師に相談してください")
                    response_parts.append("・薬の副作用が出た場合はすぐに医師に連絡してください")
                    response_parts.append("")
                
                # 5. 代替療法の提案
                if ai_analysis.get('detailed_analysis', {}).get('alternative_therapies'):
                    response_parts.append("💡 代替療法の提案")
                    for alt in ai_analysis['detailed_analysis']['alternative_therapies']:
                        priority_emoji = self._get_priority_emoji(alt.get('priority', 'medium'))
                        response_parts.append(f"{priority_emoji} {alt.get('problem', '問題')}")
                        response_parts.append(f"提案: {alt.get('suggestion', '')}")
                        if alt.get('alternatives'):
                            response_parts.append("代替案:")
                            for alternative in alt['alternatives']:
                                response_parts.append(f"・{alternative}")
                        response_parts.append("")
            
            # 参考情報の注意書き
            response_parts.append("━━━━━━━━━━━━━━")
            response_parts.append("⚠️ 重要なお知らせ")
            response_parts.append("この診断結果はAIによる分析結果です。")
            response_parts.append("最終的な判断は医師・薬剤師にご相談ください。")
            response_parts.append("緊急時は直ちに医療機関を受診してください。")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"応答メッセージ生成エラー: {e}", exc_info=True)
            return f"エラーが発生しました: {str(e)}\n薬剤師にご相談ください。"
    
    def _get_risk_emoji(self, risk_level: str) -> str:
        """リスクレベルに応じた絵文字を返す"""
        risk_emojis = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': '��'
        }
        return risk_emojis.get(risk_level, '⚠️')
    
    def _get_priority_emoji(self, priority: str) -> str:
        """優先度に応じた絵文字を取得"""
        priority_emojis = {
            'critical': '🚨',
            'urgent': '🚨',
            'high': '⚠️',
            'medium': '📋',
            'low': 'ℹ️'
        }
        return priority_emojis.get(priority, 'ℹ️')
    
    def generate_simple_response(self, drug_data: List) -> str:
        """シンプルな応答メッセージを生成（改善版）"""
        if not drug_data:
            return """🩺【薬剤名未検出】
━━━━━━━━━━━━━━
❌ 薬剤名を読み取ることができませんでした

💡 推奨事項:
・より鮮明な画像で再度お試しください
・文字がはっきり見えるように撮影してください
━━━━━━━━━━━━━━"""
        
        # 薬剤データの形式を判定
        if isinstance(drug_data[0], dict):
            # AI抽出結果の詳細情報
            drug_info_list = drug_data
        else:
            # 従来の薬剤名リスト
            drug_names = drug_data
            drug_info_list = []
            
            # DrugServiceを使用して薬剤分類を取得
            from services.drug_service import DrugService
            drug_service = DrugService()
            
            for drug_name in drug_names:
                logger.info(f"薬剤分類処理開始: {drug_name}")
                # 薬剤名補正機能を含む完全な分析を実行
                analysis = drug_service.ai_matcher.analyze_drug_name(drug_name)
                corrected_name = analysis.get('corrected', drug_name)  # 修正された薬剤名を取得
                category = analysis.get('category', 'unknown')
                
                drug_info = {
                    'name': drug_name,
                    'ai_category': category,
                    'kegg_category': '',
                    'strength': '',
                    'dose': '',
                    'freq': ''
                }
                drug_info_list.append(drug_info)
                logger.info(f"薬剤分類結果: {drug_name} -> {corrected_name} -> {category}")
        
        response_parts = []
        response_parts.append("🩺【薬剤検出完了】")
        response_parts.append("━━━━━━━━━━━━━━")
        response_parts.append(f"✅ {len(drug_info_list)}件検出しました")
        response_parts.append("")
        response_parts.append("")
        response_parts.append("📋 検出された薬剤:")
        response_parts.append("")
        
        # 薬剤カテゴリの日本語マッピング
        category_mapping = {
            'pde5_inhibitor': 'PDE5阻害薬',
            'nitrate': '硝酸薬',
            'arni': 'ARNI（心不全治療薬）',
            'ca_antagonist_arb_combination': 'Ca拮抗薬+ARB配合剤',
            'ace_inhibitor': 'ACE阻害薬',
            'p_cab': 'P-CAB（胃薬）',
            'ppi': 'PPI（胃薬）',
            'benzodiazepine': 'ベンゾジアゼピン系',
            'barbiturate': 'バルビツール酸系',
            'opioid': 'オピオイド',
            'nsaid': 'NSAIDs',
            'statin': 'スタチン',
            'arb': 'ARB',
            'beta_blocker': 'β遮断薬',
            'ca_antagonist': 'カルシウム拮抗薬',
            'diuretic': '利尿薬',
            'antihistamine': '抗ヒスタミン薬',
            'antacid': '制酸薬',
            'anticoagulant': '抗凝固薬',
            'diabetes_medication': '糖尿病治療薬',
            'antibiotic': '抗生物質',
            'antidepressant': '抗うつ薬',
            'antipsychotic': '抗精神病薬',
            'bronchodilator': '気管支拡張薬',
            'inhaled_corticosteroid': '吸入ステロイド薬',
            'leukotriene_receptor_antagonist': 'ロイコトリエン受容体拮抗薬',
            'mucolytic': '去痰薬',
            'bph_medication': '前立腺肥大症治療薬',
            'cardiac_glycoside': '強心配糖体',
            'antiarrhythmic': '抗不整脈薬',
            'antirheumatic': '抗リウマチ薬',
            'corticosteroid': '副腎皮質ホルモン',
            'immunosuppressant': '免疫抑制薬',
            'uric_acid_lowering': '尿酸生成抑制薬',
            'phosphate_binder': 'リン吸着薬',
            'vitamin_d': '活性型ビタミンD製剤',
            'sleep_medication': '睡眠薬・催眠薬',
            'ssri_antidepressant': 'SSRI抗うつ薬',
            'cyp3a4_inhibitor': 'CYP3A4阻害薬',
            'orexin_receptor_antagonist': 'オレキシン受容体拮抗薬（睡眠薬）',
            'macrolide_antibiotic_cyp3a4_inhibitor': 'マクロライド系抗菌薬・CYP3A4阻害薬',
            'antiepileptic': '抗てんかん薬',
            'antiplatelet': '抗血小板薬',
            'iron_supplement': '鉄剤',
            'unknown': '分類不明'
        }
        
        # 修正された薬剤名を使用して表示
        for i, drug_info in enumerate(drug_info_list, 1):
            drug_name = drug_info.get('name', '')
            final_classification = drug_info.get('final_classification', '')
            class_hint = drug_info.get('class_hint', '')
            kegg_category = drug_info.get('kegg_category', '')
            strength = drug_info.get('strength', '')
            dose = drug_info.get('dose', '')
            freq = drug_info.get('freq', '')
            
            # カテゴリの優先順位: 最終分類 > KEGG > AIヒント > デフォルト
            if final_classification and final_classification != '分類未設定':
                japanese_category = final_classification
            elif kegg_category:
                japanese_category = kegg_category
            elif class_hint:
                japanese_category = f"{class_hint}（AI推定）"
            else:
                japanese_category = '不明'
            
            # 番号記号の取得
            number_symbol = f"{i}."
            
            # 薬剤名と用量の表示
            display_name = drug_name
            def nz(v, fallback="不明"):
                return fallback if v in (None, "", "None") else v
            if strength:
                display_name += f" {nz(strength)}"
            if dose:
                display_name += f" {nz(dose)}"
            if freq:
                display_name += f" {nz(freq)}"
            
            response_parts.append(f"{number_symbol} {display_name}")
            response_parts.append(f"分類: {japanese_category}")
            response_parts.append("")
        
        response_parts.append("🔍 「診断」で飲み合わせチェックを実行できます")
        response_parts.append("━━━━━━━━━━━━━━")
        response_parts.append("⚠️ 重要なお知らせ")
        response_parts.append("この診断結果はAIによる分析結果です。")
        response_parts.append("最終的な判断は医師・薬剤師にご相談ください。")
        response_parts.append("緊急時は直ちに医療機関を受診してください。")
        
        return "\n".join(response_parts)
    
    def generate_detailed_analysis(self, drug_info: dict[str, Any]) -> str:
        """詳細な分析結果を生成（HTMLアプリ用）"""
        try:
            analysis_parts = []
            
            # 基本情報
            analysis_parts.append("## 薬剤情報分析結果")
            analysis_parts.append("")
            
            # 検出薬剤
            if drug_info['detected_drugs']:
                analysis_parts.append("### 検出された薬剤")
                for drug in drug_info['detected_drugs']:
                    analysis_parts.append(f"- **{drug['name']}**")
                    analysis_parts.append(f"  - 分類: {drug['category']}")
                    analysis_parts.append(f"  - 一般名: {drug['generic_name']}")
                analysis_parts.append("")
            
            # 相互作用分析
            if drug_info['interactions']:
                analysis_parts.append("### 相互作用分析")
                for interaction in drug_info['interactions']:
                    analysis_parts.append(f"- **{interaction['drug1']} + {interaction['drug2']}**")
                    analysis_parts.append(f"  - リスク: {interaction.get('description', '相互作用あり')}")
                    analysis_parts.append(f"  - 機序: {interaction.get('mechanism', '不明')}")
                analysis_parts.append("")
            
            # 同効薬分析
            if drug_info['same_effect_warnings']:
                analysis_parts.append("### 同効薬重複分析")
                for warning in drug_info['same_effect_warnings']:
                    analysis_parts.append(f"- **{warning['drug1']} + {warning['drug2']}**")
                    analysis_parts.append(f"  - 機序: {warning['mechanism']}")
                    analysis_parts.append(f"  - リスクレベル: {warning['risk_level']}")
                analysis_parts.append("")
            
            # 分類重複分析
            if drug_info['category_duplicates']:
                analysis_parts.append("### 薬剤分類重複分析")
                for duplicate in drug_info['category_duplicates']:
                    analysis_parts.append(f"- **{duplicate['category']}**: {duplicate['count']}種類")
                    for drug in duplicate['drugs']:
                        analysis_parts.append(f"  - {drug}")
                analysis_parts.append("")
            
            # KEGG情報
            if drug_info['kegg_info']:
                analysis_parts.append("### KEGGデータベース情報")
                for kegg in drug_info['kegg_info']:
                    analysis_parts.append(f"- **{kegg['drug_name']}**")
                    if kegg.get('kegg_id'):
                        analysis_parts.append(f"  - KEGG ID: {kegg['kegg_id']}")
                    if kegg.get('pathways'):
                        analysis_parts.append(f"  - 関連パスウェイ: {', '.join(kegg['pathways'])}")
                    if kegg.get('targets'):
                        analysis_parts.append(f"  - 作用ターゲット: {', '.join(kegg['targets'])}")
                analysis_parts.append("")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"詳細分析生成エラー: {e}")
            return "詳細分析の生成中にエラーが発生しました。"
    
    def generate_error_response(self, error_type: str = "general") -> str:
        """エラー応答メッセージを生成"""
        error_messages = {
            "ocr": """🩺【画像読み取りエラー】
━━━━━━━━━━━━━━
❌ 画像の読み取りに失敗しました

💡 推奨事項:
・より鮮明な画像で再度お試しください
・文字がはっきり見えるように撮影してください
━━━━━━━━━━━━━━""",
            "drug_lookup": """🩺【薬剤情報取得エラー】
━━━━━━━━━━━━━━
❌ 薬剤情報の取得に失敗しました

💡 推奨事項:
・薬剤師にご相談ください
・このBotは補助ツールです
━━━━━━━━━━━━━━""",
            "general": """🩺【エラーが発生しました】
━━━━━━━━━━━━━━
❌ 申し訳ございません
エラーが発生しました

💡 推奨事項:
・しばらく時間をおいて再度お試しください
・薬剤師にご相談ください
━━━━━━━━━━━━━━"""
        }
        
        return error_messages.get(error_type, error_messages["general"]) 

    def generate_simple_response(self, detected_drugs, interaction_result=None, show_interactions: bool = True):
        """薬剤検出結果のシンプルな表示を生成（分類付き）。
        show_interactions=True のときのみ相互作用セクションを表示。
        """
        try:
            if not detected_drugs:
                return "薬剤名が検出されませんでした。より鮮明な画像で撮影してください。"
            
            response_parts = []
            response_parts.append("🏥 【薬剤検出完了】")
            response_parts.append("━━━━━━━━━")
            response_parts.append(f"✅ {len(detected_drugs)}剤検出しました")
            response_parts.append("")
            response_parts.append("📋 検出された薬剤:")
            
            # 薬剤分類マッピング
            category_mapping = {
                'pde5_inhibitor': 'PDE5阻害薬',
                'nitrate': '硝酸薬',
                'arni': 'ARNI (心不全治療薬)',
                'angiotensin_receptor_blocker': 'ARB',
                'ca_antagonist_arb_combination': 'ARB・Ca拮抗薬配合',
                'ca_antagonist': 'カルシウム拮抗薬',
                'ace_inhibitor': 'ACE阻害薬',
                'p_cab': 'P-CAB (胃薬)',
                'ppi': 'プロトンポンプ阻害薬',
                'sleep_medication': '睡眠薬',
                'orexin_receptor_antagonist': 'オレキシン受容体拮抗薬',
                'melatonin_receptor_agonist': 'メラトニン受容体作動薬',
                'ssri_antidepressant': 'SSRI抗うつ薬',
                'macrolide_antibiotic_cyp3a4_inhibitor': 'マクロライド系抗生物質',
                'benzodiazepine': 'ベンゾジアゼピン系',
                'unknown': '分類不明'
            }
            
            # アラビア数字 + ドットで統一
            number_symbols = []
            
            for i, drug in enumerate(detected_drugs):
                # 薬剤情報の形式を判定
                if isinstance(drug, dict):
                    # 新しい分類システム（辞書形式）
                    drug_name = drug.get('name', drug.get('generic', drug.get('raw', 'Unknown')))
                    # 名称の装飾（表示のみ）
                    if 'アスピリン' in drug_name and '腸溶' not in drug_name:
                        drug_name = drug_name.replace('アスピリン', 'アスピリン腸溶')
                    if 'ニフェジピン' in drug_name and '徐放' not in drug_name:
                        drug_name = drug_name.replace('ニフェジピン', 'ニフェジピン徐放')
                    if 'ロキソプロフェン' in drug_name and 'Naテープ' not in drug_name:
                        drug_name = drug_name.replace('ロキソプロフェン', 'ロキソプロフェンNaテープ')
                    classification = drug.get('final_classification', '分類未設定')
                    strength = drug.get('strength', '')
                    dose = drug.get('dose', '')
                    freq = drug.get('freq', '')
                    
                    # 薬剤情報を表示
                    symbol = f"{i+1}."
                    response_parts.append(f"{symbol} {drug_name}")
                    def nz(v, fallback="不明"):
                        return fallback if v in (None, "", "None", "null") else v
                    if strength:
                        response_parts.append(f"   用量: {nz(strength)}")
                    if dose:
                        # 漢方薬の単位を「包」に修正
                        display_dose = nz(dose).replace('cap', 'カプセル')
                        if any(keyword in drug_name for keyword in ["芍薬甘草湯", "エキス顆粒", "ツムラ"]):
                            display_dose = display_dose.replace("錠", "包")
                        # 外用剤の数量表示を修正
                        if "テープ" in drug_name or "外用" in classification:
                            response_parts.append(f"   数量: {display_dose}")
                        else:
                            response_parts.append(f"   用法: {display_dose}")
                    if freq:
                        response_parts.append(f"   頻度: {nz(freq)}")
                    response_parts.append(f"   分類: {classification}")
                    response_parts.append("")
                else:
                    # 従来の文字列形式
                    drug_lower = drug.lower()
                    category = 'unknown'
                    
                    if any(term in drug_lower for term in ['タダラフィル']):
                        category = 'pde5_inhibitor'
                    elif any(term in drug_lower for term in ['ニコランジル']):
                        category = 'nitrate'
                    elif any(term in drug_lower for term in ['エンレスト']):
                        category = 'arni'
                    elif any(term in drug_lower for term in ['テラムロ']):
                        category = 'ca_antagonist_arb_combination'
                    elif any(term in drug_lower for term in ['エナラプリル']):
                        category = 'ace_inhibitor'
                    elif any(term in drug_lower for term in ['タケキャブ']):
                        category = 'p_cab'
                    elif any(term in drug_lower for term in ['ランソプラゾール']):
                        category = 'ppi'
                    elif any(term in drug_lower for term in ['ベルソムラ']):
                        category = 'orexin_receptor_antagonist'
                    elif any(term in drug_lower for term in ['デビゴ']):
                        category = 'orexin_receptor_antagonist'
                    elif any(term in drug_lower for term in ['フルボキサミン']):
                        category = 'ssri_antidepressant'
                    elif any(term in drug_lower for term in ['ロゼレム']):
                        category = 'melatonin_receptor_agonist'
                    elif any(term in drug_lower for term in ['アムロジピン']):
                        category = 'ca_antagonist'
                    elif any(term in drug_lower for term in ['クラリスロマイシン']):
                        category = 'macrolide_antibiotic_cyp3a4_inhibitor'
                    
                    category_jp = category_mapping.get(category, '分類不明')
                    symbol = f"{i+1}."
                    
                    # 名称の装飾（表示のみ）
                    display_name = drug
                    if 'アスピリン' in display_name and '腸溶' not in display_name:
                        display_name = display_name.replace('アスピリン', 'アスピリン腸溶')
                    if 'ニフェジピン' in display_name and '徐放' not in display_name:
                        display_name = display_name.replace('ニフェジピン', 'ニフェジピン徐放')

                    response_parts.append(f"{symbol} {display_name}")
                    response_parts.append(f"   分類: {category_jp}")
                    response_parts.append("")
            
            response_parts.append("")
            
            # 相互作用の表示はフラグで制御
            if show_interactions:
                if interaction_result and interaction_result.get("has_interactions"):
                    response_parts.append("⚠️ 【相互作用チェック結果】")
                    response_parts.append("━━━━━━━━━━━━━━")
                    
                    # 重大な相互作用
                    major_interactions = interaction_result.get("major_interactions", [])
                    if major_interactions:
                        response_parts.append("🚨 重大な相互作用:")
                        for interaction in major_interactions:
                            response_parts.append(f"• {interaction.get('name', '相互作用')}")
                            target_drugs = interaction.get('target_drugs', '')
                            if target_drugs:
                                response_parts.append(f"  対象：{target_drugs}")
                            response_parts.append(f"  対応：{interaction.get('advice', '')}")
                            response_parts.append("")
                    
                    # 注意すべき相互作用
                    moderate_interactions = interaction_result.get("moderate_interactions", [])
                    if moderate_interactions:
                        response_parts.append("⚠️ 注意すべき相互作用:")
                        for interaction in moderate_interactions:
                            response_parts.append(f"• {interaction.get('name', '相互作用')}")
                            target_drugs = interaction.get('target_drugs', '')
                            if target_drugs:
                                response_parts.append(f"  対象：{target_drugs}")
                            response_parts.append(f"  対応：{interaction.get('advice', '')}")
                            response_parts.append("")
                    
                    response_parts.append("━━━━━━━━━━━━━━")
                else:
                    response_parts.append("✅ 相互作用は検出されませんでした")
                    response_parts.append("")
            
            response_parts.append("🔍 「診断」で詳細な飲み合わせチェックを実行できます")
            response_parts.append("━━━━━━━━━━━━━━")
            response_parts.append("⚠️ 重要なお知らせ")
            response_parts.append("この診断結果はAIによる分析結果です。")
            response_parts.append("最終的な判断は医師・薬剤師にご相談ください。")
            response_parts.append("緊急時は直ちに医療機関を受診してください。")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            return "薬剤検出結果の生成中にエラーが発生しました。"
    
    def _get_severity_label(self, severity: str) -> str:
        """severityを表示用ラベルに変換"""
        return {
            "major": "🚨 重大",
            "moderate": "⚠️ 併用注意", 
            "minor": "ℹ️ 参考"
        }.get(severity, "⚠️ 併用注意")
    
    def _format_targets(self, targets) -> str:
        """対象薬剤リストを表示用にフォーマット（先頭の全角スペースを削除）"""
        if not targets:
            return "（対象薬の特定に失敗）"
        
        # targetsが文字列の場合はそのまま返す（target_drugsの場合）
        if isinstance(targets, str):
            return targets.strip()
        
        # targetsがリストの場合は結合して返す
        if isinstance(targets, list):
            return "、".join(targets).strip()
        
        # その他の場合は文字列に変換
        return str(targets).strip()

    def _log_audit_trace(self, drug_info: dict[str, Any]) -> None:
        """監査用トレース：matched_rules と matched_drugs のJSONログ出力"""
        try:
            # 薬剤情報の抽出
            drugs = drug_info.get('drugs', [])
            interactions = drug_info.get('interactions', [])
            
            # 薬剤名のリストを作成
            drug_names = []
            for drug in drugs:
                name = drug.get('generic') or drug.get('brand') or drug.get('raw', '')
                if name:
                    drug_names.append(name)
            
            # 相互作用ルールの情報を抽出（重大度を日本語→英語に正規化して保存）
            matched_rules = []
            def _normalize_severity(sev: str) -> str:
                if not sev:
                    return ''
                sev = str(sev).strip()
                mapping = {
                    '重大': 'major',
                    '併用注意': 'moderate',
                    '注意': 'moderate',
                    '軽微': 'minor',
                    'major': 'major',
                    'moderate': 'moderate',
                    'minor': 'minor'
                }
                return mapping.get(sev, sev)

            for interaction in interactions:
                sev_norm = _normalize_severity(interaction.get('severity', ''))
                rule_info = {
                    'id': interaction.get('id', ''),
                    'name': interaction.get('name', ''),
                    'severity': sev_norm,
                    'target_drugs': interaction.get('target_drugs', ''),
                    'advice': interaction.get('advice', '')
                }
                matched_rules.append(rule_info)
            
            # 監査用JSONログを出力
            audit_data = {
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'matched_drugs': drug_names,
                'matched_rules': matched_rules,
                'total_interactions': len(matched_rules),
                'major_interactions': len([r for r in matched_rules if r.get('severity') == 'major']),
                'moderate_interactions': len([r for r in matched_rules if r.get('severity') in ['moderate', 'minor']])
            }
            
            logger.info(f"AUDIT_TRACE: {json.dumps(audit_data, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            logger.error(f"Audit trace logging failed: {e}")

    def generate_manual_addition_guide(self):
        """手動追加ガイドを生成"""
        try:
            response_parts = []
            
            response_parts.append("📝 **薬剤の手動追加ガイド**")
            response_parts.append("")
            response_parts.append("検出されていない薬剤がある場合は、以下の形式で手動追加してください：")
            response_parts.append("")
            response_parts.append("**形式**: 薬剤追加：薬剤名")
            response_parts.append("")
            response_parts.append("**例**:")
            response_parts.append("• 薬剤追加：テラムロAP")
            response_parts.append("• 薬剤追加：タケキャブ")
            response_parts.append("• 薬剤追加：エンレスト")
            response_parts.append("• 薬剤追加：エナラプリル")
            response_parts.append("• 薬剤追加：ランソプラゾール")
            response_parts.append("• 薬剤追加：タダラフィル")
            response_parts.append("• 薬剤追加：ニコランジル")
            response_parts.append("")
            response_parts.append("**注意**:")
            response_parts.append("• 薬剤名は正確に入力してください")
            response_parts.append("• 複数の薬剤を追加する場合は、1つずつ入力してください")
            response_parts.append("• 追加後は「診断」で相互作用分析を実行してください")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating manual addition guide: {e}")
            return "手動追加ガイドの生成中にエラーが発生しました。"