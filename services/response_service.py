import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ResponseService:
    def __init__(self):
        pass
    
    def generate_response(self, drug_info: Dict[str, Any]) -> str:
        """薬剤情報からLINE Bot用の応答メッセージを生成（AI強化版）"""
        try:
            logger.info(f"Generating response for drug_info keys: {list(drug_info.keys())}")
            
            response_parts = []
            
            # ヘッダー
            response_parts.append("🏥 **薬剤相互作用診断システム**")
            response_parts.append("━━━━━━━━━━━━━━━━━━━━━━")
            
            # 検出された薬剤の表示
            if drug_info.get('detected_drugs'):
                response_parts.append("📋 **検出された薬剤**")
                for drug in drug_info['detected_drugs']:
                    category = drug.get('ai_category', drug.get('category', '不明'))
                    response_parts.append(f"・{drug['name']} ({category})")
                response_parts.append("")
            else:
                response_parts.append("📋 **検出された薬剤**")
                response_parts.append("薬剤情報が見つかりませんでした")
                response_parts.append("")
            
            # AI分析結果の確認
            ai_analysis = drug_info.get('ai_analysis', {})
            logger.info(f"AI analysis keys: {list(ai_analysis.keys()) if ai_analysis else 'None'}")
            
            # 全体的なリスク評価の表示
            if ai_analysis.get('overall_risk_assessment'):
                risk_assessment = ai_analysis['overall_risk_assessment']
                risk_level = risk_assessment.get('overall_risk_level', 'low')
                risk_emoji = self._get_risk_emoji(risk_level)
                response_parts.append(f"{risk_emoji} **全体的なリスク評価: {risk_level.upper()}**")
                response_parts.append(f"リスクスコア: {risk_assessment.get('risk_score', 0)}")
                response_parts.append("")
            
            # 患者安全性アラートの表示（優先度順）
            if ai_analysis.get('patient_safety_alerts'):
                alerts = ai_analysis['patient_safety_alerts']
                # 優先度順にソート（critical > high > medium）
                priority_order = {'critical': 1, 'high': 2, 'medium': 3}
                alerts.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 4))
                
                for alert in alerts:
                    priority_emoji = self._get_priority_emoji(alert.get('priority', 'medium'))
                    response_parts.append(f"{priority_emoji} **{alert.get('title', 'アラート')}**")
                    response_parts.append(f"{alert.get('message', '')}")
                    
                    if alert.get('symptoms'):
                        response_parts.append("症状:")
                        for symptom in alert['symptoms']:
                            response_parts.append(f"・{symptom}")
                    
                    if alert.get('items'):
                        response_parts.append("モニタリング項目:")
                        for item in alert['items']:
                            response_parts.append(f"・{item}")
                    
                    if alert.get('recommendation'):
                        response_parts.append(f"推奨事項: {alert['recommendation']}")
                    
                    response_parts.append("")
                    response_parts.append("━━━━━━━━━━━━━━━━━━━━━━")
                    response_parts.append("")
            
            # 詳細な臨床分析の表示
            if ai_analysis.get('detailed_analysis'):
                detailed = ai_analysis['detailed_analysis']
                
                # 患者プロファイル
                if detailed.get('patient_profile'):
                    profile = detailed['patient_profile']
                    response_parts.append("👤 **患者プロファイル分析**")
                    if profile.get('likely_conditions'):
                        response_parts.append(f"推定疾患: {', '.join(profile['likely_conditions'])}")
                    if profile.get('polypharmacy_risk') != 'low':
                        response_parts.append(f"多剤併用リスク: {profile['polypharmacy_risk']}")
                    response_parts.append("")
                
                # 臨床シナリオ
                if detailed.get('clinical_scenarios'):
                    response_parts.append("🏥 **臨床シナリオ**")
                    for scenario in detailed['clinical_scenarios']:
                        response_parts.append(f"📋 {scenario.get('description', 'シナリオ')}")
                        response_parts.append(f"対象薬剤: {', '.join(scenario.get('drugs', []))}")
                        response_parts.append("考慮事項:")
                        for consideration in scenario.get('considerations', []):
                            response_parts.append(f"・{consideration}")
                        response_parts.append("")
                
                # 代替療法の提案
                if detailed.get('alternative_therapies'):
                    response_parts.append("💡 **代替療法の提案**")
                    for alt in detailed['alternative_therapies']:
                        priority_emoji = self._get_priority_emoji(alt.get('priority', 'medium'))
                        response_parts.append(f"{priority_emoji} **{alt.get('problem', '問題')}**")
                        response_parts.append(f"提案: {alt.get('suggestion', '')}")
                        if alt.get('alternatives'):
                            response_parts.append("代替案:")
                            for alternative in alt['alternatives']:
                                response_parts.append(f"・{alternative}")
                        response_parts.append("")
            
            # 従来の相互作用チェック（バックアップ）
            if drug_info.get('interactions'):
                response_parts.append("💊 **相互作用チェック**")
                for interaction in drug_info['interactions']:
                    risk_emoji = self._get_risk_emoji(interaction.get('risk', 'medium'))
                    response_parts.append(f"{risk_emoji} {interaction['drug1']} + {interaction['drug2']}")
                    response_parts.append(f"リスク: {interaction.get('description', '相互作用あり')}")
                    if interaction.get('mechanism'):
                        response_parts.append(f"機序: {interaction['mechanism']}")
                    response_parts.append("")
            
            # AI分析結果が空の場合のフォールバック
            if not ai_analysis or not ai_analysis.get('patient_safety_alerts'):
                response_parts.append("⚠️ **診断結果**")
                response_parts.append("AI分析が完了しませんでした。")
                response_parts.append("従来の相互作用チェック結果を表示します。")
                response_parts.append("")
            
            # 参考情報の注意書き
            response_parts.append("━━━━━━━━━━━━━━━━━━━━━━")
            response_parts.append("⚠️ **重要なお知らせ**")
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
    
    def generate_simple_response(self, drug_names: List[str]) -> str:
        """シンプルな応答メッセージを生成（改善版）"""
        if not drug_names:
            return """🩺【薬剤名未検出】
━━━━━━━━━━━━━━
❌ 薬剤名を読み取ることができませんでした

💡 推奨事項:
・より鮮明な画像で再度お試しください
・文字がはっきり見えるように撮影してください
━━━━━━━━━━━━━━"""
        
        response_parts = []
        response_parts.append("🩺【薬剤検出完了】")
        response_parts.append("━━━━━━━━━━━━━━")
        response_parts.append(f"✅ {len(drug_names)}件の薬剤を検出しました")
        response_parts.append(f"現在のリスト: {len(drug_names)}件")
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
            'ca_antagonist': 'Ca拮抗薬',
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
            'unknown': '分類不明'
        }
        
        # 薬剤名からカテゴリを推定する簡易的なマッピング
        drug_category_mapping = {
            'タダラフィル': 'pde5_inhibitor',
            'シルデナフィル': 'pde5_inhibitor',
            'バルデナフィル': 'pde5_inhibitor',
            'ニコランジル': 'nitrate',
            'ニトログリセリン': 'nitrate',
            'エンレスト': 'arni',
            'サクビトリル': 'arni',
            'テラムロ': 'ca_antagonist_arb_combination',
            'アムロジピン': 'ca_antagonist',
            'テルミサルタン': 'arb',
            'エナラプリル': 'ace_inhibitor',
            'カプトプリル': 'ace_inhibitor',
            'リシノプリル': 'ace_inhibitor',
            'タケキャブ': 'p_cab',
            'ボノプラザン': 'p_cab',
            'ランソプラゾール': 'ppi',
            'オメプラゾール': 'ppi',
            'エソメプラゾール': 'ppi',
            'ジアゼパム': 'benzodiazepine',
            'クロナゼパム': 'benzodiazepine',
            'アルプラゾラム': 'benzodiazepine',
            'ロラゼパム': 'benzodiazepine',
            'アスピリン': 'nsaid',
            'イブプロフェン': 'nsaid',
            'ロキソプロフェン': 'nsaid',
            'ワルファリン': 'anticoagulant',
            'ダビガトラン': 'anticoagulant',
            'シンバスタチン': 'statin',
            'アトルバスタチン': 'statin',
            'メトホルミン': 'diabetes_medication',
            'インスリン': 'diabetes_medication'
        }
        
        for i, drug in enumerate(drug_names, 1):
            # 薬剤名からカテゴリを推定
            category = drug_category_mapping.get(drug, 'unknown')
            category_jp = category_mapping.get(category, category)
            
            response_parts.append(f"① {drug}")
            response_parts.append(f"   分類: {category_jp}")
            response_parts.append("")
        
        response_parts.append("💡 「診断」で飲み合わせチェックを実行できます")
        response_parts.append("━━━━━━━━━━━━━━")
        
        return "\n".join(response_parts)
    
    def generate_detailed_analysis(self, drug_info: Dict[str, Any]) -> str:
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