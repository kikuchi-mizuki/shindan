import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ResponseService:
    def __init__(self):
        pass
    
    def generate_response(self, drug_info: Dict[str, Any]) -> str:
        """薬剤情報からLINE Bot用の応答メッセージを生成（拡張版）"""
        try:
            response_parts = []
            
            # ヘッダー
            response_parts.append("【薬剤情報チェック結果】")
            response_parts.append("━━━━━━━━━━━━━━")
            
            # diagnosis_detailsがあれば優先してリッチテキストで表示
            if drug_info.get('diagnosis_details'):
                for detail in drug_info['diagnosis_details']:
                    # 薬剤リストの重複除去
                    unique_drugs = list(dict.fromkeys(detail.get('drugs', [])))
                    response_parts.append(f"【{detail.get('type', '診断結果')}】\n" +
                        f"\n" +
                        f"■ 対象の薬: \n  {', '.join(unique_drugs)}\n" +
                        f"■ 薬効分類: {detail.get('category', '不明')}\n" +
                        f"\n" +
                        f"■ 理由:\n{detail.get('reason', '情報がありません')}\n" +
                        f"\n" +
                        f"■ 考えられる症状:\n{detail.get('symptoms', '情報がありません')}\n" +
                        f"\n{'='*15}\n")
            else:
                # 1. 同効薬重複警告（最優先）
                if drug_info['same_effect_warnings']:
                    for warning in drug_info['same_effect_warnings']:
                        response_parts.append("【同効薬の重複】\n" +
                            f"\n■ 対象の薬: {warning['drug1']}、{warning['drug2']}\n" +
                            f"■ 薬効分類: {warning.get('category', '不明')}\n" +
                            f"\n■ 理由:\n{warning.get('reason') or '情報がありません'}\n" +
                            f"\n■ 考えられる症状:\n{warning.get('symptoms') or '情報がありません'}\n" +
                            f"\n{'='*15}\n")
                # 2. 併用禁忌・併用注意の相互作用
                critical_interactions = [i for i in drug_info['interactions'] if i.get('risk') in ['critical', 'high']]
                if critical_interactions:
                    for interaction in critical_interactions:
                        response_parts.append("【併用禁忌・併用注意】\n" +
                            f"\n■ 対象の薬: {interaction['drug1']}、{interaction['drug2']}\n" +
                            f"■ リスク: {interaction.get('description', '相互作用あり')}\n" +
                            f"\n■ 理由:\n{interaction.get('reason') or '情報がありません'}\n" +
                            f"\n■ 考えられる症状:\n{interaction.get('symptoms') or '情報がありません'}\n" +
                            f"\n{'='*15}\n")
                # 3. 薬剤分類重複チェック
                if drug_info['category_duplicates']:
                    for duplicate in drug_info['category_duplicates']:
                        response_parts.append("【薬剤分類重複】\n" +
                            f"\n■ 分類: {duplicate['category']}\n" +
                            f"■ 種類数: {duplicate['count']}\n" +
                            f"■ 薬剤: {', '.join(duplicate['drugs'])}\n" +
                            f"\n{'='*15}\n")
                
                # 4. その他の相互作用
                other_interactions = [i for i in drug_info['interactions'] if i.get('risk') not in ['critical', 'high']]
                if other_interactions:
                    for interaction in other_interactions:
                        response_parts.append("🟦【その他の相互作用】")
                        response_parts.append("")
                        response_parts.append(f"対象の薬: {interaction['drug1']}、{interaction['drug2']}")
                        response_parts.append(f"リスク: {interaction.get('description', '相互作用あり')}")
                        if 'mechanism' in interaction:
                            response_parts.append(f"機序: {interaction['mechanism']}")
                        response_parts.append("\n━━━━━━━━━━━━━━\n")
                
                # 5. KEGG情報（利用可能な場合）
                if drug_info['kegg_info']:
                    for kegg in drug_info['kegg_info']:
                        response_parts.append("🟦【KEGG情報】")
                        response_parts.append("")
                        response_parts.append(f"薬剤名: {kegg['drug_name']}")
                        if kegg.get('kegg_id'):
                            response_parts.append(f"KEGG ID: {kegg['kegg_id']}")
                        if kegg.get('pathways'):
                            response_parts.append(f"パスウェイ: {', '.join(kegg['pathways'][:2])}")
                        if kegg.get('targets'):
                            response_parts.append(f"ターゲット: {', '.join(kegg['targets'][:2])}")
                        response_parts.append("\n━━━━━━━━━━━━━━\n")
                
                # 6. 相互作用なしの場合
                if not drug_info['interactions'] and not drug_info['same_effect_warnings']:
                    response_parts.append("🟦【相互作用チェック】")
                    response_parts.append("")
                    response_parts.append("確認された相互作用はありませんでした")
                    response_parts.append("\n━━━━━━━━━━━━━━\n")
                
                # 7. 警告事項
                if drug_info['warnings']:
                    response_parts.append("🟦【警告事項】")
                    for warning in drug_info['warnings']:
                        response_parts.append(f"・{warning}")
                    response_parts.append("\n━━━━━━━━━━━━━━\n")
                
                # 8. 推奨事項
                if drug_info['recommendations']:
                    response_parts.append("🟦【推奨事項】")
                    for recommendation in drug_info['recommendations']:
                        response_parts.append(f"・{recommendation}")
                    response_parts.append("\n━━━━━━━━━━━━━━\n")
                
                # 9. 参考情報の注意書き
                response_parts.append("この結果はあくまで参考情報です。最終的な判断は医師・薬剤師にご相談ください。\n")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"応答メッセージ生成エラー: {e}")
            return "エラーが発生しました。薬剤師にご相談ください。"
    
    def _get_risk_emoji(self, risk_level: str) -> str:
        """リスクレベルに応じた絵文字を返す"""
        risk_emojis = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': '��'
        }
        return risk_emojis.get(risk_level, '⚠️')
    
    def generate_simple_response(self, drug_names: List[str]) -> str:
        """シンプルな応答メッセージを生成（テスト用）"""
        if not drug_names:
            return """🩺【薬剤名未検出】
━━━━━━━━━━━━━━
❌ 薬剤名を読み取ることができませんでした

💡 推奨事項:
・より鮮明な画像で再度お試しください
・文字がはっきり見えるように撮影してください
━━━━━━━━━━━━━━"""
        
        response_parts = []
        response_parts.append("🩺【薬剤名検出完了】")
        response_parts.append("━━━━━━━━━━━━━━")
        response_parts.append("📌 検出薬剤:")
        response_parts.append("")
        for i, drug in enumerate(drug_names, 1):
            response_parts.append(f"① {drug}")
        response_parts.append("")
        response_parts.append("💡 次のステップ:")
        response_parts.append("・「診断」と送信して飲み合わせチェック")
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