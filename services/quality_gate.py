"""
品質ゲートシステム
自動やり直し＆人確認フォールバック
"""
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

class QualityGate:
    """品質ゲートシステム"""
    
    def __init__(self):
        # 品質閾値
        self.thresholds = {
            'extraction_coverage': 0.98,  # 抽出カバレッジ
            'kegg_cover_rate': 0.90,      # KEGG分類カバレッジ
            'confidence_threshold': 0.8,  # 信頼度閾値
            'min_drug_count': 1,          # 最小薬剤数
            'max_drug_count': 20,         # 最大薬剤数（異常値検出）
        }
    
    def check_quality(self, stats: Dict[str, Any], drugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """品質チェックを実行"""
        try:
            quality_result = {
                'passed': True,
                'issues': [],
                'retry_needed': False,
                'human_review_needed': False,
                'stats': stats
            }
            
            # 1. 抽出カバレッジチェック
            coverage = stats.get('coverage', 0.0)
            if coverage < self.thresholds['extraction_coverage']:
                quality_result['issues'].append({
                    'type': 'low_coverage',
                    'message': f'抽出カバレッジが低い: {coverage:.2f} < {self.thresholds["extraction_coverage"]}',
                    'severity': 'high'
                })
                quality_result['retry_needed'] = True
            
            # 2. KEGG分類カバレッジチェック
            kegg_coverage = stats.get('kegg_cover_rate', 0.0)
            if kegg_coverage < self.thresholds['kegg_cover_rate']:
                quality_result['issues'].append({
                    'type': 'low_kegg_coverage',
                    'message': f'KEGG分類カバレッジが低い: {kegg_coverage:.2f} < {self.thresholds["kegg_cover_rate"]}',
                    'severity': 'medium'
                })
                quality_result['retry_needed'] = True
            
            # 3. 信頼度チェック
            low_conf_drugs = [d for d in drugs if d.get('confidence', 0) < self.thresholds['confidence_threshold']]
            if low_conf_drugs:
                quality_result['issues'].append({
                    'type': 'low_confidence',
                    'message': f'低信頼度薬剤: {len(low_conf_drugs)}件',
                    'severity': 'medium',
                    'details': [d.get('generic', d.get('raw', '')) for d in low_conf_drugs]
                })
                quality_result['human_review_needed'] = True
            
            # 4. 薬剤数チェック
            drug_count = len(drugs)
            if drug_count < self.thresholds['min_drug_count']:
                quality_result['issues'].append({
                    'type': 'too_few_drugs',
                    'message': f'薬剤数が少なすぎる: {drug_count} < {self.thresholds["min_drug_count"]}',
                    'severity': 'high'
                })
                quality_result['retry_needed'] = True
            
            if drug_count > self.thresholds['max_drug_count']:
                quality_result['issues'].append({
                    'type': 'too_many_drugs',
                    'message': f'薬剤数が多すぎる: {drug_count} > {self.thresholds["max_drug_count"]}',
                    'severity': 'medium'
                })
                quality_result['human_review_needed'] = True
            
            # 5. KEGG ID未設定チェック
            no_kegg_drugs = [d for d in drugs if not d.get('kegg_id')]
            if no_kegg_drugs:
                quality_result['issues'].append({
                    'type': 'missing_kegg_id',
                    'message': f'KEGG ID未設定薬剤: {len(no_kegg_drugs)}件',
                    'severity': 'low',
                    'details': [d.get('generic', d.get('raw', '')) for d in no_kegg_drugs]
                })
            
            # 総合判定
            high_severity_issues = [i for i in quality_result['issues'] if i['severity'] == 'high']
            if high_severity_issues:
                quality_result['passed'] = False
            
            logger.info(f"Quality gate result: passed={quality_result['passed']}, issues={len(quality_result['issues'])}")
            return quality_result
            
        except Exception as e:
            logger.error(f"Quality gate check failed: {e}")
            return {
                'passed': False,
                'issues': [{'type': 'error', 'message': str(e), 'severity': 'high'}],
                'retry_needed': True,
                'human_review_needed': True,
                'stats': stats
            }
    
    def should_retry(self, quality_result: Dict[str, Any]) -> bool:
        """再試行が必要かどうかを判定"""
        return quality_result.get('retry_needed', False)
    
    def needs_human_review(self, quality_result: Dict[str, Any]) -> bool:
        """人による確認が必要かどうかを判定"""
        return quality_result.get('human_review_needed', False)
    
    def get_retry_strategy(self, quality_result: Dict[str, Any]) -> Dict[str, Any]:
        """再試行戦略を取得"""
        retry_strategy = {
            'retry_extraction': False,
            'retry_classification': False,
            'retry_kegg_lookup': False,
            'use_alternative_methods': False
        }
        
        for issue in quality_result.get('issues', []):
            issue_type = issue.get('type')
            
            if issue_type == 'low_coverage':
                retry_strategy['retry_extraction'] = True
                retry_strategy['use_alternative_methods'] = True
            
            elif issue_type == 'low_kegg_coverage':
                retry_strategy['retry_classification'] = True
                retry_strategy['retry_kegg_lookup'] = True
            
            elif issue_type == 'low_confidence':
                retry_strategy['retry_extraction'] = True
                retry_strategy['use_alternative_methods'] = True
        
        return retry_strategy
    
    def generate_quality_report(self, quality_result: Dict[str, Any]) -> str:
        """品質レポートを生成"""
        if quality_result.get('passed'):
            return "✅ 品質チェック: 合格"
        
        report_parts = ["⚠️ 品質チェック: 要改善"]
        
        for issue in quality_result.get('issues', []):
            severity_emoji = {
                'high': '🚨',
                'medium': '⚠️',
                'low': 'ℹ️'
            }.get(issue.get('severity', 'low'), 'ℹ️')
            
            report_parts.append(f"{severity_emoji} {issue.get('message', '')}")
            
            if issue.get('details'):
                details = issue['details'][:3]  # 最初の3件のみ表示
                report_parts.append(f"   対象: {', '.join(details)}")
        
        return "\n".join(report_parts)
    
    def create_human_review_message(self, quality_result: Dict[str, Any], drugs: List[Dict[str, Any]]) -> str:
        """人による確認用メッセージを生成"""
        message_parts = [
            "🔍 薬剤検出結果の確認をお願いします",
            "━━━━━━━━━━━━━━"
        ]
        
        # 検出された薬剤を表示
        message_parts.append(f"📋 検出された薬剤 ({len(drugs)}件):")
        for i, drug in enumerate(drugs, 1):
            name = drug.get('generic', drug.get('raw', ''))
            confidence = drug.get('confidence', 0)
            kegg_id = drug.get('kegg_id', '未設定')
            
            confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.5 else "🔴"
            kegg_emoji = "✅" if kegg_id != '未設定' else "❌"
            
            message_parts.append(f"{i:02d}) {name}")
            message_parts.append(f"    信頼度: {confidence_emoji} {confidence:.2f}")
            message_parts.append(f"    KEGG ID: {kegg_emoji} {kegg_id}")
            message_parts.append("")
        
        # 品質問題を表示
        if quality_result.get('issues'):
            message_parts.append("⚠️ 検出された問題:")
            for issue in quality_result['issues']:
                message_parts.append(f"• {issue.get('message', '')}")
        
        message_parts.extend([
            "",
            "💡 確認事項:",
            "• 薬剤名は正確ですか？",
            "• 抜けている薬剤はありませんか？",
            "• 不要な薬剤は含まれていませんか？",
            "",
            "修正が必要な場合は「修正: 薬剤名」の形式でお知らせください。"
        ])
        
        return "\n".join(message_parts)
