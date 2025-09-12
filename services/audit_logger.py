"""
監査ログ機能 - 根拠を保存・表示
誤検知ほぼゼロ／取りこぼしほぼゼロで、根拠が示せる診断のための監査機能
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class AuditEntry:
    """監査ログエントリ"""
    timestamp: str
    user_id: str
    action: str
    drug_names: List[str]
    interactions: List[Dict[str, Any]]
    classification_results: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    kegg_ids: Dict[str, str]
    atc_codes: Dict[str, List[str]]
    rule_ids: List[str]
    processing_time_ms: int
    image_quality_score: Optional[float] = None
    ocr_confidence: Optional[float] = None
    ai_confidence: Optional[float] = None

class AuditLogger:
    """監査ログ管理"""
    
    def __init__(self):
        """初期化"""
        self.audit_logs = []
        self.logger = logging.getLogger(f"{__name__}.audit")
    
    def log_drug_analysis(self, 
                         user_id: str,
                         drug_names: List[str],
                         interactions: List[Dict[str, Any]],
                         classification_results: List[Dict[str, Any]],
                         confidence_scores: Dict[str, float],
                         kegg_ids: Dict[str, str],
                         atc_codes: Dict[str, List[str]],
                         rule_ids: List[str],
                         processing_time_ms: int,
                         image_quality_score: Optional[float] = None,
                         ocr_confidence: Optional[float] = None,
                         ai_confidence: Optional[float] = None) -> str:
        """
        薬剤分析の監査ログを記録
        
        Returns:
            audit_id: 監査ログのID
        """
        try:
            audit_entry = AuditEntry(
                timestamp=datetime.now().isoformat(),
                user_id=user_id,
                action="drug_analysis",
                drug_names=drug_names,
                interactions=interactions,
                classification_results=classification_results,
                confidence_scores=confidence_scores,
                kegg_ids=kegg_ids,
                atc_codes=atc_codes,
                rule_ids=rule_ids,
                processing_time_ms=processing_time_ms,
                image_quality_score=image_quality_score,
                ocr_confidence=ocr_confidence,
                ai_confidence=ai_confidence
            )
            
            audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
            self.audit_logs.append((audit_id, audit_entry))
            
            # ログ出力
            self.logger.info(f"Audit log created: {audit_id}")
            self.logger.info(f"Drugs: {len(drug_names)}, Interactions: {len(interactions)}, Rules: {len(rule_ids)}")
            
            return audit_id
            
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_audit_summary(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """監査ログのサマリーを取得"""
        for id, entry in self.audit_logs:
            if id == audit_id:
                return {
                    'audit_id': audit_id,
                    'timestamp': entry.timestamp,
                    'user_id': entry.user_id,
                    'drug_count': len(entry.drug_names),
                    'interaction_count': len(entry.interactions),
                    'rule_count': len(entry.rule_ids),
                    'processing_time_ms': entry.processing_time_ms,
                    'confidence_avg': sum(entry.confidence_scores.values()) / len(entry.confidence_scores) if entry.confidence_scores else 0,
                    'image_quality': entry.image_quality_score,
                    'ocr_confidence': entry.ocr_confidence,
                    'ai_confidence': entry.ai_confidence
                }
        return None
    
    def get_audit_details(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """監査ログの詳細を取得"""
        for id, entry in self.audit_logs:
            if id == audit_id:
                return asdict(entry)
        return None
    
    def generate_audit_report(self, audit_id: str) -> str:
        """監査レポートを生成"""
        details = self.get_audit_details(audit_id)
        if not details:
            return f"監査ログが見つかりません: {audit_id}"
        
        report_parts = []
        report_parts.append("🔍 【監査レポート】")
        report_parts.append("━━━━━━━━━━━━━━")
        report_parts.append(f"監査ID: {audit_id}")
        report_parts.append(f"日時: {details['timestamp']}")
        report_parts.append(f"ユーザー: {details['user_id']}")
        report_parts.append("")
        
        # 薬剤情報
        report_parts.append("📋 検出薬剤:")
        for i, drug in enumerate(details['drug_names'], 1):
            kegg_id = details['kegg_ids'].get(drug, 'N/A')
            atc_codes = details['atc_codes'].get(drug, [])
            confidence = details['confidence_scores'].get(drug, 0.0)
            report_parts.append(f"{i:02d}) {drug}")
            report_parts.append(f"    KEGG ID: {kegg_id}")
            report_parts.append(f"    ATC: {', '.join(atc_codes) if atc_codes else 'N/A'}")
            report_parts.append(f"    信頼度: {confidence:.2f}")
            report_parts.append("")
        
        # 相互作用情報
        if details['interactions']:
            report_parts.append("⚠️ 相互作用:")
            for interaction in details['interactions']:
                report_parts.append(f"• {interaction.get('description', 'N/A')}")
                report_parts.append(f"  リスク: {interaction.get('risk', 'N/A')}")
                report_parts.append(f"  機序: {interaction.get('mechanism', 'N/A')}")
                report_parts.append(f"  臨床的影響: {interaction.get('clinical_impact', 'N/A')}")
                report_parts.append("")
        
        # 適用ルール
        if details['rule_ids']:
            report_parts.append("📜 適用ルール:")
            for rule_id in details['rule_ids']:
                report_parts.append(f"• {rule_id}")
            report_parts.append("")
        
        # 処理統計
        report_parts.append("📊 処理統計:")
        report_parts.append(f"処理時間: {details['processing_time_ms']}ms")
        report_parts.append(f"画像品質: {details['image_quality_score']:.2f}" if details['image_quality_score'] else "画像品質: N/A")
        report_parts.append(f"OCR信頼度: {details['ocr_confidence']:.2f}" if details['ocr_confidence'] else "OCR信頼度: N/A")
        report_parts.append(f"AI信頼度: {details['ai_confidence']:.2f}" if details['ai_confidence'] else "AI信頼度: N/A")
        
        return "\n".join(report_parts)
    
    def get_low_confidence_drugs(self, audit_id: str, threshold: float = 0.8) -> List[str]:
        """低信頼度の薬剤を取得"""
        details = self.get_audit_details(audit_id)
        if not details:
            return []
        
        low_confidence = []
        for drug, confidence in details['confidence_scores'].items():
            if confidence < threshold:
                low_confidence.append(drug)
        
        return low_confidence
    
    def get_missing_kegg_drugs(self, audit_id: str) -> List[str]:
        """KEGG IDが取得できなかった薬剤を取得"""
        details = self.get_audit_details(audit_id)
        if not details:
            return []
        
        missing_kegg = []
        for drug in details['drug_names']:
            if drug not in details['kegg_ids'] or details['kegg_ids'][drug] == 'N/A':
                missing_kegg.append(drug)
        
        return missing_kegg
    
    def export_audit_logs(self, format: str = 'json') -> str:
        """監査ログをエクスポート"""
        if format == 'json':
            return json.dumps([asdict(entry) for _, entry in self.audit_logs], ensure_ascii=False, indent=2)
        else:
            return str(self.audit_logs)
    
    def clear_old_logs(self, days: int = 30):
        """古いログをクリア"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        original_count = len(self.audit_logs)
        self.audit_logs = [
            (audit_id, entry) for audit_id, entry in self.audit_logs
            if datetime.fromisoformat(entry.timestamp).timestamp() > cutoff_date
        ]
        
        removed_count = original_count - len(self.audit_logs)
        logger.info(f"Cleared {removed_count} old audit logs (older than {days} days)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """監査ログ統計を取得"""
        if not self.audit_logs:
            return {'total_logs': 0}
        
        total_logs = len(self.audit_logs)
        total_drugs = sum(len(entry.drug_names) for _, entry in self.audit_logs)
        total_interactions = sum(len(entry.interactions) for _, entry in self.audit_logs)
        
        # 信頼度統計
        all_confidences = []
        for _, entry in self.audit_logs:
            all_confidences.extend(entry.confidence_scores.values())
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        return {
            'total_logs': total_logs,
            'total_drugs': total_drugs,
            'total_interactions': total_interactions,
            'average_confidence': avg_confidence,
            'average_drugs_per_analysis': total_drugs / total_logs if total_logs > 0 else 0,
            'average_interactions_per_analysis': total_interactions / total_logs if total_logs > 0 else 0
        }
