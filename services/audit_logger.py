"""
監査ログシステム
kegg_id/atc/class_jp/tags/rule_hits を必ずログ出力・保存
"""
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AuditLogger:
    """監査ログシステム"""
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 監査ログ用の専用ロガー
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # ファイルハンドラーを設定
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # フォーマッターを設定
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # ハンドラーを追加
        if not self.audit_logger.handlers:
            self.audit_logger.addHandler(file_handler)
    
    def log_drug_processing(self, 
                          session_id: str,
                          drugs: List[Dict[str, Any]], 
                          interaction_result: Dict[str, Any],
                          quality_result: Dict[str, Any],
                          processing_stats: Dict[str, Any]) -> None:
        """薬剤処理の監査ログを記録"""
        try:
            audit_data = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'drugs': self._extract_drug_audit_data(drugs),
                'interactions': self._extract_interaction_audit_data(interaction_result),
                'quality': quality_result,
                'stats': processing_stats
            }
            
            # 構造化ログとして記録
            self.audit_logger.info(f"DRUG_PROCESSING: {json.dumps(audit_data, ensure_ascii=False)}")
            
            # 詳細ログファイルにも保存
            self._save_detailed_log(session_id, audit_data)
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
    
    def _extract_drug_audit_data(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """薬剤データから監査用データを抽出"""
        audit_drugs = []
        
        for drug in drugs:
            audit_drug = {
                'generic': drug.get('generic', ''),
                'brand': drug.get('brand', ''),
                'raw': drug.get('raw', ''),
                'kegg_id': drug.get('kegg_id', ''),
                'atc_codes': drug.get('atc', []),
                'class_jp': drug.get('class_jp', ''),
                'confidence': drug.get('confidence', 0.0),
                'interaction_tags': drug.get('interaction_tags', []),
                'route': drug.get('route', ''),
                'strength': drug.get('strength', ''),
                'dose': drug.get('dose', ''),
                'frequency': drug.get('frequency', '')
            }
            audit_drugs.append(audit_drug)
        
        return audit_drugs
    
    def _extract_interaction_audit_data(self, interaction_result: Dict[str, Any]) -> Dict[str, Any]:
        """相互作用結果から監査用データを抽出"""
        return {
            'has_interactions': interaction_result.get('has_interactions', False),
            'major_count': len(interaction_result.get('major_interactions', [])),
            'moderate_count': len(interaction_result.get('moderate_interactions', [])),
            'rule_hits': [
                {
                    'id': rule.get('id', ''),
                    'name': rule.get('name', ''),
                    'severity': rule.get('severity', ''),
                    'matched_drugs': rule.get('matched_drugs', [])
                }
                for rule in (interaction_result.get('major_interactions', []) + 
                           interaction_result.get('moderate_interactions', []))
            ]
        }
    
    def _save_detailed_log(self, session_id: str, audit_data: Dict[str, Any]) -> None:
        """詳細ログをファイルに保存"""
        try:
            log_file = self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save detailed log: {e}")
    
    def log_quality_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """品質メトリクスをログ記録"""
        try:
            quality_log = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'metrics': metrics
            }
            
            self.audit_logger.info(f"QUALITY_METRICS: {json.dumps(quality_log, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"Quality metrics logging failed: {e}")
    
    def log_error(self, session_id: str, error_type: str, error_message: str, context: Dict[str, Any] = None) -> None:
        """エラーログを記録"""
        try:
            error_log = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'error_type': error_type,
                'error_message': error_message,
                'context': context or {}
            }
            
            self.audit_logger.error(f"ERROR: {json.dumps(error_log, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"Error logging failed: {e}")
    
    def get_audit_summary(self, date: str = None) -> Dict[str, Any]:
        """監査サマリーを取得"""
        try:
            if not date:
                date = datetime.now().strftime('%Y%m%d')
            
            log_file = self.log_dir / f"audit_{date}.log"
            
            if not log_file.exists():
                return {'error': 'Log file not found'}
            
            # ログファイルを解析してサマリーを生成
            summary = {
                'date': date,
                'total_sessions': 0,
                'total_drugs': 0,
                'total_interactions': 0,
                'quality_issues': 0,
                'errors': 0
            }
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'DRUG_PROCESSING:' in line:
                        summary['total_sessions'] += 1
                        # 簡単な解析（実際の実装ではより詳細な解析が必要）
                        if '"drugs":' in line:
                            summary['total_drugs'] += line.count('"generic":')
                        if '"has_interactions": true' in line:
                            summary['total_interactions'] += 1
                    elif 'ERROR:' in line:
                        summary['errors'] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get audit summary: {e}")
            return {'error': str(e)}
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """古いログファイルをクリーンアップ"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            
            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
            
            for log_file in self.log_dir.glob("*.json"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
                    
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")