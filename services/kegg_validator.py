"""
KEGG検証サービス
薬剤名の存在確認と自動補正機能
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from .kegg_client import KEGGClient
from .drug_normalization_service import DrugNormalizationService

logger = logging.getLogger(__name__)

class KEGGValidator:
    """KEGG検証サービス"""
    
    def __init__(self):
        self.kegg_client = KEGGClient()
        self.normalization_service = DrugNormalizationService()
        
        # 存在しない薬剤名のパターン辞書
        self.non_existent_drugs = {
            "テラムロジン": "テラムロAP",
            "テラムロプリド": "テラムロAP", 
            "テラムロリウム": "テラムロAP",
            "ラベプラゾール": "ボノプラザン",
            "ラベプラゾールナトリウム": "ボノプラザン",
            "タケキャブ": "ボノプラザン",
            "タダラフィルリウム": "タダラフィル",
            "ニコランジルリウム": "ニコランジル",
            "エナラプリルリウム": "エナラプリル",
        }
        
        # 薬剤名の異常パターン
        self.invalid_drug_patterns = [
            r'リウム$',     # 存在しない成分名
            r'プリド$',     # 存在しない成分名（テラムロプリド等）
            r'ジン$',       # 存在しない成分名
            r'プラゾール$', # PPI系の誤認識（ラベプラゾール等）
            r'プラゾールナトリウム$', # PPI系の誤認識
        ]
        
        # 薬剤名の妥当性パターン
        self.valid_drug_patterns = [
            r'^[ア-ン]+[A-Z]*\d*$',  # カタカナ＋アルファベット＋数字
            r'^[ア-ン]+AP$',         # テラムロAP等の配合剤
            r'^[ア-ン]+OD$',         # OD錠
            r'^[ア-ン]+ZA$',         # ZA錠
            r'^[ア-ン]+/バルサルタン$', # 配合剤
            r'^[ア-ン]+/アムロジピン$', # 配合剤
        ]
    
    def validate_drug_name(self, drug_name: str) -> Dict[str, Any]:
        """
        薬剤名の存在確認と自動補正
        
        Args:
            drug_name: 確認する薬剤名
            
        Returns:
            検証結果
        """
        try:
            logger.info(f"Validating drug name: {drug_name}")
            
            # 1. 基本的な妥当性チェック
            validity_result = self._check_basic_validity(drug_name)
            if not validity_result['is_valid']:
                return {
                    'original_name': drug_name,
                    'is_valid': False,
                    'corrected_name': None,
                    'reason': validity_result['reason'],
                    'confidence': 0.0
                }
            
            # 2. 存在しない薬剤名のチェック
            if drug_name in self.non_existent_drugs:
                corrected_name = self.non_existent_drugs[drug_name]
                logger.info(f"Non-existent drug detected: {drug_name} -> {corrected_name}")
                return {
                    'original_name': drug_name,
                    'is_valid': False,
                    'corrected_name': corrected_name,
                    'reason': f"存在しない薬剤名: {drug_name}",
                    'confidence': 0.95
                }
            
            # 3. 異常パターンのチェック
            for pattern in self.invalid_drug_patterns:
                if self._matches_pattern(drug_name, pattern):
                    # 類似薬剤を検索
                    similar_drug = self._find_similar_drug(drug_name)
                    if similar_drug:
                        logger.info(f"Invalid pattern detected: {drug_name} -> {similar_drug}")
                        return {
                            'original_name': drug_name,
                            'is_valid': False,
                            'corrected_name': similar_drug,
                            'reason': f"異常パターン検出: {pattern}",
                            'confidence': 0.9
                        }
            
            # 4. KEGG照合
            kegg_result = self._validate_with_kegg(drug_name)
            if kegg_result['exists']:
                return {
                    'original_name': drug_name,
                    'is_valid': True,
                    'corrected_name': drug_name,
                    'reason': "KEGG照合で存在確認",
                    'confidence': kegg_result['confidence']
                }
            elif kegg_result['corrected_name']:
                return {
                    'original_name': drug_name,
                    'is_valid': False,
                    'corrected_name': kegg_result['corrected_name'],
                    'reason': f"KEGG照合で補正: {kegg_result['reason']}",
                    'confidence': kegg_result['confidence']
                }
            
            # 5. 最終判定
            return {
                'original_name': drug_name,
                'is_valid': True,
                'corrected_name': drug_name,
                'reason': "基本妥当性チェック通過",
                'confidence': 0.7
            }
            
        except Exception as e:
            logger.error(f"Drug validation failed: {e}")
            return {
                'original_name': drug_name,
                'is_valid': False,
                'corrected_name': None,
                'reason': f"検証エラー: {str(e)}",
                'confidence': 0.0
            }
    
    def _check_basic_validity(self, drug_name: str) -> Dict[str, Any]:
        """基本的な妥当性チェック"""
        if not drug_name or len(drug_name.strip()) == 0:
            return {'is_valid': False, 'reason': '空の薬剤名'}
        
        if len(drug_name) < 2:
            return {'is_valid': False, 'reason': '薬剤名が短すぎる'}
        
        # 妥当性パターンのチェック
        for pattern in self.valid_drug_patterns:
            if self._matches_pattern(drug_name, pattern):
                return {'is_valid': True, 'reason': '妥当性パターンに一致'}
        
        return {'is_valid': True, 'reason': '基本チェック通過'}
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """正規表現パターンマッチング"""
        import re
        return bool(re.search(pattern, text))
    
    def _find_similar_drug(self, drug_name: str) -> Optional[str]:
        """類似薬剤の検索"""
        # 部分一致による類似薬剤検索
        for non_existent, corrected in self.non_existent_drugs.items():
            if non_existent in drug_name or drug_name in non_existent:
                return corrected
        
        # 正規化サービスによる類似検索
        try:
            result = self.normalization_service.normalize_drug_name(drug_name)
            if result['confidence'] > 0.8:
                return result['normalized']
        except Exception:
            pass
        
        return None
    
    def _validate_with_kegg(self, drug_name: str) -> Dict[str, Any]:
        """KEGG照合による検証"""
        try:
            # KEGG検索
            kegg_result = self.kegg_client.best_kegg_and_atc(drug_name)
            
            if kegg_result and kegg_result.get('kegg_id'):
                return {
                    'exists': True,
                    'corrected_name': None,
                    'reason': 'KEGG照合で存在確認',
                    'confidence': 0.9
                }
            
            # 類似候補の検索
            similar_candidates = self.kegg_client._apply_similar_candidate_scoring(drug_name)
            if similar_candidates != drug_name:
                similar_result = self.kegg_client.best_kegg_and_atc(similar_candidates)
                if similar_result and similar_result.get('kegg_id'):
                    return {
                        'exists': False,
                        'corrected_name': similar_candidates,
                        'reason': f'類似候補で補正: {similar_candidates}',
                        'confidence': 0.8
                    }
            
            return {
                'exists': False,
                'corrected_name': None,
                'reason': 'KEGG照合で見つからず',
                'confidence': 0.0
            }
            
        except Exception as e:
            logger.error(f"KEGG validation failed: {e}")
            return {
                'exists': False,
                'corrected_name': None,
                'reason': f'KEGG照合エラー: {str(e)}',
                'confidence': 0.0
            }
    
    def batch_validate(self, drug_names: List[str]) -> List[Dict[str, Any]]:
        """複数薬剤名の一括検証"""
        results = []
        for drug_name in drug_names:
            result = self.validate_drug_name(drug_name)
            results.append(result)
        return results
