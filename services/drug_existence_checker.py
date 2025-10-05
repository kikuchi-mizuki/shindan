"""
薬剤存在確認サービス
KEGG/PMDA/JANでの薬名存在チェックと自動補正
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from .kegg_client import KEGGClient
from .drug_normalization_service import DrugNormalizationService

logger = logging.getLogger(__name__)

class DrugExistenceChecker:
    """薬剤存在確認サービス"""
    
    def __init__(self):
        self.kegg_client = KEGGClient()
        self.normalization_service = DrugNormalizationService()
        
        # 存在しない薬剤名のパターン辞書（最終版）
        self.non_existent_drugs = {
            "テラムロジン": "テラムロAP",
            "テラムロプリド": "テラムロAP", 
            "テラムロリウム": "テラムロAP",
            "ラベプラゾール": "ボノプラザン",  # PPI→P-CABの誤認識
            "ラベプラゾールナトリウム": "ボノプラザン",
            "タケキャブ": "ボノプラザン",  # 商品名→一般名
            "タダラフィルリウム": "タダラフィル",
            "ニコランジルリウム": "ニコランジル",
            "エナラプリルリウム": "エナラプリル",
        }
        
        # 薬剤名の妥当性パターン
        self.valid_drug_patterns = [
            r'^[ア-ン]+[A-Z]*\d*$',  # カタカナ＋アルファベット＋数字
            r'^[ア-ン]+AP$',         # テラムロAP等の配合剤
            r'^[ア-ン]+OD$',         # OD錠
            r'^[ア-ン]+ZA$',         # ZA錠
            r'^[ア-ン]+/バルサルタン$', # 配合剤
            r'^[ア-ン]+/アムロジピン$', # 配合剤
        ]
        
        # 薬剤名の異常パターン（強化版）
        self.invalid_drug_patterns = [
            r'リウム$',     # 存在しない成分名
            r'プリド$',     # 存在しない成分名（テラムロプリド等）
            r'ジン$',       # 存在しない成分名
            r'プラゾール$', # PPI系の誤認識（ラベプラゾール等）
            r'プラゾールナトリウム$', # PPI系の誤認識（ラベプラゾールナトリウム等）
        ]
    
    def check_drug_existence(self, drug_name: str, manufacturer: str = None) -> Dict[str, Any]:
        """
        薬剤名の存在確認と自動補正
        
        Args:
            drug_name: 確認する薬剤名
            manufacturer: メーカー名（オプション）
            
        Returns:
            存在確認結果
        """
        try:
            logger.info(f"Checking drug existence: {drug_name}")
            
            # 1. 基本的な妥当性チェック
            validity_result = self._check_basic_validity(drug_name)
            
            # 2. 存在しない薬剤名パターンチェック
            correction_result = self._check_non_existent_patterns(drug_name)
            
            # 3. KEGGデータベース照合
            kegg_result = self._check_kegg_existence(drug_name)
            
            # 4. メーカー名との整合性チェック
            manufacturer_result = self._check_manufacturer_consistency(drug_name, manufacturer)
            
            # 5. 結果統合
            final_result = self._integrate_existence_results(
                drug_name, validity_result, correction_result, kegg_result, manufacturer_result
            )
            
            logger.info(f"Existence check result: {final_result['exists']} (confidence: {final_result['confidence']:.2f})")
            return final_result
            
        except Exception as e:
            logger.error(f"Drug existence check failed: {e}")
            return {
                'original_name': drug_name,
                'exists': False,
                'confidence': 0.0,
                'corrected_name': None,
                'correction_reason': 'エラー',
                'error': str(e)
            }
    
    def _check_basic_validity(self, drug_name: str) -> Dict[str, Any]:
        """基本的な妥当性チェック"""
        import re
        
        # 異常パターンチェック
        for pattern in self.invalid_drug_patterns:
            if re.search(pattern, drug_name):
                return {
                    'valid': False,
                    'reason': f'異常パターン検出: {pattern}',
                    'confidence': 0.9
                }
        
        # 妥当パターンチェック
        for pattern in self.valid_drug_patterns:
            if re.match(pattern, drug_name):
                return {
                    'valid': True,
                    'reason': f'妥当パターン: {pattern}',
                    'confidence': 0.8
                }
        
        # デフォルト：妥当とみなす
        return {
            'valid': True,
            'reason': '基本的な妥当性',
            'confidence': 0.5
        }
    
    def _check_non_existent_patterns(self, drug_name: str) -> Dict[str, Any]:
        """存在しない薬剤名パターンチェック"""
        if drug_name in self.non_existent_drugs:
            corrected_name = self.non_existent_drugs[drug_name]
            return {
                'exists': False,
                'corrected_name': corrected_name,
                'reason': '存在しない薬剤名パターン',
                'confidence': 0.95
            }
        
        return {
            'exists': True,
            'corrected_name': None,
            'reason': '存在しない薬剤名パターンなし',
            'confidence': 0.7
        }
    
    def _check_kegg_existence(self, drug_name: str) -> Dict[str, Any]:
        """KEGGデータベースでの存在確認"""
        try:
            kegg_result = self.kegg_client.best_kegg_and_atc(drug_name)
            
            if kegg_result and kegg_result.get('kegg_id'):
                return {
                    'exists': True,
                    'kegg_id': kegg_result['kegg_id'],
                    'atc_codes': kegg_result.get('atc', []),
                    'reason': 'KEGGデータベースに存在',
                    'confidence': 0.9
                }
            else:
                return {
                    'exists': False,
                    'kegg_id': None,
                    'atc_codes': [],
                    'reason': 'KEGGデータベースに存在しない',
                    'confidence': 0.8
                }
                
        except Exception as e:
            logger.warning(f"KEGG existence check failed: {e}")
            return {
                'exists': None,
                'kegg_id': None,
                'atc_codes': [],
                'reason': 'KEGG照合エラー',
                'confidence': 0.3
            }
    
    def _check_manufacturer_consistency(self, drug_name: str, manufacturer: str = None) -> Dict[str, Any]:
        """メーカー名との整合性チェック"""
        if not manufacturer:
            return {
                'consistent': None,
                'reason': 'メーカー名なし',
                'confidence': 0.5
            }
        
        try:
            # メーカー名から推測される薬剤を取得
            manufacturer_hints = self.normalization_service.get_manufacturer_hints(drug_name, manufacturer)
            
            if manufacturer_hints:
                best_hint = max(manufacturer_hints, key=lambda x: x['score'])
                if best_hint['score'] >= 0.8:
                    return {
                        'consistent': True,
                        'reason': f'{manufacturer}の主要薬剤と一致',
                        'confidence': best_hint['score']
                    }
            
            return {
                'consistent': False,
                'reason': f'{manufacturer}の主要薬剤と不一致',
                'confidence': 0.6
            }
            
        except Exception as e:
            logger.warning(f"Manufacturer consistency check failed: {e}")
            return {
                'consistent': None,
                'reason': 'メーカー整合性チェックエラー',
                'confidence': 0.3
            }
    
    def _integrate_existence_results(self, 
                                   drug_name: str,
                                   validity_result: Dict[str, Any],
                                   correction_result: Dict[str, Any],
                                   kegg_result: Dict[str, Any],
                                   manufacturer_result: Dict[str, Any]) -> Dict[str, Any]:
        """存在確認結果の統合"""
        
        # 存在しない薬剤名パターンの場合は即座に補正
        if not correction_result['exists']:
            return {
                'original_name': drug_name,
                'exists': False,
                'confidence': correction_result['confidence'],
                'corrected_name': correction_result['corrected_name'],
                'correction_reason': correction_result['reason'],
                'kegg_id': None,
                'atc_codes': [],
                'manufacturer_consistent': None
            }
        
        # 基本的な妥当性チェック
        if not validity_result['valid']:
            return {
                'original_name': drug_name,
                'exists': False,
                'confidence': validity_result['confidence'],
                'corrected_name': None,
                'correction_reason': validity_result['reason'],
                'kegg_id': None,
                'atc_codes': [],
                'manufacturer_consistent': None
            }
        
        # KEGGデータベースでの存在確認
        if kegg_result['exists'] is False:
            return {
                'original_name': drug_name,
                'exists': False,
                'confidence': kegg_result['confidence'],
                'corrected_name': None,
                'correction_reason': kegg_result['reason'],
                'kegg_id': None,
                'atc_codes': [],
                'manufacturer_consistent': manufacturer_result['consistent']
            }
        
        # メーカー名との整合性
        if manufacturer_result['consistent'] is False:
            return {
                'original_name': drug_name,
                'exists': False,
                'confidence': manufacturer_result['confidence'],
                'corrected_name': None,
                'correction_reason': manufacturer_result['reason'],
                'kegg_id': kegg_result.get('kegg_id'),
                'atc_codes': kegg_result.get('atc_codes', []),
                'manufacturer_consistent': False
            }
        
        # 全てのチェックをパス
        return {
            'original_name': drug_name,
            'exists': True,
            'confidence': min(
                validity_result['confidence'],
                correction_result['confidence'],
                kegg_result['confidence'],
                manufacturer_result['confidence']
            ),
            'corrected_name': None,
            'correction_reason': '全てのチェックをパス',
            'kegg_id': kegg_result.get('kegg_id'),
            'atc_codes': kegg_result.get('atc_codes', []),
            'manufacturer_consistent': manufacturer_result['consistent']
        }
    
    def batch_check_drugs(self, drug_names: List[str], manufacturers: List[str] = None) -> List[Dict[str, Any]]:
        """複数薬剤の一括存在確認"""
        results = []
        
        for i, drug_name in enumerate(drug_names):
            manufacturer = manufacturers[i] if manufacturers and i < len(manufacturers) else None
            result = self.check_drug_existence(drug_name, manufacturer)
            results.append(result)
        
        return results
    
    def get_correction_suggestions(self, drug_name: str) -> List[Dict[str, Any]]:
        """薬剤名の補正候補を取得"""
        suggestions = []
        
        # 存在しない薬剤名パターンからの補正
        if drug_name in self.non_existent_drugs:
            suggestions.append({
                'original': drug_name,
                'corrected': self.non_existent_drugs[drug_name],
                'reason': '存在しない薬剤名パターン',
                'confidence': 0.95
            })
        
        # 類似候補からの補正
        similar_candidates = self.normalization_service._get_similar_candidates(drug_name)
        for candidate in similar_candidates:
            suggestions.append({
                'original': drug_name,
                'corrected': candidate['name'],
                'reason': candidate['reason'],
                'confidence': candidate['score']
            })
        
        # 信頼度でソート
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions
    
    def validate_drug_list(self, drug_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """薬剤リストの妥当性検証"""
        total_drugs = len(drug_list)
        valid_drugs = 0
        corrected_drugs = 0
        invalid_drugs = 0
        
        for drug in drug_list:
            drug_name = drug.get('name', '')
            manufacturer = drug.get('manufacturer')
            
            result = self.check_drug_existence(drug_name, manufacturer)
            
            if result['exists']:
                valid_drugs += 1
            elif result['corrected_name']:
                corrected_drugs += 1
            else:
                invalid_drugs += 1
        
        return {
            'total_drugs': total_drugs,
            'valid_drugs': valid_drugs,
            'corrected_drugs': corrected_drugs,
            'invalid_drugs': invalid_drugs,
            'validity_rate': valid_drugs / total_drugs if total_drugs > 0 else 0.0,
            'correction_rate': corrected_drugs / total_drugs if total_drugs > 0 else 0.0
        }
