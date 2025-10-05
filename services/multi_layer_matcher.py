"""
多層辞書照合サービス
KEGG＋JAN＋PMDA＋メーカー名逆引きによる高精度マッチング
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from .kegg_client import KEGGClient
from .drug_normalization_service import DrugNormalizationService

logger = logging.getLogger(__name__)

class MultiLayerMatcher:
    """多層辞書照合サービス"""
    
    def __init__(self):
        self.kegg_client = KEGGClient()
        self.normalization_service = DrugNormalizationService()
        
        # JANコード辞書（実装例）
        self.jan_dictionary = {
            "4901085123456": {"name": "タダラフィル", "manufacturer": "サンド"},
            "4901085123467": {"name": "ニコランジル", "manufacturer": "トーワ"},
            "4901085123478": {"name": "エンレスト", "manufacturer": "ノバルティス"},
            "4901085123489": {"name": "テラムロAP", "manufacturer": "サワイ"},
            "4901085123490": {"name": "エナラプリル", "manufacturer": "トーワ"},
            "4901085123491": {"name": "タケキャブ", "manufacturer": "武田"},
            "4901085123492": {"name": "ランソプラゾール", "manufacturer": "トーワ"},
        }
        
        # PMDAデータベース辞書（実装例）
        self.pmda_dictionary = {
            "タダラフィル": {
                "generic_name": "タダラフィル",
                "brand_names": ["シアリス"],
                "manufacturer": "サンド",
                "atc_code": "G04BE08",
                "category": "PDE5阻害薬"
            },
            "ニコランジル": {
                "generic_name": "ニコランジル",
                "brand_names": ["シグマート"],
                "manufacturer": "トーワ",
                "atc_code": "C01DX16",
                "category": "冠血管拡張薬"
            },
            "エンレスト": {
                "generic_name": "サクビトリル/バルサルタン",
                "brand_names": ["エンレスト"],
                "manufacturer": "ノバルティス",
                "atc_code": "C09DX04",
                "category": "ARNI"
            },
            "テラムロAP": {
                "generic_name": "テルミサルタン/アムロジピン",
                "brand_names": ["テラムロAP"],
                "manufacturer": "サワイ",
                "atc_code": "C09DB07",
                "category": "降圧薬（ARB＋Ca拮抗薬）"
            },
            "エナラプリル": {
                "generic_name": "エナラプリル",
                "brand_names": ["レニベース"],
                "manufacturer": "トーワ",
                "atc_code": "C09AA02",
                "category": "ACE阻害薬"
            },
            "タケキャブ": {
                "generic_name": "ボノプラザン",
                "brand_names": ["タケキャブ"],
                "manufacturer": "武田",
                "atc_code": "A02BC06",
                "category": "P-CAB"
            },
            "ランソプラゾール": {
                "generic_name": "ランソプラゾール",
                "brand_names": ["タケプロン"],
                "manufacturer": "トーワ",
                "atc_code": "A02BC03",
                "category": "PPI"
            }
        }
        
        # メーカー名マッピング
        self.manufacturer_mapping = {
            "サンド": ["タダラフィル", "シアリス"],
            "トーワ": ["ニコランジル", "エナラプリル", "ランソプラゾール"],
            "サワイ": ["テラムロAP", "テルミサルタン", "アムロジピン"],
            "武田": ["タケキャブ", "ボノプラザン"],
            "ノバルティス": ["エンレスト", "サクビトリル", "バルサルタン"]
        }
    
    def match_drug(self, ocr_text: str, manufacturer: str = None, dose: str = None) -> Dict[str, Any]:
        """
        多層辞書照合による薬剤名マッチング
        
        Args:
            ocr_text: OCRで抽出されたテキスト
            manufacturer: メーカー名（オプション）
            dose: 用量（オプション）
            
        Returns:
            マッチング結果
        """
        try:
            logger.info(f"Multi-layer matching for: {ocr_text}")
            
            # 1. 正規化
            normalized_text = self.normalization_service.fix_ocr_aliases(ocr_text)
            
            # 2. 各層での照合
            matches = {
                'kegg': self._match_kegg(normalized_text),
                'pmda': self._match_pmda(normalized_text),
                'jan': self._match_jan(normalized_text),
                'manufacturer': self._match_manufacturer(normalized_text, manufacturer)
            }
            
            # 3. 統合スコアリング
            best_match = self._integrate_matches(matches, manufacturer, dose)
            
            # 4. 信頼度評価
            confidence = self._calculate_confidence(best_match, matches)
            
            result = {
                'original_text': ocr_text,
                'normalized_text': normalized_text,
                'best_match': best_match,
                'confidence': confidence,
                'all_matches': matches,
                'manufacturer_hint': manufacturer,
                'dose_hint': dose
            }
            
            logger.info(f"Match result: {best_match['name']} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Multi-layer matching failed: {e}")
            return {
                'original_text': ocr_text,
                'normalized_text': ocr_text,
                'best_match': None,
                'confidence': 0.0,
                'all_matches': {},
                'error': str(e)
            }
    
    def _match_kegg(self, text: str) -> Optional[Dict[str, Any]]:
        """KEGGデータベース照合"""
        try:
            kegg_result = self.kegg_client.best_kegg_and_atc(text)
            if kegg_result:
                return {
                    'name': text,
                    'kegg_id': kegg_result.get('kegg_id'),
                    'atc_codes': kegg_result.get('atc', []),
                    'label': kegg_result.get('label'),
                    'corrected_name': kegg_result.get('corrected_name'),
                    'score': 0.8
                }
            return None
        except Exception as e:
            logger.warning(f"KEGG matching failed: {e}")
            return None
    
    def _match_pmda(self, text: str) -> Optional[Dict[str, Any]]:
        """PMDAデータベース照合"""
        try:
            # 直接マッチ
            if text in self.pmda_dictionary:
                pmda_info = self.pmda_dictionary[text]
                return {
                    'name': text,
                    'generic_name': pmda_info['generic_name'],
                    'brand_names': pmda_info['brand_names'],
                    'manufacturer': pmda_info['manufacturer'],
                    'atc_code': pmda_info['atc_code'],
                    'category': pmda_info['category'],
                    'score': 0.9
                }
            
            # 部分マッチ（商品名検索）
            for drug_name, pmda_info in self.pmda_dictionary.items():
                for brand_name in pmda_info['brand_names']:
                    if brand_name in text or text in brand_name:
                        return {
                            'name': drug_name,
                            'generic_name': pmda_info['generic_name'],
                            'brand_names': pmda_info['brand_names'],
                            'manufacturer': pmda_info['manufacturer'],
                            'atc_code': pmda_info['atc_code'],
                            'category': pmda_info['category'],
                            'score': 0.7
                        }
            
            return None
        except Exception as e:
            logger.warning(f"PMDA matching failed: {e}")
            return None
    
    def _match_jan(self, text: str) -> Optional[Dict[str, Any]]:
        """JANコード辞書照合"""
        try:
            # JANコード辞書から検索
            for jan_code, jan_info in self.jan_dictionary.items():
                if jan_info['name'] in text or text in jan_info['name']:
                    return {
                        'name': jan_info['name'],
                        'manufacturer': jan_info['manufacturer'],
                        'jan_code': jan_code,
                        'score': 0.6
                    }
            return None
        except Exception as e:
            logger.warning(f"JAN matching failed: {e}")
            return None
    
    def _match_manufacturer(self, text: str, manufacturer: str = None) -> Optional[Dict[str, Any]]:
        """メーカー名逆引き照合"""
        try:
            if not manufacturer:
                return None
            
            # メーカー名から推測
            if manufacturer in self.manufacturer_mapping:
                common_drugs = self.manufacturer_mapping[manufacturer]
                for drug in common_drugs:
                    if drug in text or text in drug:
                        return {
                            'name': drug,
                            'manufacturer': manufacturer,
                            'score': 0.7
                        }
            
            return None
        except Exception as e:
            logger.warning(f"Manufacturer matching failed: {e}")
            return None
    
    def _integrate_matches(self, matches: Dict[str, Any], manufacturer: str = None, dose: str = None) -> Dict[str, Any]:
        """マッチ結果の統合とリランキング"""
        try:
            # 有効なマッチを収集
            valid_matches = []
            for layer, match in matches.items():
                if match:
                    valid_matches.append((layer, match))
            
            if not valid_matches:
                return None
            
            # 重複除去とスコアリング
            unique_matches = {}
            for layer, match in valid_matches:
                name = match['name']
                score = match['score']
                
                # メーカー名ボーナス
                if manufacturer and match.get('manufacturer') == manufacturer:
                    score += 0.1
                
                # 既存のマッチより高いスコアの場合のみ更新
                if name not in unique_matches or score > unique_matches[name]['score']:
                    unique_matches[name] = {
                        'name': name,
                        'score': score,
                        'layers': [layer],
                        'details': match
                    }
                elif name in unique_matches:
                    unique_matches[name]['layers'].append(layer)
                    unique_matches[name]['score'] = max(unique_matches[name]['score'], score)
            
            # スコアでソート
            sorted_matches = sorted(
                unique_matches.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            
            if sorted_matches:
                best_match = sorted_matches[0]
                logger.info(f"Best match: {best_match['name']} (score: {best_match['score']:.2f})")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Match integration failed: {e}")
            return None
    
    def _calculate_confidence(self, best_match: Dict[str, Any], all_matches: Dict[str, Any]) -> float:
        """信頼度計算"""
        if not best_match:
            return 0.0
        
        base_score = best_match['score']
        layer_count = len(best_match['layers'])
        
        # 複数層での一致は信頼度向上
        layer_bonus = min(layer_count * 0.05, 0.2)
        
        # 最終信頼度
        confidence = min(base_score + layer_bonus, 1.0)
        
        return confidence
    
    def get_manufacturer_hints(self, drug_name: str) -> List[str]:
        """薬剤名からメーカー名を推測"""
        hints = []
        for manufacturer, drugs in self.manufacturer_mapping.items():
            if drug_name in drugs:
                hints.append(manufacturer)
        return hints
    
    def validate_match(self, match_result: Dict[str, Any]) -> bool:
        """マッチ結果の妥当性検証"""
        if not match_result or not match_result.get('best_match'):
            return False
        
        confidence = match_result.get('confidence', 0.0)
        return confidence >= 0.7  # 信頼度閾値
