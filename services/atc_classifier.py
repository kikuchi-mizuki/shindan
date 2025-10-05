"""
ATCコードベース分類器
辞書ベース固定分類＋AI補助説明
"""
import logging
from typing import Dict, Any, Optional, List
from .drug_normalization_service import DrugNormalizationService

logger = logging.getLogger(__name__)

class ATCClassifier:
    """ATCコードベース分類器"""
    
    def __init__(self):
        self.normalization_service = DrugNormalizationService()
        
        # ATCコード分類辞書（主要分類）
        self.atc_classifications = {
            # A: 消化管・代謝系薬
            "A02": "消化管機能薬",
            "A02B": "制酸薬",
            "A02BC": "プロトンポンプ阻害薬",
            "A02BC03": "ランソプラゾール（PPI）",
            "A02BC06": "ボノプラザン（P-CAB）",
            
            # B: 血液・造血器系薬
            "B01": "抗血栓薬",
            "B01AC": "抗血小板薬",
            
            # C: 循環器系薬
            "C01": "強心薬",
            "C01DX16": "ニコランジル（冠血管拡張薬）",
            "C09": "RAAS系薬",
            "C09AA": "ACE阻害薬",
            "C09AA02": "エナラプリル（ACE阻害薬）",
            "C09CA": "ARB",
            "C09DB": "ARB＋Ca拮抗薬配合",
            "C09DB07": "テルミサルタン/アムロジピン（降圧薬）",
            "C09DX": "ARNI",
            "C09DX04": "サクビトリル/バルサルタン（ARNI）",
            
            # G: 泌尿生殖器系・性ホルモン薬
            "G04": "泌尿生殖器系薬",
            "G04BE": "勃起不全治療薬",
            "G04BE08": "タダラフィル（PDE5阻害薬）",
            
            # H: 全身ホルモン製剤
            "H02": "副腎皮質ホルモン",
            
            # J: 全身感染症治療薬
            "J01": "全身感染症治療薬",
            "J01FA": "マクロライド系抗生物質",
            
            # L: 抗腫瘍薬・免疫抑制薬
            "L01": "抗腫瘍薬",
            
            # M: 筋骨格系薬
            "M01": "抗炎症薬・抗リウマチ薬",
            "M01A": "NSAIDs",
            
            # N: 神経系薬
            "N05": "精神安定薬",
            "N05B": "睡眠薬",
            "N06": "抗うつ薬",
            "N06A": "抗うつ薬",
            "N06AB": "SSRI",
            
            # R: 呼吸器系薬
            "R03": "気管支拡張薬",
            "R06": "抗ヒスタミン薬",
            
            # S: 感覚器系薬
            "S01": "眼科用薬",
            
            # V: その他
            "V03": "その他の治療薬",
        }
        
        # 薬効分類マッピング
        self.drug_category_mapping = {
            "PDE5阻害薬": {
                "atc_codes": ["G04BE08"],
                "description": "前立腺肥大症・ED・肺高血圧治療薬",
                "mechanism": "PDE5酵素阻害による血管拡張"
            },
            "冠血管拡張薬": {
                "atc_codes": ["C01DX16"],
                "description": "狭心症治療薬",
                "mechanism": "冠血管拡張による心筋血流改善"
            },
            "ARNI": {
                "atc_codes": ["C09DX04"],
                "description": "心不全治療薬",
                "mechanism": "ネプリライシン阻害＋ARB作用"
            },
            "降圧薬（ARB＋Ca拮抗薬）": {
                "atc_codes": ["C09DB07"],
                "description": "高血圧治療薬",
                "mechanism": "ARB＋Ca拮抗薬の配合剤"
            },
            "ACE阻害薬": {
                "atc_codes": ["C09AA02"],
                "description": "高血圧・心不全治療薬",
                "mechanism": "ACE阻害による血圧降下"
            },
            "P-CAB": {
                "atc_codes": ["A02BC06"],
                "description": "新型胃酸抑制薬",
                "mechanism": "カリウムイオン競合型酸分泌抑制"
            },
            "PPI": {
                "atc_codes": ["A02BC03"],
                "description": "プロトンポンプ阻害薬",
                "mechanism": "プロトンポンプ阻害による胃酸分泌抑制"
            }
        }
    
    def classify_drug(self, drug_name: str, atc_codes: List[str] = None) -> Dict[str, Any]:
        """
        薬剤の分類（ATCコードベース）
        
        Args:
            drug_name: 薬剤名
            atc_codes: ATCコードリスト（オプション）
            
        Returns:
            分類結果
        """
        try:
            logger.info(f"Classifying drug: {drug_name}")
            
            # 1. 正規化
            normalized = self.normalization_service.normalize_drug_name(drug_name)
            
            # 2. ATCコード取得
            if not atc_codes:
                atc_codes = self._get_atc_codes(normalized)
            
            # 3. ATCコードベース分類
            classification = self._classify_by_atc(atc_codes)
            
            # 4. 薬効分類マッピング
            category_info = self._get_category_info(classification)
            
            # 5. 結果統合
            result = {
                'drug_name': drug_name,
                'normalized_name': normalized.get('normalized', drug_name),
                'atc_codes': atc_codes,
                'classification': classification,
                'category': category_info,
                'confidence': self._calculate_classification_confidence(classification, atc_codes)
            }
            
            logger.info(f"Classification result: {classification}")
            return result
            
        except Exception as e:
            logger.error(f"Drug classification failed: {e}")
            return {
                'drug_name': drug_name,
                'normalized_name': drug_name,
                'atc_codes': [],
                'classification': '不明',
                'category': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _get_atc_codes(self, normalized_drug: Dict[str, Any]) -> List[str]:
        """薬剤名からATCコードを取得"""
        try:
            # 薬剤辞書から直接取得
            if normalized_drug.get('category'):
                category = normalized_drug['category']
                if category in self.drug_category_mapping:
                    return self.drug_category_mapping[category]['atc_codes']
            
            # KEGGクライアントから取得
            from .kegg_client import KEGGClient
            kegg_client = KEGGClient()
            kegg_result = kegg_client.best_kegg_and_atc(normalized_drug.get('normalized', ''))
            if kegg_result and kegg_result.get('atc'):
                return kegg_result['atc']
            
            return []
            
        except Exception as e:
            logger.warning(f"ATC code retrieval failed: {e}")
            return []
    
    def _classify_by_atc(self, atc_codes: List[str]) -> str:
        """ATCコードによる分類"""
        if not atc_codes:
            return '不明'
        
        # 最も詳細なATCコードで分類
        best_atc = max(atc_codes, key=len)
        
        # 直接マッチ
        if best_atc in self.atc_classifications:
            return self.atc_classifications[best_atc]
        
        # 部分マッチ（上位分類）
        for atc_code, classification in self.atc_classifications.items():
            if best_atc.startswith(atc_code):
                return classification
        
        # デフォルト分類
        first_letter = best_atc[0] if best_atc else 'V'
        default_classifications = {
            'A': '消化管・代謝系薬',
            'B': '血液・造血器系薬',
            'C': '循環器系薬',
            'D': '皮膚科用薬',
            'G': '泌尿生殖器系薬',
            'H': '全身ホルモン製剤',
            'J': '全身感染症治療薬',
            'L': '抗腫瘍薬・免疫抑制薬',
            'M': '筋骨格系薬',
            'N': '神経系薬',
            'P': '駆虫薬・殺虫薬',
            'R': '呼吸器系薬',
            'S': '感覚器系薬',
            'V': 'その他'
        }
        
        return default_classifications.get(first_letter, '不明')
    
    def _get_category_info(self, classification: str) -> Optional[Dict[str, Any]]:
        """薬効分類情報を取得"""
        for category, info in self.drug_category_mapping.items():
            if category in classification or classification in category:
                return {
                    'name': category,
                    'description': info['description'],
                    'mechanism': info['mechanism'],
                    'atc_codes': info['atc_codes']
                }
        
        return None
    
    def _calculate_classification_confidence(self, classification: str, atc_codes: List[str]) -> float:
        """分類信頼度計算"""
        if not classification or classification == '不明':
            return 0.0
        
        if not atc_codes:
            return 0.5
        
        # ATCコードの詳細度に基づく信頼度
        max_atc_length = max(len(code) for code in atc_codes)
        
        if max_atc_length >= 7:  # 7桁以上は高信頼度
            return 0.9
        elif max_atc_length >= 5:  # 5-6桁は中信頼度
            return 0.7
        else:  # 4桁以下は低信頼度
            return 0.5
    
    def get_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """薬剤相互作用情報を取得"""
        try:
            normalized = self.normalization_service.normalize_drug_name(drug_name)
            tags = self.normalization_service.get_interaction_tags(normalized.get('normalized', ''))
            
            interactions = []
            for tag in tags:
                interactions.append({
                    'tag': tag,
                    'description': self._get_interaction_description(tag)
                })
            
            return interactions
            
        except Exception as e:
            logger.error(f"Interaction retrieval failed: {e}")
            return []
    
    def _get_interaction_description(self, tag: str) -> str:
        """相互作用タグの説明を取得"""
        descriptions = {
            'RAAS': 'RAAS系薬剤との併用注意',
            'ARNI': 'ARNI系薬剤との併用注意',
            'ARB': 'ARB系薬剤との併用注意',
            'ACEI': 'ACE阻害薬との併用注意',
            'CCB': 'Ca拮抗薬との併用注意',
            'PPI': 'PPI系薬剤との併用注意',
            'P-CAB': 'P-CAB系薬剤との併用注意',
            'PDE5': 'PDE5阻害薬との併用注意',
            'NITRATE': '硝酸薬との併用注意',
            'STIM_LAX': '刺激性下剤との併用注意',
            'ANTI_PLATELET': '抗血小板薬との併用注意',
            'PHOS_BINDER': 'リン吸着薬との併用注意',
            'OPIOID': 'オピオイド系薬剤との併用注意',
            'ANALGESIC': '鎮痛薬との併用注意'
        }
        
        return descriptions.get(tag, '相互作用の可能性あり')
    
    def validate_classification(self, classification_result: Dict[str, Any]) -> bool:
        """分類結果の妥当性検証"""
        if not classification_result:
            return False
        
        confidence = classification_result.get('confidence', 0.0)
        classification = classification_result.get('classification', '')
        
        return confidence >= 0.7 and classification != '不明'
