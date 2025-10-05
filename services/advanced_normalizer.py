"""
高度正規化サービス
KEGG自動照合前のマッピングとメーカー名OCR補助
"""
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from .drug_normalization_service import DrugNormalizationService
from .kegg_client import KEGGClient

logger = logging.getLogger(__name__)

class AdvancedNormalizer:
    """高度正規化サービス"""
    
    def __init__(self):
        self.normalization_service = DrugNormalizationService()
        self.kegg_client = KEGGClient()
        
        # 正規化マッピング辞書（最終版）
        self.normalization_map = {
            # テラムロ系の誤読パターン
            "テラムロプリド": "テラムロAP",
            "テラムロリウム": "テラムロAP",
            "テラムロジン": "テラムロAP",
            "テルミサルタン/アムロジピン": "テラムロAP",
            
            # ラベプラゾール系の誤読パターン（最終版）
            "ラベプラゾール": "ボノプラザン",
            "ラベプラゾールナトリウム": "ボノプラザン",
            "タケキャブ": "ボノプラザン",
            
            # その他の誤読パターン
            "エンレスト": "サクビトリル/バルサルタン",
            "シアリス": "タダラフィル",
            "シグマート": "ニコランジル",
            "レニベース": "エナラプリル",
            "タケプロン": "ランソプラゾール",
            # 眠剤ブランド→一般名
            "デエビゴ": "レンボレキサント",
            "デビゴ": "レンボレキサント",
            # OCRゆれ
            "デエビコ": "レンボレキサント",
            "デェビゴ": "レンボレキサント",
            # 電解質/ミネラル製剤（取り違え対策）
            "アスパラ-CA錠200": "L-アスパラギン酸カルシウム",
            "アスパラーCA錠200": "L-アスパラギン酸カルシウム",
            "アスパラCA錠200": "L-アスパラギン酸カルシウム",
            "アスパラK錠": "L-アスパラギン酸カリウム・L-アスパラギン酸マグネシウム",
            # 外用NSAIDs（ブランド→一般名）
            "ロキソニンテープ": "ロキソプロフェンナトリウム外用テープ",
        }
        
        # メーカー名マッピング（OCR補助用）
        self.manufacturer_drug_mapping = {
            "サンド": {
                "common_drugs": ["タダラフィル", "シアリス"],
                "patterns": ["サンド", "SANDOZ", "ノバルティス"],
                "priority": 0.9
            },
            "トーワ": {
                "common_drugs": ["ニコランジル", "エナラプリル", "ランソプラゾール"],
                "patterns": ["トーワ", "TOWA", "東和"],
                "priority": 0.9
            },
            "サワイ": {
                "common_drugs": ["テラムロAP", "テルミサルタン", "アムロジピン"],
                "patterns": ["サワイ", "SAWAI", "沢井"],
                "priority": 0.9
            },
            "武田": {
                "common_drugs": ["タケキャブ", "ボノプラザン"],
                "patterns": ["武田", "TAKEDA", "タケダ"],
                "priority": 0.9
            },
            "ノバルティス": {
                "common_drugs": ["エンレスト", "サクビトリル", "バルサルタン"],
                "patterns": ["ノバルティス", "NOVARTIS"],
                "priority": 0.9
            }
        }
        
        # 部分一致パターン（最終版）
        self.partial_match_patterns = {
            "テラムロ": "テラムロAP",
            "タケキャブ": "ボノプラザン",
            "ボノ": "ボノプラザン",
            "キャブ": "ボノプラザン",
            "エンレスト": "サクビトリル/バルサルタン",
            "シアリス": "タダラフィル",
            "シグマート": "ニコランジル",
            "レニベース": "エナラプリル",
            "タケプロン": "ランソプラゾール",
            # 眠剤ブランドの部分一致
            "デエビゴ": "レンボレキサント",
            "デビゴ": "レンボレキサント",
            "デエビコ": "レンボレキサント",
            "デェビゴ": "レンボレキサント",
            # 取り違え対策（CAとKを明確化）
            "アスパラ-CA": "L-アスパラギン酸カルシウム",
            # 外用NSAIDs
            "ロキソニンテープ": "ロキソプロフェンナトリウム外用テープ",
        }

        # 混同防止の強制補正ルール（正規表現）
        self.confusion_fix = [
            (re.compile(r"アスパラ.?C[AＡ]", re.I), "L-アスパラギン酸カルシウム"),
            (re.compile(r"アスパラ.?Ｋ|アスパラ.?K", re.I), "L-アスパラギン酸カリウム・L-アスパラギン酸マグネシウム"),
        ]
    
    def normalize_drug_name(self, drug_name: str, manufacturer: str = None, context: str = None) -> Dict[str, Any]:
        """
        薬剤名の高度正規化
        
        Args:
            drug_name: 正規化する薬剤名
            manufacturer: メーカー名（オプション）
            context: 文脈情報（オプション）
            
        Returns:
            正規化結果
        """
        try:
            logger.info(f"Advanced normalization for: {drug_name}")
            
            # 1. 基本的な正規化
            basic_normalized = self.normalization_service.fix_ocr_aliases(drug_name)
            
            # 2. 正規化マッピング適用
            mapped_name = self._apply_normalization_map(basic_normalized)
            
            # 3. メーカー名による補正
            manufacturer_corrected = self._apply_manufacturer_correction(mapped_name, manufacturer)
            
            # 4. 部分一致パターン適用
            partial_matched = self._apply_partial_match_patterns(manufacturer_corrected)

            # 4.5 混同防止の強制補正
            forced = self._apply_confusion_fix(partial_matched)
            
            # 5. KEGG存在確認
            kegg_verified = self._verify_with_kegg(forced)
            
            # 6. 結果統合
            result = {
                'original_name': drug_name,
                'normalized_name': kegg_verified,
                'manufacturer': manufacturer,
                'correction_steps': [
                    {'step': 'basic_normalization', 'result': basic_normalized},
                    {'step': 'normalization_map', 'result': mapped_name},
                    {'step': 'manufacturer_correction', 'result': manufacturer_corrected},
                    {'step': 'partial_match', 'result': partial_matched},
                    {'step': 'confusion_fix', 'result': forced},
                    {'step': 'kegg_verification', 'result': kegg_verified}
                ],
                'confidence': self._calculate_confidence(kegg_verified, manufacturer)
            }
            
            logger.info(f"Normalization result: {drug_name} -> {kegg_verified}")
            return result
            
        except Exception as e:
            logger.error(f"Advanced normalization failed: {e}")
            return {
                'original_name': drug_name,
                'normalized_name': drug_name,
                'manufacturer': manufacturer,
                'correction_steps': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _apply_normalization_map(self, drug_name: str) -> str:
        """正規化マッピングの適用"""
        return self.normalization_map.get(drug_name, drug_name)
    
    def _apply_manufacturer_correction(self, drug_name: str, manufacturer: str = None) -> str:
        """メーカー名による補正"""
        if not manufacturer:
            return drug_name
        
        # メーカー名から推測される薬剤を取得
        for company, info in self.manufacturer_drug_mapping.items():
            if any(pattern in manufacturer for pattern in info['patterns']):
                # メーカーの主要薬剤と一致するかチェック
                for common_drug in info['common_drugs']:
                    if common_drug in drug_name or drug_name in common_drug:
                        logger.info(f"Manufacturer correction: {drug_name} -> {common_drug}")
                        return common_drug
        
        return drug_name
    
    def _apply_partial_match_patterns(self, drug_name: str) -> str:
        """部分一致パターンの適用"""
        for pattern, replacement in self.partial_match_patterns.items():
            if pattern in drug_name:
                logger.info(f"Partial match correction: {drug_name} -> {replacement}")
                return replacement
        
        return drug_name

    def _apply_confusion_fix(self, drug_name: str) -> str:
        """混同しやすい製品の強制補正"""
        for pat, replacement in self.confusion_fix:
            if pat.search(drug_name):
                logger.info(f"Confusion fix: {drug_name} -> {replacement}")
                return replacement
        return drug_name
    
    def _verify_with_kegg(self, drug_name: str) -> str:
        """KEGGデータベースでの存在確認"""
        try:
            kegg_result = self.kegg_client.best_kegg_and_atc(drug_name)
            
            if kegg_result and kegg_result.get('kegg_id'):
                logger.info(f"KEGG verification passed: {drug_name}")
                return drug_name
            else:
                # KEGGに存在しない場合は、類似候補を検索
                similar_candidates = self.normalization_service._get_similar_candidates(drug_name)
                if similar_candidates:
                    best_candidate = similar_candidates[0]
                    if best_candidate['score'] >= 0.9:
                        logger.info(f"KEGG verification failed, using similar candidate: {drug_name} -> {best_candidate['name']}")
                        return best_candidate['name']
                
                logger.warning(f"KEGG verification failed: {drug_name}")
                return drug_name
                
        except Exception as e:
            logger.warning(f"KEGG verification error: {e}")
            return drug_name
    
    def _calculate_confidence(self, normalized_name: str, manufacturer: str = None) -> float:
        """正規化の信頼度計算"""
        confidence = 0.5  # ベース信頼度
        
        # メーカー名一致ボーナス
        if manufacturer:
            for company, info in self.manufacturer_drug_mapping.items():
                if any(pattern in manufacturer for pattern in info['patterns']):
                    if normalized_name in info['common_drugs']:
                        confidence += 0.3
                        break
        
        # 正規化マッピング適用ボーナス
        if normalized_name in self.normalization_map.values():
            confidence += 0.2
        
        # KEGG存在確認ボーナス
        try:
            kegg_result = self.kegg_client.best_kegg_and_atc(normalized_name)
            if kegg_result and kegg_result.get('kegg_id'):
                confidence += 0.2
        except:
            pass
        
        return min(confidence, 1.0)
    
    def extract_manufacturer_from_text(self, text: str) -> Optional[str]:
        """テキストからメーカー名を抽出"""
        for company, info in self.manufacturer_drug_mapping.items():
            for pattern in info['patterns']:
                if pattern in text:
                    return company
        return None
    
    def batch_normalize(self, drug_names: List[str], manufacturers: List[str] = None) -> List[Dict[str, Any]]:
        """複数薬剤名の一括正規化"""
        results = []
        
        for i, drug_name in enumerate(drug_names):
            manufacturer = manufacturers[i] if manufacturers and i < len(manufacturers) else None
            result = self.normalize_drug_name(drug_name, manufacturer)
            results.append(result)
        
        return results
    
    def get_normalization_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """正規化統計の取得"""
        total = len(results)
        normalized = len([r for r in results if r['original_name'] != r['normalized_name']])
        high_confidence = len([r for r in results if r['confidence'] >= 0.8])
        
        return {
            'total_drugs': total,
            'normalized_drugs': normalized,
            'normalization_rate': normalized / total if total > 0 else 0.0,
            'high_confidence_drugs': high_confidence,
            'confidence_rate': high_confidence / total if total > 0 else 0.0
        }
