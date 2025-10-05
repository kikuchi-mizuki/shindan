"""
ハイブリッド薬剤検出パイプライン
AI×辞書の最適化設計による高精度検出
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
from .hybrid_ocr_service import HybridOCRService
from .multi_layer_matcher import MultiLayerMatcher
from .atc_classifier import ATCClassifier
from .drug_normalization_service import DrugNormalizationService

logger = logging.getLogger(__name__)

class HybridPipeline:
    """ハイブリッド薬剤検出パイプライン"""
    
    def __init__(self):
        self.ocr_service = HybridOCRService()
        self.matcher = MultiLayerMatcher()
        self.classifier = ATCClassifier()
        self.normalization_service = DrugNormalizationService()
        
        # パイプライン設定
        self.confidence_threshold = 0.8
        self.max_candidates = 10
        
    def process_prescription_image(self, 
                                 image_path: str, 
                                 session_id: str = None) -> Dict[str, Any]:
        """
        処方箋画像を処理して薬剤情報を抽出・分析
        
        Args:
            image_path: 処方箋画像のパス
            session_id: セッションID（オプション）
            
        Returns:
            薬剤検出結果
        """
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            logger.info(f"Starting hybrid pipeline processing for session: {session_id}")
            
            # 1. ハイブリッドOCR（複数候補生成）
            ocr_candidates = self.ocr_service.extract_drug_candidates(image_path)
            logger.info(f"OCR candidates: {len(ocr_candidates)}")
            
            # 2. 候補の妥当性検証
            validated_candidates = self.ocr_service.validate_candidates(ocr_candidates)
            logger.info(f"Validated candidates: {len(validated_candidates)}")
            
            # 3. 多層辞書照合
            matched_drugs = []
            for candidate in validated_candidates[:self.max_candidates]:
                match_result = self.matcher.match_drug(
                    candidate['text'],
                    manufacturer=self._extract_manufacturer(candidate['text']),
                    dose=self._extract_dose(candidate['text'])
                )
                
                if self.matcher.validate_match(match_result):
                    matched_drugs.append(match_result)
            
            logger.info(f"Matched drugs: {len(matched_drugs)}")
            
            # 4. ATCコードベース分類
            classified_drugs = []
            for drug_match in matched_drugs:
                classification_result = self.classifier.classify_drug(
                    drug_match['best_match']['name'],
                    drug_match['best_match'].get('details', {}).get('atc_codes', [])
                )
                
                if self.classifier.validate_classification(classification_result):
                    classified_drugs.append({
                        'drug_match': drug_match,
                        'classification': classification_result
                    })
            
            logger.info(f"Classified drugs: {len(classified_drugs)}")
            
            # 5. 結果統合
            final_drugs = self._integrate_results(classified_drugs)
            
            # 6. 品質評価
            quality_metrics = self._evaluate_quality(final_drugs, ocr_candidates)
            
            # 7. 結果返却
            result = {
                'session_id': session_id,
                'drugs': final_drugs,
                'quality_metrics': quality_metrics,
                'processing_stats': {
                    'ocr_candidates': len(ocr_candidates),
                    'validated_candidates': len(validated_candidates),
                    'matched_drugs': len(matched_drugs),
                    'classified_drugs': len(classified_drugs),
                    'final_drugs': len(final_drugs)
                }
            }
            
            logger.info(f"Pipeline completed. Final drugs: {len(final_drugs)}")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid pipeline processing failed: {e}")
            return {
                'session_id': session_id,
                'drugs': [],
                'quality_metrics': {'overall_confidence': 0.0},
                'processing_stats': {},
                'error': str(e)
            }
    
    def _extract_manufacturer(self, text: str) -> Optional[str]:
        """テキストからメーカー名を抽出（強化版）"""
        # 主要製薬会社名のパターン
        manufacturer_patterns = {
            'サンド': ['サンド', 'SANDOZ', 'ノバルティス'],
            'トーワ': ['トーワ', 'TOWA', '東和'],
            'サワイ': ['サワイ', 'SAWAI', '沢井'],
            '武田': ['武田', 'TAKEDA', 'タケダ'],
            'ノバルティス': ['ノバルティス', 'NOVARTIS'],
            'ファイザー': ['ファイザー', 'PFIZER'],
            'MSD': ['MSD', 'メルク'],
            'アステラス': ['アステラス', 'ASTELLAS'],
            '第一三共': ['第一三共', 'DAIICHI SANKYO'],
            'エーザイ': ['エーザイ', 'EISAI']
        }
        
        # テキストからメーカー名を検索
        for manufacturer, patterns in manufacturer_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return manufacturer
        
        return None
    
    def _extract_dose(self, text: str) -> Optional[str]:
        """テキストから用量を抽出"""
        import re
        
        # 用量パターンを検索
        dose_patterns = [
            r'(\d+\.?\d*)\s*mg',
            r'(\d+\.?\d*)\s*g',
            r'(\d+\.?\d*)\s*ml',
            r'(\d+\.?\d*)\s*μg'
        ]
        
        for pattern in dose_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1) + pattern.split('\\')[1]
        
        return None
    
    def _integrate_results(self, classified_drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """結果の統合と重複除去"""
        integrated_drugs = []
        seen_drugs = set()
        
        for drug_data in classified_drugs:
            drug_match = drug_data['drug_match']
            classification = drug_data['classification']
            
            drug_name = drug_match['best_match']['name']
            
            # 重複チェック
            if drug_name in seen_drugs:
                continue
            
            seen_drugs.add(drug_name)
            
            # 統合結果
            integrated_drug = {
                'name': drug_name,
                'normalized_name': classification['normalized_name'],
                'dose': self._extract_dose(drug_match['original_text']),
                'manufacturer': drug_match.get('manufacturer_hint'),
                'classification': classification['classification'],
                'category': classification['category'],
                'atc_codes': classification['atc_codes'],
                'confidence': drug_match['confidence'],
                'classification_confidence': classification['confidence'],
                'overall_confidence': (drug_match['confidence'] + classification['confidence']) / 2,
                'source_text': drug_match['original_text'],
                'processing_method': 'hybrid_pipeline'
            }
            
            integrated_drugs.append(integrated_drug)
        
        # 信頼度でソート
        integrated_drugs.sort(key=lambda x: x['overall_confidence'], reverse=True)
        
        return integrated_drugs
    
    def _evaluate_quality(self, drugs: List[Dict[str, Any]], ocr_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """品質評価"""
        if not drugs:
            return {'overall_confidence': 0.0, 'quality_score': 0.0}
        
        # 信頼度統計
        confidences = [drug['overall_confidence'] for drug in drugs]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # 高信頼度薬剤の割合
        high_confidence_count = len([c for c in confidences if c >= 0.8])
        high_confidence_ratio = high_confidence_count / len(confidences)
        
        # 品質スコア計算
        quality_score = (
            avg_confidence * 0.4 +
            high_confidence_ratio * 0.3 +
            min_confidence * 0.2 +
            (len(drugs) / len(ocr_candidates)) * 0.1 if ocr_candidates else 0
        )
        
        return {
            'overall_confidence': avg_confidence,
            'quality_score': quality_score,
            'confidence_stats': {
                'min': min_confidence,
                'max': max_confidence,
                'avg': avg_confidence
            },
            'high_confidence_ratio': high_confidence_ratio,
            'drug_count': len(drugs),
            'ocr_candidate_count': len(ocr_candidates)
        }
    
    def get_drug_interactions(self, drugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """薬剤相互作用チェック"""
        try:
            interactions = []
            
            for drug in drugs:
                drug_interactions = self.classifier.get_drug_interactions(drug['name'])
                if drug_interactions:
                    interactions.append({
                        'drug': drug['name'],
                        'interactions': drug_interactions
                    })
            
            return {
                'interactions': interactions,
                'interaction_count': len(interactions)
            }
            
        except Exception as e:
            logger.error(f"Interaction checking failed: {e}")
            return {'interactions': [], 'interaction_count': 0}
    
    def validate_pipeline_result(self, result: Dict[str, Any]) -> bool:
        """パイプライン結果の妥当性検証"""
        if not result or 'drugs' not in result:
            return False
        
        drugs = result['drugs']
        if not drugs:
            return False
        
        # 最低1つの高信頼度薬剤が必要
        high_confidence_drugs = [d for d in drugs if d.get('overall_confidence', 0) >= 0.8]
        
        return len(high_confidence_drugs) >= 1
    
    def get_processing_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """処理結果のサマリー取得"""
        if not result:
            return {}
        
        drugs = result.get('drugs', [])
        quality_metrics = result.get('quality_metrics', {})
        processing_stats = result.get('processing_stats', {})
        
        return {
            'total_drugs': len(drugs),
            'high_confidence_drugs': len([d for d in drugs if d.get('overall_confidence', 0) >= 0.8]),
            'average_confidence': quality_metrics.get('overall_confidence', 0.0),
            'quality_score': quality_metrics.get('quality_score', 0.0),
            'processing_efficiency': processing_stats.get('final_drugs', 0) / max(processing_stats.get('ocr_candidates', 1), 1),
            'success_rate': len(drugs) / max(processing_stats.get('validated_candidates', 1), 1)
        }
