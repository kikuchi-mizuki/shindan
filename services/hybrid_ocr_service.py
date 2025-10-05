"""
ハイブリッドOCRサービス
Gemini/Visionモデルによる複数候補生成と補正
"""
import logging
import re
from typing import List, Dict, Any, Optional
import requests
import json

logger = logging.getLogger(__name__)

class HybridOCRService:
    """ハイブリッドOCRサービス（複数候補生成＋補正）"""
    
    def __init__(self):
        self.ocr_engines = ['gemini', 'vision', 'tesseract']
        self.confidence_threshold = 0.8
        
    def extract_drug_candidates(self, image_path: str) -> List[Dict[str, Any]]:
        """
        画像から薬剤名候補を複数生成
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            薬剤名候補リスト（信頼度付き）
        """
        candidates = []
        
        # 1. Gemini OCR
        gemini_candidates = self._extract_with_gemini(image_path)
        if gemini_candidates:
            candidates.extend(gemini_candidates)
        
        # 2. Google Vision OCR
        vision_candidates = self._extract_with_vision(image_path)
        if vision_candidates:
            candidates.extend(vision_candidates)
        
        # 3. Tesseract OCR（フォールバック）
        tesseract_candidates = self._extract_with_tesseract(image_path)
        if tesseract_candidates:
            candidates.extend(tesseract_candidates)
        
        # 4. 候補の重複除去とスコアリング
        unique_candidates = self._deduplicate_and_score(candidates)
        
        return unique_candidates
    
    def _extract_with_gemini(self, image_path: str) -> List[Dict[str, Any]]:
        """Gemini OCRによる薬剤名抽出"""
        try:
            # Gemini API呼び出し（実装例）
            prompt = """
            この画像から薬剤名を抽出してください。
            以下の形式で回答してください：
            薬剤名1: 信頼度(0-1)
            薬剤名2: 信頼度(0-1)
            ...
            """
            
            # 実際のAPI呼び出しは環境に応じて実装
            # response = gemini_api_call(image_path, prompt)
            
            # モックデータ（実際の実装時は削除）
            mock_candidates = [
                {"text": "タダラフィル 5mg", "confidence": 0.95, "engine": "gemini"},
                {"text": "ニコランジル 5mg", "confidence": 0.92, "engine": "gemini"},
                {"text": "エンレスト 100mg", "confidence": 0.88, "engine": "gemini"},
                {"text": "テラムロAP", "confidence": 0.85, "engine": "gemini"},
                {"text": "エナラプリル 2.5mg", "confidence": 0.90, "engine": "gemini"},
                {"text": "タケキャブ 10mg", "confidence": 0.87, "engine": "gemini"},
                {"text": "ランソプラゾール 15mg", "confidence": 0.93, "engine": "gemini"}
            ]
            
            return mock_candidates
            
        except Exception as e:
            logger.error(f"Gemini OCR failed: {e}")
            return []
    
    def _extract_with_vision(self, image_path: str) -> List[Dict[str, Any]]:
        """Google Vision OCRによる薬剤名抽出"""
        try:
            # Google Vision API呼び出し（実装例）
            # response = vision_api_call(image_path)
            
            # モックデータ（実際の実装時は削除）
            mock_candidates = [
                {"text": "タダラフィル 5mg ZA", "confidence": 0.94, "engine": "vision"},
                {"text": "ニコランジル 5mg トーワ", "confidence": 0.91, "engine": "vision"},
                {"text": "サクビトリル/バルサルタン 100mg", "confidence": 0.89, "engine": "vision"},
                {"text": "テラムロAP サワイ", "confidence": 0.86, "engine": "vision"},
                {"text": "エナラプリル 2.5mg トーワ", "confidence": 0.92, "engine": "vision"},
                {"text": "タケキャブ 10mg", "confidence": 0.88, "engine": "vision"},
                {"text": "ランソプラゾールOD 15mg トーワ", "confidence": 0.94, "engine": "vision"}
            ]
            
            return mock_candidates
            
        except Exception as e:
            logger.error(f"Vision OCR failed: {e}")
            return []
    
    def _extract_with_tesseract(self, image_path: str) -> List[Dict[str, Any]]:
        """Tesseract OCRによる薬剤名抽出（フォールバック）"""
        try:
            # Tesseract OCR呼び出し（実装例）
            # response = tesseract_ocr(image_path)
            
            # モックデータ（実際の実装時は削除）
            mock_candidates = [
                {"text": "タダラフィル 5mg", "confidence": 0.75, "engine": "tesseract"},
                {"text": "ニコランジル 5mg", "confidence": 0.73, "engine": "tesseract"},
                {"text": "エンレスト 100mg", "confidence": 0.71, "engine": "tesseract"},
                {"text": "テラムロプリド", "confidence": 0.65, "engine": "tesseract"},  # 誤認識例
                {"text": "エナラプリル 2.5mg", "confidence": 0.78, "engine": "tesseract"},
                {"text": "ラベプラゾール 10mg", "confidence": 0.68, "engine": "tesseract"},  # 誤認識例
                {"text": "ランソプラゾール 15mg", "confidence": 0.76, "engine": "tesseract"}
            ]
            
            return mock_candidates
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return []
    
    def _deduplicate_and_score(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """候補の重複除去とスコアリング"""
        unique_candidates = {}
        
        for candidate in candidates:
            text = candidate['text'].strip()
            confidence = candidate['confidence']
            engine = candidate['engine']
            
            # 重複チェック（類似度ベース）
            is_duplicate = False
            for existing_text, existing_candidate in unique_candidates.items():
                if self._calculate_text_similarity(text, existing_text) > 0.9:
                    # より高い信頼度の候補を採用
                    if confidence > existing_candidate['confidence']:
                        unique_candidates[existing_text] = candidate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates[text] = candidate
        
        # 信頼度でソート
        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return sorted_candidates
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキスト類似度計算"""
        if not text1 or not text2:
            return 0.0
        
        # 正規化
        text1 = re.sub(r'\s+', '', text1.lower())
        text2 = re.sub(r'\s+', '', text2.lower())
        
        # 完全一致
        if text1 == text2:
            return 1.0
        
        # 部分一致
        if text1 in text2 or text2 in text1:
            return 0.9
        
        # 文字レベル類似度
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()
    
    def get_best_candidates(self, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """上位候補を取得"""
        return candidates[:top_k]
    
    def validate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """候補の妥当性検証（存在確認付き）"""
        validated_candidates = []
        
        # 薬剤存在確認サービスをインポート
        try:
            from .drug_existence_checker import DrugExistenceChecker
            existence_checker = DrugExistenceChecker()
        except ImportError:
            existence_checker = None
        
        for candidate in candidates:
            text = candidate['text']
            confidence = candidate['confidence']
            
            # 基本的な妥当性チェック
            if not self._is_valid_drug_name(text) or confidence < self.confidence_threshold:
                continue
            
            # 薬剤存在確認
            if existence_checker:
                existence_result = existence_checker.check_drug_existence(text)
                
                # 存在しない薬剤名の場合は補正候補を取得
                if not existence_result['exists'] and existence_result['corrected_name']:
                    # 補正された薬剤名で候補を更新
                    corrected_candidate = candidate.copy()
                    corrected_candidate['text'] = existence_result['corrected_name']
                    corrected_candidate['correction_reason'] = existence_result['correction_reason']
                    corrected_candidate['original_text'] = text
                    validated_candidates.append(corrected_candidate)
                    continue
                elif not existence_result['exists']:
                    # 補正できない場合は除外
                    continue
            
            validated_candidates.append(candidate)
        
        return validated_candidates
    
    def _is_valid_drug_name(self, text: str) -> bool:
        """薬剤名の妥当性チェック"""
        if not text or len(text) < 2:
            return False
        
        # 薬剤名らしいパターンをチェック
        drug_patterns = [
            r'^[ア-ン]+',  # ひらがな・カタカナ
            r'\d+mg',      # 用量表記
            r'[錠|カプセル|散|液]',  # 剤形
            r'[A-Z]{2,}',   # アルファベット（商品名）
        ]
        
        for pattern in drug_patterns:
            if re.search(pattern, text):
                return True
        
        return False
