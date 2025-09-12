"""
AI抽出サービス - OpenAI APIを使用した薬剤名抽出・正規化
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIExtractorService:
    """AI抽出サービス"""
    
    def __init__(self):
        """初期化"""
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """OpenAI APIの初期化"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API: {e}")
            self.openai_client = None
    
    def extract_drugs(self, ocr_text: str) -> Dict[str, Any]:
        """
        OCRテキストから薬剤情報を抽出・正規化
        
        Args:
            ocr_text: OCRで抽出された生テキスト
            
        Returns:
            Dict containing extracted drugs and metadata
        """
        if not self.openai_client:
            logger.error("OpenAI API not available")
            return {
                'drugs': [],
                'confidence': 'low',
                'error': 'OpenAI API not available',
                'raw_text': ocr_text
            }
        
        try:
            logger.info("Starting AI drug extraction")
            
            # プロンプトの構築
            prompt = self._build_extraction_prompt(ocr_text)
            
            # OpenAI API呼び出し
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは薬剤情報抽出の専門家です。処方箋や薬剤リストから正確な薬剤情報を抽出し、JSON形式で返してください。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # レスポンスの解析
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"AI response received: {len(ai_response)} characters")
            
            # JSON解析
            extracted_data = self._parse_ai_response(ai_response)
            
            # 信頼度評価
            confidence = self._evaluate_confidence(extracted_data, ocr_text)
            
            result = {
                'drugs': extracted_data.get('drugs', []),
                'confidence': confidence,
                'raw_text': ocr_text,
                'ai_response': ai_response,
                'extraction_metadata': {
                    'model': 'gpt-4o-mini',
                    'tokens_used': response.usage.total_tokens if response.usage else 0
                }
            }
            
            logger.info(f"AI extraction completed: {len(result['drugs'])} drugs, confidence: {confidence}")
            return result
            
        except Exception as e:
            logger.error(f"AI extraction error: {e}")
            return {
                'drugs': [],
                'confidence': 'low',
                'error': str(e),
                'raw_text': ocr_text
            }
    
    def _build_extraction_prompt(self, ocr_text: str) -> str:
        """抽出用プロンプトを構築"""
        return f"""
以下のOCRテキストから薬剤情報を抽出し、JSON形式で返してください。

【OCRテキスト】
{ocr_text}

【抽出ルール】
1. 薬剤名は商品名から一般名（正式名称）に正規化してください
2. 用量（mg、μg、錠数など）を正確に抽出してください
3. 用法・用量（1日何回、いつ服用など）を抽出してください
4. 日数（何日分）を抽出してください
5. 不明な情報は空文字列またはnullにしてください
6. 薬剤以外の情報（患者名、医師名など）は無視してください
7. 行頭番号の有無に関わらず、剤形・用量パターン（錠|カプセル|顆粒|ゲル|液など）を含む行を薬剤として認識してください
8. 番号なしで記載されている薬剤も必ず抽出してください

【出力形式】
```json
{{
  "drugs": [
    {{
      "raw": "元のテキスト",
      "generic": "一般名（正式名称）",
      "strength": "用量（例：5mg、10錠）",
      "dose": "1回量（例：1錠、2錠）",
      "freq": "服用頻度（例：1日3回、就寝前）",
      "days": 日数（数値）,
      "category": "薬効分類（例：消化管機能薬、高尿酸血症治療薬、NSAIDs、下剤、漢方など）"
    }}
  ]
}}
```

【注意事項】
- 商品名は必ず一般名に変換してください（例：マイスリー→ゾルピデム酒石酸塩、オルケディア→エボカルセト）
- 用量は正確に抽出し、単位も含めてください
- 複数の薬剤がある場合は配列に追加してください
- 薬剤以外の情報は抽出しないでください
- 不明確な情報は推測せず、空文字列にしてください
- 行頭に番号がない薬剤も必ず抽出してください（例：「センノシド錠12mg」「ラキソベロン錠2.5mg」など）
- 剤形（錠、カプセル、顆粒、ゲル、液など）を含む行は薬剤として認識してください
- 薬効分類は以下のカテゴリから適切なものを選択してください：
  * 消化管機能薬（PPI、H2ブロッカー、P-CAB、消化管運動改善薬など）
  * 高尿酸血症治療薬（尿酸生成阻害薬、尿酸排泄促進薬など）
  * NSAIDs（非ステロイド性抗炎症薬）
  * 下剤（刺激性下剤、浸透圧性下剤など）
  * 漢方（漢方薬）
  * 睡眠薬（ベンゾジアゼピン系、非ベンゾジアゼピン系、オレキシン受容体拮抗薬など）
  * 抗うつ薬（SSRI、SNRI、三環系など）
  * 抗アレルギー薬（抗ヒスタミン薬、ロイコトリエン受容体拮抗薬など）
  * カルシウム拮抗薬
  * その他（適切な分類がない場合）
"""
    
    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """AIレスポンスをJSONとして解析"""
        try:
            # JSONブロックを抽出
            if "```json" in ai_response:
                json_start = ai_response.find("```json") + 7
                json_end = ai_response.find("```", json_start)
                if json_end != -1:
                    json_text = ai_response[json_start:json_end].strip()
                else:
                    json_text = ai_response[json_start:].strip()
            elif "```" in ai_response:
                json_start = ai_response.find("```") + 3
                json_end = ai_response.find("```", json_start)
                if json_end != -1:
                    json_text = ai_response[json_start:json_end].strip()
                else:
                    json_text = ai_response[json_start:].strip()
            else:
                json_text = ai_response.strip()
            
            # JSON解析
            parsed_data = json.loads(json_text)
            
            # データ検証
            if not isinstance(parsed_data, dict) or 'drugs' not in parsed_data:
                raise ValueError("Invalid JSON structure: missing 'drugs' key")
            
            if not isinstance(parsed_data['drugs'], list):
                raise ValueError("Invalid JSON structure: 'drugs' must be a list")
            
            # 各薬剤データの検証と正規化
            validated_drugs = []
            for drug in parsed_data['drugs']:
                if isinstance(drug, dict):
                    validated_drug = self._validate_drug_data(drug)
                    if validated_drug:
                        validated_drugs.append(validated_drug)
            
            return {'drugs': validated_drugs}
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"AI response: {ai_response}")
            return {'drugs': []}
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return {'drugs': []}
    
    def _validate_drug_data(self, drug: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """薬剤データの検証と正規化"""
        try:
            # 必須フィールドのチェック
            if not drug.get('generic') or not drug.get('generic').strip():
                return None
            
            # データの正規化
            validated_drug = {
                'raw': str(drug.get('raw', '')).strip(),
                'generic': str(drug.get('generic', '')).strip(),
                'strength': str(drug.get('strength', '')).strip(),
                'dose': str(drug.get('dose', '')).strip(),
                'freq': str(drug.get('freq', '')).strip(),
                'days': self._parse_days(drug.get('days')),
                'category': str(drug.get('category', '')).strip()
            }
            
            return validated_drug
            
        except Exception as e:
            logger.warning(f"Drug data validation error: {e}")
            return None
    
    def _parse_days(self, days_value: Any) -> Optional[int]:
        """日数の解析"""
        if days_value is None:
            return None
        
        try:
            if isinstance(days_value, (int, float)):
                return int(days_value)
            elif isinstance(days_value, str):
                # 文字列から数値を抽出
                import re
                numbers = re.findall(r'\d+', days_value)
                if numbers:
                    return int(numbers[0])
            return None
        except (ValueError, TypeError):
            return None
    
    def _evaluate_confidence(self, extracted_data: Dict[str, Any], ocr_text: str) -> str:
        """抽出結果の信頼度を評価"""
        try:
            drugs = extracted_data.get('drugs', [])
            
            if not drugs:
                return 'low'
            
            # 信頼度の評価指標
            confidence_score = 0
            total_checks = 0
            
            for drug in drugs:
                # 一般名が存在するか
                if drug.get('generic'):
                    confidence_score += 1
                total_checks += 1
                
                # 用量が存在するか
                if drug.get('strength'):
                    confidence_score += 1
                total_checks += 1
                
                # 元テキストとの関連性
                if drug.get('raw') and drug.get('raw') in ocr_text:
                    confidence_score += 1
                total_checks += 1
            
            # 信頼度の判定（閾値を緩和）
            if total_checks == 0:
                return 'low'
            
            confidence_ratio = confidence_score / total_checks
            
            # 薬剤数が多い場合は信頼度を上げる
            drug_count_bonus = min(len(drugs) * 0.1, 0.3)  # 最大0.3のボーナス
            adjusted_ratio = confidence_ratio + drug_count_bonus
            
            if adjusted_ratio >= 0.7:  # 閾値を0.8から0.7に緩和
                return 'high'
            elif adjusted_ratio >= 0.5:  # 閾値を0.6から0.5に緩和
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.warning(f"Confidence evaluation error: {e}")
            return 'low'
    
    def generate_confirmation_message(self, extraction_result: Dict[str, Any]) -> str:
        """信頼度が低い場合の確認メッセージを生成"""
        try:
            drugs = extraction_result.get('drugs', [])
            confidence = extraction_result.get('confidence', 'low')
            
            if confidence == 'high':
                return ""
            
            message_parts = []
            
            if confidence == 'medium':
                message_parts.append("⚠️ 一部の薬剤情報が不明確です")
                message_parts.append("")
                message_parts.append("検出された薬剤:")
            else:
                message_parts.append("❌ 薬剤情報の抽出に問題があります")
                message_parts.append("")
                message_parts.append("検出された薬剤（要確認）:")
            
            for i, drug in enumerate(drugs, 1):
                drug_info = f"{i}. {drug.get('generic', '不明')}"
                if drug.get('strength'):
                    drug_info += f" {drug.get('strength')}"
                if drug.get('dose'):
                    drug_info += f" {drug.get('dose')}"
                if drug.get('freq'):
                    drug_info += f" {drug.get('freq')}"
                
                message_parts.append(drug_info)
            
            message_parts.append("")
            message_parts.append("💡 手動で薬剤名を入力することもできます:")
            message_parts.append("例: アムロジピン 5mg")
            message_parts.append("")
            message_parts.append("📋 より良い画像で撮影し直すこともお勧めします")
            
            return "\n".join(message_parts)
            
        except Exception as e:
            logger.error(f"Confirmation message generation error: {e}")
            return "薬剤情報の抽出に問題があります。手動で薬剤名を入力するか、より良い画像で撮影し直してください。"
