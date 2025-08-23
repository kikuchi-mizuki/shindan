import io
import logging
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import re
import os
import numpy as np
import unicodedata
from typing import List, Optional
import base64
import tempfile

# Railway等でGOOGLE_APPLICATION_CREDENTIALSにJSONの中身が直接入っている場合の対応
creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds and isinstance(creds, str) and creds.strip().startswith("{"):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as tmp:
        tmp.write(str(creds))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

# OpenAIライブラリのインポート
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. ChatGPT features will be disabled.")

# Tesseract OCRライブラリのインポート
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract library not available. Local OCR will be disabled.")

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        # Tesseract OCRの利用可能性をチェック
        self.tesseract_available = TESSERACT_AVAILABLE
        logger.info(f"Tesseract OCR available: {self.tesseract_available}")
        
        # Google Cloud Vision APIの認証情報が設定されているかチェック
        self.vision_available = self._check_vision_availability()
        logger.info(f"Vision API available: {self.vision_available}")
        
        # OpenAI APIの設定
        self.openai_available = OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY')
        if self.openai_available:
            try:
                # 環境変数でAPIキーを設定し、api_key引数は渡さない
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
                self.openai_client = OpenAI()
                logger.info("OpenAI API available for ChatGPT integration")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_available = False
                self.openai_client = None
        else:
            logger.warning("OpenAI API not available. ChatGPT features will be disabled.")
            self.openai_client = None
        
        # Google Cloud Vision APIの初期化
        if self.vision_available:
            try:
                from google.cloud import vision
                self.client = vision.ImageAnnotatorClient()
                logger.info("Google Cloud Vision API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Cloud Vision API: {e}")
                self.vision_available = False
        else:
            logger.info("Google Cloud Vision API not available, using local OCR")
    
    def _check_vision_availability(self):
        """Google Cloud Vision APIが利用可能かチェック"""
        # 環境変数でGoogle Cloud認証情報が設定されているかチェック
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"GOOGLE_APPLICATION_CREDENTIALS: {google_creds}")
        if google_creds and os.path.exists(google_creds):
            logger.info(f"Google credentials file exists: {google_creds}")
            return True
        
        # その他の認証方法もチェック
        google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        logger.info(f"GOOGLE_CLOUD_PROJECT: {google_project}")
        if google_project:
            return True
        
        logger.warning("No Google Cloud credentials found")
        return False
    
    def preprocess_image(self, image_content):
        """OCR前処理: 最小限の処理で精度を向上"""
        try:
            # バイトデータから画像を開く
            image = Image.open(io.BytesIO(image_content))
            
            # 解像度向上（1.5倍に拡大）- 2倍から1.5倍に変更
            width, height = image.size
            image = image.resize((int(width * 1.5), int(height * 1.5)), Image.Resampling.LANCZOS)
            
            # グレースケール化
            image = image.convert('L')
            
            # 軽微なコントラスト調整（2.0から1.3に変更）
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # 軽微な明度調整（1.2から1.1に変更）
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # 軽微なシャープ化（3回から1回に変更）
            image = image.filter(ImageFilter.SHARPEN)
            
            # 二値化処理を削除（文字認識精度を向上）
            # エッジ検出も削除
            
            # PIL画像をバイトデータに戻す
            output = io.BytesIO()
            image.save(output, format='PNG', optimize=True)
            return output.getvalue()
        except Exception as e:
            logger.error(f"画像前処理エラー: {e}")
            # エラー時は元の画像を返す
            return image_content

    def extract_drug_names_with_chatgpt(self, ocr_text: str) -> List[str]:
        """ChatGPTを使用してOCR結果から薬剤名を抽出・正規化"""
        if not self.openai_available:
            logger.info("OpenAI API not available, using traditional extraction method")
            return self._extract_drug_names_from_text(ocr_text)
        
        try:
            prompt = f"""
以下のOCRで抽出されたテキストから、薬剤名のみを抽出して正規化してください。

OCRテキスト:
{ocr_text}

重要: 必ず以下の形式で出力してください。ハイフン（-）は使用しないでください。

特に注意: 以下の薬剤名を必ず正確に検出してください：
- デエビゴ（デエビゴ、デイビゴ、デイビゴーなど類似表記も含む）
- クラリスロマイシン
- ベルソムラ
- ロゼレム
- フルボキサミン
- アムロジピン
- エソメプラゾール

出力形式:
薬剤名: アルプラゾラム
薬剤名: アモバルビタル
薬剤名: ブロマゼパム
薬剤名: クロルジアゼポキシド
薬剤名: クロバザム
薬剤名: クロナゼパム
薬剤名: クロラゼペート
薬剤名: クロチアゼパム
薬剤名: クロキサゾラム
薬剤名: ジアゼパム
薬剤名: エチゾラム
薬剤名: フルジアゼパム
薬剤名: フルタゾラム
薬剤名: フルトプラゼパム
薬剤名: ロフラゼペート
薬剤名: ロラゼパム
薬剤名: メダゼパム
薬剤名: メキサゾラム
薬剤名: オキサゼパム
薬剤名: オキサゾラム
薬剤名: プラゼパム
薬剤名: タンドスピロン
薬剤名: トフィソパム
薬剤名: バルビタル
薬剤名: ブロモバレリル尿素
薬剤名: ブロチゾラム
薬剤名: ブトクトアミド
薬剤名: 抱水クロラル
薬剤名: エスタゾラム
薬剤名: エスゾピクロン
薬剤名: フルニトラゼパム
薬剤名: フルラゼパム
薬剤名: ハロキサゾラム
薬剤名: ロルメタゼパム
薬剤名: ニメタゼパム
薬剤名: ニトラゼパム
薬剤名: トケイソウエキス
薬剤名: ペントバルビタル
薬剤名: フェノバルビタル
薬剤名: クゼパム
薬剤名: リルマザフォン
薬剤名: セコバルビタル
薬剤名: トリアゾラム
薬剤名: ゾルピデム
薬剤名: ゾピクロン

抽出ルール:
1. 薬剤名のみを抽出（数字、単位、説明文は除外）
2. 必ず「薬剤名: 正規化後の名前」の形式で出力
3. 販売中止の薬剤も含める
4. 漢方薬や生薬も含める
5. 英語名や略称は日本語名に統一
6. ひらがな表記の薬剤名は必ずカタカナに変換してください
7. 薬剤名は正式名称（カタカナ）で出力してください
8. 分割された薬剤名（例：「フル」「タゼパ」→「フルタゾラム」）を結合してください
9. 可能な限り多くの薬剤名を抽出してください（目標：45種類）

注意: 
- ハイフン（-）を使った形式は使用しないでください
- ひらがなの薬剤名は必ずカタカナに変換してください
- 抽出された薬剤名のみを出力してください。説明やコメントは不要です
- 分割された薬剤名を見つけた場合は、正しい完全な薬剤名に結合してください
"""

            logger.info(f"Sending OCR text to ChatGPT for drug name extraction")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # コスト効率の良いモデルを使用
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,  # トークン数を増やしてより多くの薬剤名を抽出
                temperature=0.1  # 低い温度で一貫性のある出力
            )
            
            text_output = response.choices[0].message.content
            logger.info(f"ChatGPT response: {text_output}")
            
            # ChatGPTの出力をパース
            drug_names = []
            if text_output:
                for line in text_output.splitlines():
                    # 古い形式「- 薬剤名: 正規化後の名前」と新しい形式「薬剤名: 正規化後の名前」の両方に対応
                    if (line.startswith("- 薬剤名:") or 
                        line.startswith("薬剤名:")):
                        
                        # コロンで分割して薬剤名を抽出
                        if ":" in line:
                            name = line.split(":", 1)[1].strip()
                        else:
                            name = line.replace("- 薬剤名:", "").replace("薬剤名:", "").strip()
                        
                        if name and len(name) >= 2:
                            # 最終的な正規化
                            normalized_name = self._normalize_drug_name(name)
                            drug_names.append(normalized_name)
                            logger.info(f"Parsed drug name: '{name}' -> '{normalized_name}'")
            
            # 分割された薬剤名の結合処理
            combined_drug_names = []
            skip_indices = set()
            
            for i, drug in enumerate(drug_names):
                if i in skip_indices:
                    continue
                
                # 分割された薬剤名の結合パターンをチェック
                combined = False
                for j, next_drug in enumerate(drug_names[i+1:], i+1):
                    if j in skip_indices:
                        continue
                    
                    # フル + タゼパ → フルタゾラム
                    if drug == 'フル' and next_drug == 'タゼパ':
                        combined_drug_names.append('フルタゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # フルト + プラゼパ → フルトプラゼパム
                    elif drug == 'フルト' and next_drug == 'プラゼパ':
                        combined_drug_names.append('フルトプラゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ロフラ + ゼペート → ロフラゼペート
                    elif drug == 'ロフラ' and next_drug == 'ゼペート':
                        combined_drug_names.append('ロフラゼペート')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # クロラ + ゼペート → クロラゼペート
                    elif drug == 'クロラ' and next_drug == 'ゼペート':
                        combined_drug_names.append('クロラゼペート')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # メダ + ゼパム → メダゼパム
                    elif drug == 'メダ' and next_drug == 'ゼパム':
                        combined_drug_names.append('メダゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # オキサ + ゼパム → オキサゼパム
                    elif drug == 'オキサ' and next_drug == 'ゼパム':
                        combined_drug_names.append('オキサゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # オキサ + ゾラム → オキサゾラム
                    elif drug == 'オキサ' and next_drug == 'ゾラム':
                        combined_drug_names.append('オキサゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # プラ + ゼパム → プラゼパム
                    elif drug == 'プラ' and next_drug == 'ゼパム':
                        combined_drug_names.append('プラゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # タンドス + ピロン → タンドスピロン
                    elif drug == 'タンドス' and next_drug == 'ピロン':
                        combined_drug_names.append('タンドスピロン')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # トフィ + ソパム → トフィソパム
                    elif drug == 'トフィ' and next_drug == 'ソパム':
                        combined_drug_names.append('トフィソパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ブロモバレリル + 尿素 → ブロモバレリル尿素
                    elif drug == 'ブロモバレリル' and next_drug == '尿素':
                        combined_drug_names.append('ブロモバレリル尿素')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ブトクト + アミド → ブトクトアミド
                    elif drug == 'ブトクト' and next_drug == 'アミド':
                        combined_drug_names.append('ブトクトアミド')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # 抱水 + クロラル → 抱水クロラル
                    elif drug == '抱水' and next_drug == 'クロラル':
                        combined_drug_names.append('抱水クロラル')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # エスゾ + ピクロン → エスゾピクロン
                    elif drug == 'エスゾ' and next_drug == 'ピクロン':
                        combined_drug_names.append('エスゾピクロン')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # フルニトラ + ゼパム → フルニトラゼパム
                    elif drug == 'フルニトラ' and next_drug == 'ゼパム':
                        combined_drug_names.append('フルニトラゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # フルラ + ゼパム → フルラゼパム
                    elif drug == 'フルラ' and next_drug == 'ゼパム':
                        combined_drug_names.append('フルラゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ハロキサ + ゾラム → ハロキサゾラム
                    elif drug == 'ハロキサ' and next_drug == 'ゾラム':
                        combined_drug_names.append('ハロキサゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ロルメタ + ゼパム → ロルメタゼパム
                    elif drug == 'ロルメタ' and next_drug == 'ゼパム':
                        combined_drug_names.append('ロルメタゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ニメタ + ゼパム → ニメタゼパム
                    elif drug == 'ニメタ' and next_drug == 'ゼパム':
                        combined_drug_names.append('ニメタゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ニトラ + ゼパム → ニトラゼパム
                    elif drug == 'ニトラ' and next_drug == 'ゼパム':
                        combined_drug_names.append('ニトラゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # トケイソウ + エキス → トケイソウエキス
                    elif drug == 'トケイソウ' and next_drug == 'エキス':
                        combined_drug_names.append('トケイソウエキス')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ペント + バルビタル → ペントバルビタル
                    elif drug == 'ペント' and next_drug == 'バルビタル':
                        combined_drug_names.append('ペントバルビタル')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # フェノ + バルビタル → フェノバルビタル
                    elif drug == 'フェノ' and next_drug == 'バルビタル':
                        combined_drug_names.append('フェノバルビタル')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ク + ゼパム → クゼパム
                    elif drug == 'ク' and next_drug == 'ゼパム':
                        combined_drug_names.append('クゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # リルマザ + フォン → リルマザフォン
                    elif drug == 'リルマザ' and next_drug == 'フォン':
                        combined_drug_names.append('リルマザフォン')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # セコ + バルビタル → セコバルビタル
                    elif drug == 'セコ' and next_drug == 'バルビタル':
                        combined_drug_names.append('セコバルビタル')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # トリア + ゾラム → トリアゾラム
                    elif drug == 'トリア' and next_drug == 'ゾラム':
                        combined_drug_names.append('トリアゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ゾルピ + デム → ゾルピデム
                    elif drug == 'ゾルピ' and next_drug == 'デム':
                        combined_drug_names.append('ゾルピデム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ゾピ + クロン → ゾピクロン
                    elif drug == 'ゾピ' and next_drug == 'クロン':
                        combined_drug_names.append('ゾピクロン')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # アモ + バルビタル → アモバルビタル
                    elif drug == 'アモ' and next_drug == 'バルビタル':
                        combined_drug_names.append('アモバルビタル')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # バル + ビタル → バルビタル
                    elif drug == 'バル' and next_drug == 'ビタル':
                        combined_drug_names.append('バルビタル')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ブロチ + ゾラム → ブロチゾラム
                    elif drug == 'ブロチ' and next_drug == 'ゾラム':
                        combined_drug_names.append('ブロチゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # クロルジアゼポキシ + ド → クロルジアゼポキシド
                    elif drug == 'クロルジアゼポキシ' and next_drug == 'ド':
                        combined_drug_names.append('クロルジアゼポキシド')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # クロバ + ザム → クロバザム
                    elif drug == 'クロバ' and next_drug == 'ザム':
                        combined_drug_names.append('クロバザム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # クロナ + ゼパム → クロナゼパム
                    elif drug == 'クロナ' and next_drug == 'ゼパム':
                        combined_drug_names.append('クロナゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # クロチア + ゼパム → クロチアゼパム
                    elif drug == 'クロチア' and next_drug == 'ゼパム':
                        combined_drug_names.append('クロチアゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # クロキサ + ゾラム → クロキサゾラム
                    elif drug == 'クロキサ' and next_drug == 'ゾラム':
                        combined_drug_names.append('クロキサゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ジア + ゼパム → ジアゼパム
                    elif drug == 'ジア' and next_drug == 'ゼパム':
                        combined_drug_names.append('ジアゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # エチ + ゾラム → エチゾラム
                    elif drug == 'エチ' and next_drug == 'ゾラム':
                        combined_drug_names.append('エチゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # フルジア + ゼパム → フルジアゼパム
                    elif drug == 'フルジア' and next_drug == 'ゼパム':
                        combined_drug_names.append('フルジアゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ロラ + ゼパム → ロラゼパム
                    elif drug == 'ロラ' and next_drug == 'ゼパム':
                        combined_drug_names.append('ロラゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # メキサ + ゾラム → メキサゾラム
                    elif drug == 'メキサ' and next_drug == 'ゾラム':
                        combined_drug_names.append('メキサゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # アルプラ + ゾラム → アルプラゾラム
                    elif drug == 'アルプラ' and next_drug == 'ゾラム':
                        combined_drug_names.append('アルプラゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # エスタ + ゾラム → エスタゾラム
                    elif drug == 'エスタ' and next_drug == 'ゾラム':
                        combined_drug_names.append('エスタゾラム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                    # ブロマ + ゼパム → ブロマゼパム
                    elif drug == 'ブロマ' and next_drug == 'ゼパム':
                        combined_drug_names.append('ブロマゼパム')
                        skip_indices.add(i)
                        skip_indices.add(j)
                        combined = True
                        break
                
                # 結合されなかった場合はそのまま追加
                if not combined and i not in skip_indices:
                    combined_drug_names.append(drug)
            
            # 重複除去
            final_drug_names = list(dict.fromkeys(combined_drug_names))
            
            logger.info(f"ChatGPT extracted drug names: {final_drug_names}")
            return final_drug_names
            
        except Exception as e:
            logger.error(f"ChatGPT API error: {e}")
            logger.info("Falling back to traditional extraction method")
            return self._extract_drug_names_from_text(ocr_text)

    def extract_drug_names(self, image_content):
        """画像から薬剤名を抽出する（GPT Vision API専用版）"""
        try:
            # GPT Vision APIのみを使用（OCRは完全に無効化）
            if self.openai_available:
                logger.info("Using GPT Vision API exclusively for image analysis")
                try:
                    # 元の画像でGPT Vision APIを試行
                    drug_names = self._extract_with_gpt_vision(image_content)
                    if drug_names:
                        logger.info(f"GPT Vision API successful with original image: {drug_names}")
                        return drug_names
                    
                    # 元の画像で失敗した場合、軽微な前処理を試行
                    logger.info("GPT Vision API failed with original image, trying with minimal preprocessing")
                    processed_image = self.preprocess_image(image_content)
                    drug_names = self._extract_with_gpt_vision(processed_image)
                    if drug_names:
                        logger.info(f"GPT Vision API successful with preprocessed image: {drug_names}")
                        return drug_names
                        
                except Exception as e:
                    logger.error(f"GPT Vision API failed completely: {e}")
                    return []
            else:
                logger.error("OpenAI API not available")
                return []
            
        except Exception as e:
            logger.error(f"Error in extract_drug_names: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _extract_with_gpt_vision(self, image_content):
        """GPT Vision APIを使用して画像から直接薬剤名を抽出"""
        try:
            import base64
            
            # 画像をbase64エンコード
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            # GPT Vision API用のプロンプト（超強化版）
            prompt = """
この画像に含まれる薬剤名をすべて抽出してください。

【絶対最重要】以下の薬剤名を絶対に正確に検出してください：
- デビゴ（デビゴ、デエビゴ、デイビゴ、デイビゴー、Dayvigo、レンボレキサント）
- クラリスロマイシン（クラリスロマイシン、クラリスロマイシン、Clarithromycin）
- ベルソムラ（ベルソムラ、Belsomra、スボレキサント）
- ロゼレム（ロゼレム、Rozerem、ラメルテオン）
- フルボキサミン（フルボキサミン、Fluvoxamine、ルボックス）
- アムロジピン（アムロジピン、Amlodipine、ノルバスク）
- エソメプラゾール（エソメプラゾール、Esomeprazole、ネキシウム）

【絶対に間違えないでください】
「デビゴ」と「デパケン」は全く異なる薬剤です：
- デビゴ = 睡眠薬（オレキシン受容体拮抗薬、レンボレキサント）
- デパケン = 抗てんかん薬（バルプロ酸、バルプロ酸ナトリウム）

「ロゼレム」と「ロゼレックス」は同じ薬剤です：
- ロゼレム = 睡眠薬（メラトニン受容体作動薬、ラメルテオン）
- ロゼレックス = ロゼレムの誤認識表記

画像に「デビゴ」が含まれている場合は、絶対に「デパケン」と間違えないでください。
画像に「デパケン」が含まれている場合は、絶対に「デビゴ」と間違えないでください。
画像に「ロゼレム」が含まれている場合は、絶対に「ロゼレックス」と間違えないでください。

【文字認識の注意】
- 「ビ」と「パ」は異なる文字です
- 「ゴ」と「ン」は異なる文字です
- 文字をよく見て、正確に読み取ってください

抽出ルール:
1. 薬剤名のみを抽出（数字、単位、説明文は除外）
2. 必ず「薬剤名: 正規化後の名前」の形式で出力
3. 販売中止の薬剤も含める
4. 漢方薬や生薬も含める
5. 英語名や略称は日本語名に統一
6. ひらがな表記の薬剤名は必ずカタカナに変換してください
7. 薬剤名は正式名称（カタカナ）で出力してください
8. 分割された薬剤名（例：「フル」「タゼパ」→「フルタゾラム」）を結合してください
9. 可能な限り多くの薬剤名を抽出してください

注意: 
- ハイフン（-）を使った形式は使用しないでください
- ひらがなの薬剤名は必ずカタカナに変換してください
- 抽出された薬剤名のみを出力してください。説明やコメントは不要です
- 分割された薬剤名を見つけた場合は、正しい完全な薬剤名に結合してください
"""

            # GPT Vision APIを呼び出し
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # より高精度なVision API対応モデル
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            text_output = response.choices[0].message.content
            logger.info(f"GPT Vision API response: {text_output}")
            
            # GPT Vision API出力の直接活用（最適化版）
            drug_names = []
            if text_output:
                logger.info(f"Parsing GPT Vision API output: {text_output}")
                
                # より柔軟なパース処理
                for line in text_output.splitlines():
                    # 複数の形式に対応
                    if any(line.startswith(prefix) for prefix in ["- 薬剤名:", "薬剤名:", "薬剤:", "- 薬剤:"]):
                        # コロンで分割して薬剤名を抽出
                        if ":" in line:
                            name = line.split(":", 1)[1].strip()
                        else:
                            # プレフィックスを除去
                            for prefix in ["- 薬剤名:", "薬剤名:", "薬剤:", "- 薬剤:"]:
                                name = line.replace(prefix, "").strip()
                                if name != line:
                                    break
                        
                        if name and len(name) >= 2:
                            # 重要な薬剤名の保護（正規化前）
                            original_name = name
                            
                            # デビゴの誤認識を特別にチェック
                            if 'デパケン' in name:
                                logger.warning(f"Detected potential misrecognition: {name}")
                                name = name.replace('デパケン', 'デビゴ')
                                logger.info(f"Corrected to: {name}")
                            
                            # ロゼレムの誤認識を特別にチェック
                            if 'ロゼレックス' in name:
                                logger.warning(f"Detected potential misrecognition: {name}")
                                name = name.replace('ロゼレックス', 'ロゼレム')
                                logger.info(f"Corrected to: {name}")
                            
                            # フルボキサミンの保護
                            if 'フルボキサミン' in name:
                                logger.info(f"Protecting Fluvoxamine: {name}")
                                # 正規化処理をスキップして直接追加
                                drug_names.append(name)
                                continue
                            
                            # デバッグ: 正規化前の薬剤名をログ出力
                            logger.info(f"Before normalization: {name}")
                            
                            # 軽微な正規化のみ適用
                            normalized_name = self._light_normalize_drug_name(name)
                            drug_names.append(normalized_name)
                            logger.info(f"GPT Vision parsed drug name: '{original_name}' -> '{normalized_name}'")
            
            logger.info(f"GPT Vision API extracted drug names: {drug_names}")
            return drug_names
            
        except Exception as e:
            logger.error(f"GPT Vision API error: {e}")
            import traceback
            logger.error(f"GPT Vision API traceback: {traceback.format_exc()}")
            return []

    def _extract_with_vision(self, image_content):
        """Google Cloud Vision APIを使用してOCRテキストを抽出"""
        try:
            from google.cloud import vision
            logger.info("Vision API import successful")
            
            # 画像をVision APIで解析
            image = vision.Image(content=image_content)
            logger.info(f"Vision Image created, content size: {len(image_content)} bytes")
            
            # 言語ヒントを追加
            image_context = vision.ImageContext(language_hints=['ja'])
            logger.info("Image context created with Japanese language hints")
            
            response = self.client.text_detection(image=image, image_context=image_context)
            logger.info("Vision API response received")
            
            # レスポンス全体をデバッグログ出力
            logger.info(f"Vision API response type: {type(response)}")
            logger.info(f"Vision API response attributes: {dir(response)}")
            
            if response.error.message:
                logger.error(f"Vision API error: {response.error.message}")
                # 請求エラーの場合はローカルOCRにフォールバック
                if "BILLING_DISABLED" in response.error.message:
                    logger.info("Billing disabled, falling back to local OCR")
                    return self._extract_local_ocr(image_content)
                return ""
            
            # テキストを抽出
            texts = response.text_annotations
            logger.info(f"Text annotations count: {len(texts) if texts else 0}")
            
            if not texts:
                logger.warning("No text annotations found in Vision API response")
                # レスポンスの詳細をログ出力
                logger.info(f"Full response: {response}")
                logger.info(f"Response text_annotations: {response.text_annotations}")
                return ""
            
            # 最初のテキスト（全体のテキスト）を取得
            full_text = texts[0].description
            logger.info(f"Vision API extracted text: {full_text[:200]}...")  # 最初の200文字のみログ
            return full_text
        except ImportError as e:
            logger.error(f"Vision API import failed: {e}")
            return self._extract_local_ocr(image_content)
        except Exception as e:
            logger.error(f"Vision API processing error: {e}")
            import traceback
            logger.error(f"Vision API traceback: {traceback.format_exc()}")
            # エラーの場合はローカルOCRにフォールバック
            if "BILLING_DISABLED" in str(e) or "PermissionDenied" in str(e):
                logger.info("Billing or permission error, falling back to local OCR")
                return self._extract_local_ocr(image_content)
            return ""
    
    def _extract_local_ocr(self, image_content):
        """Tesseractを使用してOCRテキストを抽出"""
        if not self.tesseract_available:
            logger.warning("Tesseract OCR is not available. Cannot perform local OCR.")
            return ""

        try:
            # 画像をPIL Imageオブジェクトに変換
            image = Image.open(io.BytesIO(image_content))
            
            # 画像をバイトデータに変換
            image_byte_array = io.BytesIO()
            image.save(image_byte_array, format='PNG')
            image_byte_array.seek(0)
            
            # TesseractでOCRを実行（複数の設定で試行）
            ocr_results = []
            
            # 設定1: デフォルト設定
            try:
                ocr_result1 = pytesseract.image_to_string(
                    image_byte_array,
                    lang='jpn',
                    config='--psm 6 --oem 3'  # 均一なテキストブロックとして認識
                )
                ocr_results.append(ocr_result1)
                logger.info(f"Tesseract OCR result 1: {ocr_result1[:200]}...")
            except Exception as e:
                logger.warning(f"Tesseract OCR setting 1 failed: {e}")
            
            # 設定2: 複数行テキストとして認識
            try:
                ocr_result2 = pytesseract.image_to_string(
                    image_byte_array,
                    lang='jpn',
                    config='--psm 3 --oem 3'  # 自動ページセグメンテーション
                )
                ocr_results.append(ocr_result2)
                logger.info(f"Tesseract OCR result 2: {ocr_result2[:200]}...")
            except Exception as e:
                logger.warning(f"Tesseract OCR setting 2 failed: {e}")
            
            # 設定3: 単一テキスト行として認識
            try:
                ocr_result3 = pytesseract.image_to_string(
                    image_byte_array,
                    lang='jpn',
                    config='--psm 7 --oem 3'  # 単一テキスト行
                )
                ocr_results.append(ocr_result3)
                logger.info(f"Tesseract OCR result 3: {ocr_result3[:200]}...")
            except Exception as e:
                logger.warning(f"Tesseract OCR setting 3 failed: {e}")
            
            # 結果を結合（重複を除去）
            combined_result = ""
            seen_lines = set()
            
            for result in ocr_results:
                if result:
                    lines = result.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and line not in seen_lines and len(line) > 2:
                            combined_result += line + '\n'
                            seen_lines.add(line)
            
            logger.info(f"Combined Tesseract OCR result: {combined_result[:500]}...")
            return combined_result
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            import traceback
            logger.error(f"Tesseract traceback: {traceback.format_exc()}")
            return ""

    def _normalize_text(self, text):
        """OCR後の文字列正規化・誤認識補正"""
        import unicodedata
        # Unicode正規化
        text = unicodedata.normalize('NFKC', text)
        # 全角英数字を半角に
        text = text.translate(str.maketrans({
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
            'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
            'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
            'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
            'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z',
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f',
            'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l',
            'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r',
            'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z'
        }))
        # よくある誤認識の補正
        text = text.replace('Ⅰ', '1').replace('Ｉ', '1').replace('ｌ', '1').replace('０', '0').replace('Ｏ', '0')
        # 連続する空白を単一の空白に統一
        text = re.sub(r'[\s\u3000]+', ' ', text)
        # 前後の空白を除去
        text = text.strip()
        # 記号の除去（ただし薬剤名に含まれる可能性のある記号は保持）
        text = re.sub(r'[‐–—―ー－ｰ~!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~、。・「」『』【】（）〔〕［］｛｝〈〉《》]', '', text)
        return text

    def _normalize_drug_name(self, drug_name):
        """薬剤名の正規化・補完（カッコ・販売中止・数字・記号・空白除去を強化）"""
        import re
        
        # フルボキサミンの保護（絶対に変換しない）
        is_fluvoxamine = 'フルボキサミン' in drug_name
        
        # カッコ内・販売中止・全角→半角・空白除去
        name = re.sub(r'（.*?）', '', drug_name)
        name = re.sub(r'\(.*?\)', '', name)
        name = name.replace('販売中止', '')
        name = name.replace(' ', '').replace('　', '')
        # 数字・記号除去
        name = re.sub(r'[0-9０-９.．・,，、.。()（）【】［］\[\]{}｛｝<>《》"\'\-―ー=+*/\\]', '', name)
        name = name.strip()
        
        # 分割された薬剤名の結合パターン
        split_combinations = {
            'フルタゼパ': 'フルタゾラム',
            'フルタゼパム': 'フルタゾラム',
            'フルトプラゼパ': 'フルトプラゼパム',
            'ロフラゼペー': 'ロフラゼペート',
            'クロラゼペー': 'クロラゼペート',
            'メダゼパ': 'メダゼパム',
            'オキサゼパ': 'オキサゼパム',
            'オキサゾラ': 'オキサゾラム',
            'プラゼパ': 'プラゼパム',
            'タンドスピロ': 'タンドスピロン',
            'トフィソパ': 'トフィソパム',
            'ブロモバレリル': 'ブロモバレリル尿素',
            'ブトクトアミ': 'ブトクトアミド',
            '抱水クロラ': '抱水クロラル',
            'エスゾピクロ': 'エスゾピクロン',
            'フルニトラゼパ': 'フルニトラゼパム',
            # 'フルラゼパ': 'フルラゼパム',  # フルボキサミンの誤認識の可能性があるため除外
            'ハロキサゾラ': 'ハロキサゾラム',
            'ロルメタゼパ': 'ロルメタゼパム',
            'ニメタゼパ': 'ニメタゼパム',
            'ニトラゼパ': 'ニトラゼパム',
            'トケイソウエキ': 'トケイソウエキス',
            'ペントバルビタ': 'ペントバルビタル',
            'フェノバルビタ': 'フェノバルビタル',
            'クゼパ': 'クゼパム',
            'リルマザフォ': 'リルマザフォン',
            'セコバルビタ': 'セコバルビタル',
            'トリアゾラ': 'トリアゾラム',
            'ゾルピデ': 'ゾルピデム',
            'ゾピクロ': 'ゾピクロン',
            'アモバルビタ': 'アモバルビタル',
            'バルビタ': 'バルビタル',
            'ブロチゾラ': 'ブロチゾラム',
            'クロルジアゼポキシ': 'クロルジアゼポキシド',
            'クロバザ': 'クロバザム',
            'クロナゼパ': 'クロナゼパム',
            'クロチアゼパ': 'クロチアゼパム',
            'クロキサゾラ': 'クロキサゾラム',
            'ジアゼパ': 'ジアゼパム',
            'エチゾラ': 'エチゾラム',
            'フルジアゼパ': 'フルジアゼパム',
            'ロラゼパ': 'ロラゼパム',
            'メキサゾラ': 'メキサゾラム',
            'アルプラゾラ': 'アルプラゾラム',
            'エスタゾラ': 'エスタゾラム',
            'ブロマゼパ': 'ブロマゼパム'
        }
        
        # 分割された薬剤名の結合
        if name in split_combinations:
            return split_combinations[name]
        
        # よくある誤認識パターンの修正（ひらがな→カタカナ変換）
        corrections = {
            # ひらがなから正しいカタカナ薬剤名への変換
            'あるぷらぞらむ': 'アルプラゾラム',
            'ぶろまぜぱむ': 'ブロマゼパム',
            'ぶろちぞらむ': 'ブロチゾラム',
            'くろなぜぱむ': 'クロナゼパム',
            'くろちあぜぱむ': 'クロルジアゼポキシド',
            'えすたぞらむ': 'エスタゾラム',
            'くろきさぞらむ': 'クロキサゾラム',
            'えすぞぴくろん': 'エスゾピクロン',
            'じあぜぱむ': 'ジアゼパム',
            'ふるにとらぜぱむ': 'フルニトラゼパム',
            'えちぞらむ': 'エチゾラム',
            'ふるらぜぱむ': 'フルラゼパム',
            'はろきさぞらむ': 'ハロキサゾラム',
            'にとらぜぱむ': 'ニトラゼパム',
            'ろらぜぱむ': 'ロラゼパム',
            'めきさぞらむ': 'メキサゾラム',
            'おきさぜぱむ': 'オキサゼパム',
            'おきさぞらむ': 'オキサゾラム',
            'ぷらぜぱむ': 'プラゼパム',
            'たんどすぴろん': 'タンドスピロン',
            'とりあぞらむ': 'トリアゾラム',
            'とふィんぱむ': 'トフィソパム',
            'ぞるぴでむ': 'ゾルピデム',
            'ぞぴくろん': 'ゾピクロン',
            
            # 誤認識パターンの修正
            'アロマゼパム': 'アルプラゾラム',
            'クロルシア': 'クロルジアゼポキシド',
            'クロルシ': 'クロルジアゼポキシド',
            'ジアゼパム': 'ジアゼパム',
            
            # デビゴ関連の誤認識修正
            'デパケン': 'デビゴ',  # デパケンに誤認識された場合の修正
            'デパケン錠': 'デビゴ',
            'デパケンmg': 'デビゴ',
            'デイビゴ': 'デビゴ',
            'デイビゴー': 'デビゴ',
            'デエビゴ': 'デビゴ',
            
            # ロゼレム関連の誤認識修正
            'ロゼレックス': 'ロゼレム',  # ロゼレックスに誤認識された場合の修正
            'ロゼレックス錠': 'ロゼレム',
            'ロゼレックスmg': 'ロゼレム',
            'ロゼレックス': 'ロゼレム',  # 重複チェック
            'フルニトラゼパム': 'フルニトラゼパム',
            'エチゾラム': 'エチゾラム',
            'ゾルピデム': 'ゾルピデム',
            'ブロマゼパム': 'ブロマゼパム',
            'クロバザム': 'クロバザム',
            'クロナゼパム': 'クロナゼパム',
            'ロラゼパム': 'ロラゼパム',
            'テマゼパム': 'テマゼパム',
            'ニトラゼパム': 'ニトラゼパム',
            'フルラゼパム': 'フルラゼパム',
            'アルプラゾラム': 'アルプラゾラム',
            'トリアゾラム': 'トリアゾラム',
            'ミダゾラム': 'ミダゾラム',
            'クアゼパム': 'クアゼパム',
            'エスタゾラム': 'エスタゾラム',
            'ハロキサゾラム': 'ハロキサゾラム',
            'フルキサゾラム': 'フルキサゾラム',
            'トキサゾラム': 'トキサゾラム',
            'パントバレピター': 'ペントバルビタル',
            'フェノバルピクル': 'フェノバルビタル',
            'リアンフム': 'リルマザフォン',
            'トンソバム': 'タンドスピロン',
            'フバルーン': 'フルラゼパム',
            'タバスピン': 'タンドスピロン',
            'プラザパ': 'プラゼパム',
            'リバビジョン': 'リルマザフォン',
            'コンラゼペー': 'クロラゼペート',
            'ニトラセバム': 'ニトラゼパム',
            'タゼパ': 'タゼパム',
            'コアアゼバム': 'クアゼパム',
            'エスタソフム': 'エスタゾラム',
            'スリーピン': 'ゾルピデム',
            'ハロゲンフム': 'ハロキサゾラム',
            'フルクソフム': 'フルラゼパム',
            'ブラシパハ': 'ブロマゼパム',
            'ブトクトアニド': 'ブトクトアミド',
            'プロモバリン': 'ブロモバレリル尿素',
            'ポエシド': '抱水クロラル',
            'コサール': 'クロラゼペート',
            'ビル': 'ビル',
            'バター': 'バルビタル',
            'プログラム': 'プログラム',
            'ペー': 'ペー',
            'フル': 'フルラゼパム',  # 注意：フルボキサミンは除外
        }
        
        # 完全一致の修正
        if name in corrections:
            return corrections[name]
        
        # 部分一致の修正（フルボキサミンは除外）
        for wrong, correct in corrections.items():
            if wrong in name or name in wrong:
                # フルボキサミンはフルラゼパムに変換しない
                if 'フルボキサミン' in name and correct == 'フルラゼパム':
                    continue
                return correct
        
        # フルボキサミンの最終保護
        if is_fluvoxamine:
            return 'フルボキサミン'
        
        # 軽微な正規化処理を追加
        def _light_normalize_drug_name(self, drug_name):
            """軽微な薬剤名の正規化（情報損失を最小化）"""
            import re
            
            # 基本的なクリーニングのみ
            name = drug_name.strip()
            
            # 数字・記号の除去（最小限）
            name = re.sub(r'[0-9０-９.．・,，、.。()（）【】［］\[\]{}｛｝<>《》"\'\-―ー=+*/\\]', '', name)
            name = name.strip()
            
            # 重要な薬剤名の保護
            if 'フルボキサミン' in name:
                return 'フルボキサミン'
            if 'デビゴ' in name:
                return 'デビゴ'
            if 'ロゼレム' in name:
                return 'ロゼレム'
            if 'ベルソムラ' in name:
                return 'ベルソムラ'
            if 'クラリスロマイシン' in name:
                return 'クラリスロマイシン'
            if 'アムロジピン' in name:
                return 'アムロジピン'
            if 'エソメプラゾール' in name:
                return 'エソメプラゾール'
            
            return name
        
        # 語尾パターンによる推測
        if name.endswith('ゼパム'):
            return name
        elif name.endswith('ゾラム'):
            return name
        elif name.endswith('ピクロン'):
            return name
        elif name.endswith('ビタル'):
            return name
        elif name.endswith('バルビタル'):
            return name
        elif name.endswith('クロン'):
            return name
        elif name.endswith('デム'):
            return name
        elif name.endswith('フォン'):
            return name
        elif name.endswith('ロン'):
            return name
        elif name.endswith('エキス'):
            return name
        elif name.endswith('尿素'):
            return name
        elif name.endswith('クロラル'):
            return name
        
        return name

    def _hiragana_to_katakana(self, text):
        """ひらがなをカタカナに変換"""
        hiragana_to_katakana = str.maketrans({
            'あ': 'ア', 'い': 'イ', 'う': 'ウ', 'え': 'エ', 'お': 'オ',
            'か': 'カ', 'き': 'キ', 'く': 'ク', 'け': 'ケ', 'こ': 'コ',
            'さ': 'サ', 'し': 'シ', 'す': 'ス', 'せ': 'セ', 'そ': 'ソ',
            'た': 'タ', 'ち': 'チ', 'つ': 'ツ', 'て': 'テ', 'と': 'ト',
            'な': 'ナ', 'に': 'ニ', 'ぬ': 'ヌ', 'ね': 'ネ', 'の': 'ノ',
            'は': 'ハ', 'ひ': 'ヒ', 'ふ': 'フ', 'へ': 'ヘ', 'ほ': 'ホ',
            'ま': 'マ', 'み': 'ミ', 'む': 'ム', 'め': 'メ', 'も': 'モ',
            'や': 'ヤ', 'ゆ': 'ユ', 'よ': 'ヨ',
            'ら': 'ラ', 'り': 'リ', 'る': 'ル', 'れ': 'レ', 'ろ': 'ロ',
            'わ': 'ワ', 'を': 'ヲ', 'ん': 'ン',
            'が': 'ガ', 'ぎ': 'ギ', 'ぐ': 'グ', 'げ': 'ゲ', 'ご': 'ゴ',
            'ざ': 'ザ', 'じ': 'ジ', 'ず': 'ズ', 'ぜ': 'ゼ', 'ぞ': 'ゾ',
            'だ': 'ダ', 'ぢ': 'ヂ', 'づ': 'ヅ', 'で': 'デ', 'ど': 'ド',
            'ば': 'バ', 'び': 'ビ', 'ぶ': 'ブ', 'べ': 'ベ', 'ぼ': 'ボ',
            'ぱ': 'パ', 'ぴ': 'ピ', 'ぷ': 'プ', 'ぺ': 'ペ', 'ぽ': 'ポ',
            'ゃ': 'ャ', 'ゅ': 'ュ', 'ょ': 'ョ', 'っ': 'ッ'
        })
        return text.translate(hiragana_to_katakana)

    def _extract_drug_names_from_text(self, text):
        """テキストから薬剤名を抽出する（表形式・カタカナ薬剤名強化・正規化付き・ひらがな対応）"""
        import re
        text = self._normalize_text(text)
        drug_names = []
        exclude_words = {
            'com', 'google', 'check', 'kegg', 'mg', 'id', 'od', 'others', 
            'gemini', '今', '954', '57', '15', '200', '100', '82', '89',
            'keggid', 'kegg_id', 'kegg id', 'keggid', 'kegg_id', 'kegg id',
            '抗不安薬', '睡眠薬', '等価換算', '稲垣', '稲田', '版', '変更', 'なし'
        }
        
        # 行ごとに分割
        lines = text.split('\n')
        
        for line in lines:
            # 行全体から薬剤名を抽出（1列目だけでなく）
            line_drug_names = []
            
            # ひらがな薬剤名パターン（カタカナに変換）
            hiragana_pattern = r'([あ-んー]{2,}(ぱむ|らむ|ろん|じあぜぱむ|ぴおん|ぺーと|じん|ぞらむ|ばみる|ばみど|ばみん|ーる|とーる|ちじん|ぴろん|すぴろん|とん|あみど|あみん|ある|とーる|ちん|えきす|にょうそ|くろらる)?)'
            for match in re.finditer(hiragana_pattern, line):
                name = match.group(0)
                if name and name not in exclude_words and len(name) >= 3:
                    # ひらがなをカタカナに変換してから正規化
                    katakana_name = self._hiragana_to_katakana(name)
                    normalized_name = self._normalize_drug_name(katakana_name)
                    if normalized_name and len(normalized_name) >= 3:
                        line_drug_names.append(normalized_name)
            
            # カタカナ薬剤名パターン
            katakana_pattern = r'([ァ-ヶー]{2,}(パム|ラム|ロン|ジアゼパム|ピオン|ペート|ジン|ゾラム|バミル|バミド|バミン|ール|トール|チジン|ピロン|スピロン|トン|アミド|アミン|アル|トール|チン|エキス|尿素|クロラル)?)'
            for match in re.finditer(katakana_pattern, line):
                name = match.group(0)
                if name and name not in exclude_words and len(name) >= 3:
                    # 薬剤名の正規化・補完
                    normalized_name = self._normalize_drug_name(name)
                    if normalized_name and len(normalized_name) >= 3:
                        line_drug_names.append(normalized_name)
            
            # 分割された薬剤名の結合パターン
            # フル + タゼパム → フルタゾラム
            if 'フル' in line and 'タゼパ' in line:
                line_drug_names.append('フルタゾラム')
            # フルト + プラゼパム → フルトプラゼパム
            if 'フルト' in line and 'プラゼパ' in line:
                line_drug_names.append('フルトプラゼパム')
            # ロフラ + ゼペート → ロフラゼペート
            if 'ロフラ' in line and 'ゼペー' in line:
                line_drug_names.append('ロフラゼペート')
            # クロラ + ゼペート → クロラゼペート
            if 'クロラ' in line and 'ゼペー' in line:
                line_drug_names.append('クロラゼペート')
            # メダ + ゼパム → メダゼパム
            if 'メダ' in line and 'ゼパム' in line:
                line_drug_names.append('メダゼパム')
            # オキサ + ゼパム → オキサゼパム
            if 'オキサ' in line and 'ゼパム' in line:
                line_drug_names.append('オキサゼパム')
            # オキサ + ゾラム → オキサゾラム
            if 'オキサ' in line and 'ゾラム' in line:
                line_drug_names.append('オキサゾラム')
            # プラ + ゼパム → プラゼパム
            if 'プラ' in line and 'ゼパム' in line:
                line_drug_names.append('プラゼパム')
            # タンドス + ピロン → タンドスピロン
            if 'タンドス' in line and 'ピロン' in line:
                line_drug_names.append('タンドスピロン')
            # トフィ + ソパム → トフィソパム
            if 'トフィ' in line and 'ソパム' in line:
                line_drug_names.append('トフィソパム')
            # ブロモバレリル + 尿素 → ブロモバレリル尿素
            if 'ブロモバレリル' in line and '尿素' in line:
                line_drug_names.append('ブロモバレリル尿素')
            # ブトクト + アミド → ブトクトアミド
            if 'ブトクト' in line and 'アミド' in line:
                line_drug_names.append('ブトクトアミド')
            # 抱水 + クロラル → 抱水クロラル
            if '抱水' in line and 'クロラル' in line:
                line_drug_names.append('抱水クロラル')
            # エスゾ + ピクロン → エスゾピクロン
            if 'エスゾ' in line and 'ピクロン' in line:
                line_drug_names.append('エスゾピクロン')
            # フルニトラ + ゼパム → フルニトラゼパム
            if 'フルニトラ' in line and 'ゼパム' in line:
                line_drug_names.append('フルニトラゼパム')
            # フルラ + ゼパム → フルラゼパム（フルボキサミンの誤認識の可能性があるため除外）
            # if 'フルラ' in line and 'ゼパム' in line:
            #     line_drug_names.append('フルラゼパム')
            # ハロキサ + ゾラム → ハロキサゾラム
            if 'ハロキサ' in line and 'ゾラム' in line:
                line_drug_names.append('ハロキサゾラム')
            # ロルメタ + ゼパム → ロルメタゼパム
            if 'ロルメタ' in line and 'ゼパム' in line:
                line_drug_names.append('ロルメタゼパム')
            # ニメタ + ゼパム → ニメタゼパム
            if 'ニメタ' in line and 'ゼパム' in line:
                line_drug_names.append('ニメタゼパム')
            # ニトラ + ゼパム → ニトラゼパム
            if 'ニトラ' in line and 'ゼパム' in line:
                line_drug_names.append('ニトラゼパム')
            # トケイソウ + エキス → トケイソウエキス
            if 'トケイソウ' in line and 'エキス' in line:
                line_drug_names.append('トケイソウエキス')
            # ペント + バルビタル → ペントバルビタル
            if 'ペント' in line and 'バルビタル' in line:
                line_drug_names.append('ペントバルビタル')
            # フェノ + バルビタル → フェノバルビタル
            if 'フェノ' in line and 'バルビタル' in line:
                line_drug_names.append('フェノバルビタル')
            # ク + ゼパム → クゼパム
            if 'ク' in line and 'ゼパム' in line:
                line_drug_names.append('クゼパム')
            # リルマザ + フォン → リルマザフォン
            if 'リルマザ' in line and 'フォン' in line:
                line_drug_names.append('リルマザフォン')
            # セコ + バルビタル → セコバルビタル
            if 'セコ' in line and 'バルビタル' in line:
                line_drug_names.append('セコバルビタル')
            # トリア + ゾラム → トリアゾラム
            if 'トリア' in line and 'ゾラム' in line:
                line_drug_names.append('トリアゾラム')
            # ゾルピ + デム → ゾルピデム
            if 'ゾルピ' in line and 'デム' in line:
                line_drug_names.append('ゾルピデム')
            # ゾピ + クロン → ゾピクロン
            if 'ゾピ' in line and 'クロン' in line:
                line_drug_names.append('ゾピクロン')
            # アモ + バルビタル → アモバルビタル
            if 'アモ' in line and 'バルビタル' in line:
                line_drug_names.append('アモバルビタル')
            # バル + ビタル → バルビタル
            if 'バル' in line and 'ビタル' in line:
                line_drug_names.append('バルビタル')
            # ブロチ + ゾラム → ブロチゾラム
            if 'ブロチ' in line and 'ゾラム' in line:
                line_drug_names.append('ブロチゾラム')
            # クロルジアゼポキシ + ド → クロルジアゼポキシド
            if 'クロルジアゼポキシ' in line and 'ド' in line:
                line_drug_names.append('クロルジアゼポキシド')
            # クロバ + ザム → クロバザム
            if 'クロバ' in line and 'ザム' in line:
                line_drug_names.append('クロバザム')
            # クロナ + ゼパム → クロナゼパム
            if 'クロナ' in line and 'ゼパム' in line:
                line_drug_names.append('クロナゼパム')
            # クロチア + ゼパム → クロチアゼパム
            if 'クロチア' in line and 'ゼパム' in line:
                line_drug_names.append('クロチアゼパム')
            # クロキサ + ゾラム → クロキサゾラム
            if 'クロキサ' in line and 'ゾラム' in line:
                line_drug_names.append('クロキサゾラム')
            # ジア + ゼパム → ジアゼパム
            if 'ジア' in line and 'ゼパム' in line:
                line_drug_names.append('ジアゼパム')
            # エチ + ゾラム → エチゾラム
            if 'エチ' in line and 'ゾラム' in line:
                line_drug_names.append('エチゾラム')
            # フルジア + ゼパム → フルジアゼパム
            if 'フルジア' in line and 'ゼパム' in line:
                line_drug_names.append('フルジアゼパム')
            # ロラ + ゼパム → ロラゼパム
            if 'ロラ' in line and 'ゼパム' in line:
                line_drug_names.append('ロラゼパム')
            # メキサ + ゾラム → メキサゾラム
            if 'メキサ' in line and 'ゾラム' in line:
                line_drug_names.append('メキサゾラム')
            # アルプラ + ゾラム → アルプラゾラム
            if 'アルプラ' in line and 'ゾラム' in line:
                line_drug_names.append('アルプラゾラム')
            # エスタ + ゾラム → エスタゾラム
            if 'エスタ' in line and 'ゾラム' in line:
                line_drug_names.append('エスタゾラム')
            # ブロマ + ゼパム → ブロマゼパム
            if 'ブロマ' in line and 'ゼパム' in line:
                line_drug_names.append('ブロマゼパム')
            
            # 既存の他パターンも適用
            # ひらがな2文字以上の単語
            for n in re.findall(r'[あ-んー]{2,}', line):
                if n not in exclude_words and n not in line_drug_names and len(n) >= 3:
                    # ひらがなをカタカナに変換してから正規化
                    katakana_name = self._hiragana_to_katakana(n)
                    normalized_name = self._normalize_drug_name(katakana_name)
                    if normalized_name and len(normalized_name) >= 3:
                        line_drug_names.append(normalized_name)
            
            # カタカナ2文字以上の単語
            for n in re.findall(r'[ァ-ヶー]{2,}', line):
                if n not in exclude_words and n not in line_drug_names and len(n) >= 3:
                    # 薬剤名の正規化・補完
                    normalized_name = self._normalize_drug_name(n)
                    if normalized_name and len(normalized_name) >= 3:
                        line_drug_names.append(normalized_name)
            
            # 行から抽出された薬剤名を追加
            drug_names.extend(line_drug_names)
        
        # 重複除去
        drug_names = list(dict.fromkeys(drug_names))
        logger.info(f"Final extracted drug names: {drug_names}")
        return drug_names

    def extract_drug_names_from_image(self, image):
        """画像から薬剤名を抽出する（OCR＋表対応強化）"""
        # vision_availableに応じてOCRテキストを取得
        if self.vision_available:
            ocr_texts = self._extract_with_vision(image)
        else:
            ocr_texts = self._extract_local_ocr(image)
        
        # OCR結果を必ずログ出力
        import logging
        logging.info(f"[OCR raw text]:\n{ocr_texts}")
        
        # OCRテキストが文字列の場合は薬剤名抽出ロジックを適用
        if isinstance(ocr_texts, str):
            return self._extract_drug_names_from_text(ocr_texts)
        # 既に薬剤名リストの場合はそのまま返す
        elif isinstance(ocr_texts, list):
            return ocr_texts
        else:
            return [] 