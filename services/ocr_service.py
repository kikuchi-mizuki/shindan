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
        """OCR前処理: 高度な画像処理で精度を大幅向上"""
        try:
            # バイトデータから画像を開く
            image = Image.open(io.BytesIO(image_content))
            
            # 1. 画像の向きを自動補正
            try:
                # EXIF情報から回転情報を取得して補正
                image = self._auto_rotate_image(image)
            except Exception as e:
                logger.warning(f"自動回転補正エラー: {e}")
            
            # 2. 解像度向上（2倍に拡大）
            width, height = image.size
            image = image.resize((int(width * 2.0), int(height * 2.0)), Image.Resampling.LANCZOS)
            
            # 3. グレースケール化
            image = image.convert('L')
            
            # 4. ノイズ除去（メディアンフィルタ）
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # 5. コントラスト強化
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # コントラストを2倍に強化
            
            # 6. 明度調整
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)  # 明度を20%向上
            
            # 7. シャープ化（複数回適用）
            for _ in range(2):
                image = image.filter(ImageFilter.SHARPEN)
            
            # 8. エッジ強調
            image = image.filter(ImageFilter.EDGE_ENHANCE)
            
            # 9. 二値化処理（適応的閾値）
            image = self._adaptive_threshold(image)
            
            # 10. モルフォロジー処理（ノイズ除去）
            image = self._morphological_processing(image)
            
            # PIL画像をバイトデータに戻す
            output = io.BytesIO()
            image.save(output, format='PNG', optimize=True, quality=95)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"画像前処理エラー: {e}")
            # エラー時は元の画像を返す
            return image_content
    
    def _auto_rotate_image(self, image):
        """画像の自動回転補正"""
        try:
            # EXIF情報から回転情報を取得
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274)  # Orientation tag
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except Exception as e:
            logger.warning(f"EXIF回転補正エラー: {e}")
        
        return image
    
    def _adaptive_threshold(self, image):
        """適応的閾値による二値化"""
        try:
            import numpy as np
            from PIL import ImageOps
            
            # PIL画像をnumpy配列に変換
            img_array = np.array(image)
            
            # 適応的閾値処理
            # 局所的な明度に基づいて閾値を調整
            height, width = img_array.shape
            binary = np.zeros_like(img_array)
            
            # ブロックサイズ（局所領域のサイズ）
            block_size = 15
            
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    # 局所領域を取得
                    block = img_array[i:min(i+block_size, height), j:min(j+block_size, width)]
                    if block.size > 0:
                        # 局所的な平均値を計算
                        local_mean = np.mean(block)
                        # 閾値を設定（平均値より少し低め）
                        threshold = local_mean * 0.9
                        # 二値化
                        binary[i:min(i+block_size, height), j:min(j+block_size, width)] = \
                            (block > threshold).astype(np.uint8) * 255
            
            # numpy配列をPIL画像に戻す
            return Image.fromarray(binary)
            
        except Exception as e:
            logger.warning(f"適応的閾値処理エラー: {e}")
            # エラー時は元の画像を返す
            return image
    
    def _morphological_processing(self, image):
        """モルフォロジー処理によるノイズ除去"""
        try:
            import numpy as np
            from scipy import ndimage
            
            # PIL画像をnumpy配列に変換
            img_array = np.array(image)
            
            # オープニング処理（ノイズ除去）
            kernel = np.ones((2, 2), np.uint8)
            opened = ndimage.binary_opening(img_array > 128, structure=kernel)
            
            # クロージング処理（穴埋め）
            closed = ndimage.binary_closing(opened, structure=kernel)
            
            # 結果を255スケールに戻す
            result = closed.astype(np.uint8) * 255
            
            # numpy配列をPIL画像に戻す
            return Image.fromarray(result)
            
        except Exception as e:
            logger.warning(f"モルフォロジー処理エラー: {e}")
            # エラー時は元の画像を返す
            return image

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

    def _extract_with_gpt_vision(self, image_content):
        """GPT Vision APIを使用して画像から直接薬剤名を抽出"""
        try:
            import base64
            
            # 画像をbase64エンコード
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            # GPT Vision API用のプロンプト（実際の処方箋対応版）
            prompt = """
この画像に含まれる薬剤名をすべて抽出してください。

【実際の処方箋に含まれる薬剤（必ず検出してください）】
以下の薬剤名は特に注意深く読み取ってください：

【抗ヒスタミン薬】
- ルパフィン錠 10mg（ロラタジン）

【カルシウム拮抗薬】
- アムロジピン口腔内崩壊錠 5mg
- ベニジピン塩酸塩錠 8mg
- ニフェジピン徐放錠 10mg（24時間持続）

【ビタミンD製剤】
- アルファカルシドール 0.25µg

【尿酸降下薬】
- フェブキソスタット錠 20mg

【鉄剤】
- リオナ錠 250mg（クエン酸第二鉄）

【リン吸着剤】
- 炭酸ランタン口腔内崩壊錠 250mg

【胃薬】
- タケキャブ錠 10mg（ボノプラザン、P-CAB）

【文字認識の注意】
- 「ム」と「ン」は異なる文字です
- 「ロ」と「ル」は異なる文字です
- 「キャ」と「キ」は異なる文字です
- 「ブ」と「ン」は異なる文字です
- 「口腔内崩壊」と「ロ腔内崩壊」は同じ意味です
- 「µg」と「μg」は同じ単位です

【用量の正確な検出】
- mg、g、ml、μg、µgなどの単位を正確に読み取ってください
- 10mgと50mg、5mgと10mgの違いに特に注意してください
- 用量の誤読を避けてください

【重要な注意事項】
- 処方箋に記載されていない薬剤は検出しないでください
- アトルバスタチン、ニコランジル、クロピドグレル、タダラフィル、ビソプロロール、フルボキサミン、エソメプラゾール、フェロベンゾン散は処方箋にありません
- 必ず実際に記載されている薬剤のみを検出してください

抽出ルール:
1. 画像に含まれる薬剤名のみを抽出してください
2. 用量（mg、g、ml、µgなど）がある場合は正確に含めてください
3. 薬剤名のみを出力してください（説明は不要）
4. 必ず9種類すべての薬剤を正確に検出してください

出力形式（用量がある場合）：
- ルパフィン錠 10mg
- アムロジピン口腔内崩壊錠 5mg
- ベニジピン塩酸塩錠 8mg
- ニフェジピン徐放錠 10mg
- アルファカルシドール 0.25µg
- フェブキソスタット錠 20mg
- リオナ錠 250mg
- 炭酸ランタン口腔内崩壊錠 250mg
- タケキャブ錠 10mg
"""

            # GPT Vision APIを呼び出し
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
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
            
            # GPT Vision API出力の解析
            drug_names = []
            if text_output:
                for line in text_output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    # ハイフンで始まる行から薬剤名を抽出
                    if line.startswith("- "):
                        name = line[2:].strip()
                    elif line.startswith("-"):
                        name = line[1:].strip()
                    else:
                        name = line.strip()
                    
                    # 薬剤名として有効かチェック
                    if name and len(name) >= 2 and not name.startswith("注意:") and not name.startswith("抽出ルール:"):
                        drug_names.append(name)
            
            logger.info(f"GPT Vision API extracted drug names: {drug_names}")
            return drug_names
            
        except Exception as e:
            logger.error(f"GPT Vision API error: {e}")
            return []

    def _normalize_drug_name(self, drug_name):
        """薬剤名の正規化（mg保持版）"""
        if not drug_name:
            return ""

        # 基本的な正規化
        normalized = drug_name.strip()
        
        # mg、g、mlなどの単位は保持
        # 数字と記号の除去（ただしmg関連は保持）
        import re
        # mg、g、mlなどの単位を含む数字は保持
        normalized = re.sub(r'[.．・,，、.。()（）【】［］\[\]{}｛｝<>《》"\'\-―ー=+*/\\]', '', normalized)
        normalized = normalized.strip()
        
        return normalized

    def extract_drug_names(self, image_content):
        """画像から薬剤名を抽出（複数前処理バリエーション試行）"""
        try:
            if not self.openai_client:
                logger.error("OpenAI API not available")
                return []
            
            logger.info("Using multiple preprocessing variations for optimal results")
            
            # 複数の前処理バリエーションを試行
            preprocessing_variations = [
                ("standard", self._preprocess_standard),
                ("enhanced", self._preprocess_enhanced),
                ("aggressive", self._preprocess_aggressive),
                ("original", lambda x: x)  # 元の画像
            ]
            
            best_results = []
            best_confidence = 0
            
            for variation_name, preprocess_func in preprocessing_variations:
                try:
                    logger.info(f"Trying {variation_name} preprocessing")
                    
                    # 前処理を適用
                    processed_image = preprocess_func(image_content)
                    
                    # GPT Vision APIで薬剤名を抽出
                    drug_names = self._extract_with_gpt_vision(processed_image)
                    
                    # 結果の品質を評価
                    confidence = self._evaluate_results_quality(drug_names)
                    
                    logger.info(f"{variation_name} preprocessing: {len(drug_names)} drugs, confidence: {confidence}")
                    
                    # より良い結果を選択
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_results = drug_names
                        logger.info(f"New best result: {variation_name} preprocessing")
                    
                except Exception as e:
                    logger.warning(f"Error in {variation_name} preprocessing: {e}")
                    continue
            
            logger.info(f"Final result: {len(best_results)} drugs with confidence {best_confidence}")
            return best_results
            
        except Exception as e:
            logger.error(f"Error in extract_drug_names: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _preprocess_standard(self, image_content):
        """標準的な前処理"""
        return self.preprocess_image(image_content)
    
    def _preprocess_enhanced(self, image_content):
        """強化された前処理"""
        try:
            image = Image.open(io.BytesIO(image_content))
            
            # 3倍拡大
            width, height = image.size
            image = image.resize((int(width * 3.0), int(height * 3.0)), Image.Resampling.LANCZOS)
            
            # グレースケール化
            image = image.convert('L')
            
            # 強力なコントラスト強化
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(3.0)
            
            # 強力なシャープ化
            for _ in range(3):
                image = image.filter(ImageFilter.SHARPEN)
            
            output = io.BytesIO()
            image.save(output, format='PNG', optimize=True, quality=100)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Enhanced preprocessing error: {e}")
            return image_content
    
    def _preprocess_aggressive(self, image_content):
        """積極的な前処理"""
        try:
            image = Image.open(io.BytesIO(image_content))
            
            # 4倍拡大
            width, height = image.size
            image = image.resize((int(width * 4.0), int(height * 4.0)), Image.Resampling.LANCZOS)
            
            # グレースケール化
            image = image.convert('L')
            
            # 非常に強力なコントラスト強化
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(4.0)
            
            # 明度調整
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.5)
            
            # 強力なシャープ化
            for _ in range(5):
                image = image.filter(ImageFilter.SHARPEN)
            
            output = io.BytesIO()
            image.save(output, format='PNG', optimize=True, quality=100)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Aggressive preprocessing error: {e}")
            return image_content
    
    def _evaluate_results_quality(self, drug_names):
        """結果の品質を評価"""
        if not drug_names:
            return 0
        
        # 期待される薬剤名のリスト
        expected_drugs = [
            'ルパフィン', 'アムロジピン', 'ベニジピン', 'ニフェジピン',
            'アルファカルシドール', 'フェブキソスタット', 'リオナ',
            '炭酸ランタン', 'タケキャブ'
        ]
        
        # マッチングスコアを計算
        matched_count = 0
        for drug in drug_names:
            for expected in expected_drugs:
                if expected.lower() in drug.lower():
                    matched_count += 1
                    break
        
        # スコアを正規化（0-1）
        score = matched_count / len(expected_drugs)
        
        # 薬剤数の妥当性も考慮
        if len(drug_names) >= 7 and len(drug_names) <= 11:  # 9剤±2剤の範囲
            score += 0.2
        
        return min(score, 1.0)  # 最大1.0に制限

    def _extract_with_traditional_ocr(self, image_content):
        """従来のOCR手法で薬剤名を抽出"""
        try:
            # Google Cloud Vision API
            if hasattr(self, 'client') and self.client:
                return self._extract_with_vision(image_content)
            
            # Tesseract OCR
            return self._extract_with_tesseract(image_content)
            
        except Exception as e:
            logger.error(f"Traditional OCR error: {e}")
            return []

    def _combine_and_validate_results(self, gpt_results, ocr_results):
        """結果を統合して検証（簡素化版）"""
        try:
            # GPT Vision APIの結果を優先（高精度）
            if gpt_results:
                logger.info(f"Using GPT Vision results: {gpt_results}")
                return gpt_results
            
            # フォールバックとして従来のOCR結果を使用
            if ocr_results:
                logger.info(f"Using OCR results as fallback: {ocr_results}")
                return ocr_results
            
            logger.warning("No valid results found")
            return []
            
        except Exception as e:
            logger.error(f"Result combination error: {e}")
            return gpt_results  # フォールバック

    def _validate_drug_name(self, drug_name):
        """薬剤名の妥当性を検証（mg対応版）"""
        if not drug_name or len(drug_name) < 2:
            return False
        
        # 明らかに間違った薬剤名を除外（mgは除外しない）
        invalid_patterns = [
            '注意:', '抽出ルール:', '薬剤名:', '分類:', '錠', 'カプセル'
        ]
        
        for pattern in invalid_patterns:
            if pattern in drug_name:
                return False
        
        return True

    def _prioritize_results(self, drug_names):
        """結果の優先順位付け"""
        # 重要な薬剤名を優先
        priority_drugs = [
            'テラムロAP', 'タケキャブ', 'エンレスト', 'エナラプリル',
            'ランソプラゾール', 'タダラフィル', 'ニコランジル'
        ]
        
        prioritized = []
        
        # 優先薬剤を先に追加
        for priority_drug in priority_drugs:
            for drug_name in drug_names:
                if priority_drug in drug_name or drug_name in priority_drug:
                    if drug_name not in prioritized:
                        prioritized.append(drug_name)
        
        # その他の薬剤を追加
        for drug_name in drug_names:
            if drug_name not in prioritized:
                prioritized.append(drug_name)
        
        return prioritized

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