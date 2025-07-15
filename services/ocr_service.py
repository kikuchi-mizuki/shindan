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
        """OCR前処理: グレースケール化・二値化・シャープ化・ノイズ除去・コントラスト調整・解像度向上"""
        try:
            # バイトデータから画像を開く
            image = Image.open(io.BytesIO(image_content))
            
            # 解像度向上（2倍に拡大）
            width, height = image.size
            image = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
            
            # グレースケール化
            image = image.convert('L')
            
            # コントラスト調整（PILのEnhance機能を使用）
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # コントラストを2倍に
            
            # 明度調整
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)  # 明度を1.2倍に
            
            # ノイズ除去（ガウシアンフィルタ）
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
            
            # シャープ化（複数回適用）
            image = image.filter(ImageFilter.SHARPEN)
            image = image.filter(ImageFilter.SHARPEN)
            image = image.filter(ImageFilter.SHARPEN)
            
            # 二値化（適応的閾値処理）
            img_array = np.array(image)
            # 局所的な閾値を計算
            threshold = np.mean(img_array) + np.std(img_array) * 0.5
            binary = img_array > threshold
            image = Image.fromarray(binary.astype(np.uint8) * 255)
            
            # モルフォロジー処理でノイズ除去（PILベース）
            # 小さなノイズを除去するため、エッジ検出と組み合わせ
            image = image.filter(ImageFilter.FIND_EDGES)
            image = image.filter(ImageFilter.SMOOTH)
            
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

出力形式:
薬剤名: アルプラゾラム
薬剤名: アモバルビタール
薬剤名: ブロマゼパム

抽出ルール:
1. 薬剤名のみを抽出（数字、単位、説明文は除外）
2. 必ず「薬剤名: 正規化後の名前」の形式で出力
3. 販売中止の薬剤も含める
4. 漢方薬や生薬も含める
5. 英語名や略称は日本語名に統一
6. ひらがな表記の薬剤名は必ずカタカナに変換してください
7. 薬剤名は正式名称（カタカナ）で出力してください

注意: 
- ハイフン（-）を使った形式は使用しないでください
- ひらがなの薬剤名は必ずカタカナに変換してください
- 抽出された薬剤名のみを出力してください。説明やコメントは不要です
"""

            logger.info(f"Sending OCR text to ChatGPT for drug name extraction")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # コスト効率の良いモデルを使用
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
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
            
            logger.info(f"ChatGPT extracted drug names: {drug_names}")
            return drug_names
            
        except Exception as e:
            logger.error(f"ChatGPT API error: {e}")
            logger.info("Falling back to traditional extraction method")
            return self._extract_drug_names_from_text(ocr_text)

    def extract_drug_names(self, image_content):
        """画像から薬剤名を抽出する（ChatGPT統合版）"""
        try:
            # 2. OCRテキストを取得（まず元の画像で試行）
            logger.info(f"Vision available: {self.vision_available}")
            if self.vision_available:
                logger.info("Using Vision API for OCR")
                ocr_text = self._extract_with_vision(image_content)
            else:
                logger.info("Using local OCR processing")
                ocr_text = self._extract_local_ocr(image_content)
            
            # OCRテキストが空の場合は前処理した画像で再試行
            if not ocr_text or len(ocr_text.strip()) < 10:
                logger.info("OCR text too short, trying with preprocessed image")
                processed_image = self.preprocess_image(image_content)
                if self.vision_available:
                    ocr_text = self._extract_with_vision(processed_image)
                else:
                    ocr_text = self._extract_local_ocr(processed_image)
            
            # OCR結果を必ずログ出力
            logger.info(f"[OCR raw text]:\n{ocr_text}")
            logger.info(f"OCR text length: {len(ocr_text) if ocr_text else 0}")
            
            if not ocr_text or len(ocr_text.strip()) < 5:
                logger.warning("OCR text too short or empty")
                return []
            
            # ChatGPTを使用して薬剤名を抽出・正規化
            if self.openai_available:
                logger.info("Using ChatGPT for drug name extraction and normalization")
                drug_names = self.extract_drug_names_with_chatgpt(ocr_text)
            else:
                logger.info("Using traditional drug name extraction")
                drug_names = self._extract_drug_names_from_text(ocr_text)
            
            logger.info(f"Final extracted drug names: {drug_names}")
            return drug_names
            
        except Exception as e:
            logger.error(f"Error in extract_drug_names: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
            
            # TesseractでOCRを実行
            # 日本語モデルを使用
            ocr_result = pytesseract.image_to_string(
                image_byte_array,
                lang='jpn',
                config='--psm 11 --oem 3' # テキストブロックを1つのテキスト行として認識
            )
            
            logger.info(f"Tesseract OCR result: {ocr_result}")
            return ocr_result
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
        # カッコ内・販売中止・全角→半角・空白除去
        name = re.sub(r'（.*?）', '', drug_name)
        name = re.sub(r'\(.*?\)', '', name)
        name = name.replace('販売中止', '')
        name = name.replace(' ', '').replace('　', '')
        # 数字・記号除去
        name = re.sub(r'[0-9０-９.．・,，、.。()（）【】［］\[\]{}｛｝<>《》"\'\-―ー=+*/\\]', '', name)
        name = name.strip()
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
            'えすぞぴくろん': 'エソピクロン',
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
            'パントバレピター': 'パントバレピター',
            'フェノバレピクル': 'フェノバレピクル',
            'リアンフム': 'リアンフム',
            'トンソバム': 'トンソバム',
            'フバルーン': 'フバルーン',
            'タバスピン': 'タバスピン',
            'プラザパ': 'プラザパ',
            'リバビジョン': 'リバビジョン',
            'コンラゼペー': 'コンラゼペー',
            'ニトラセバム': 'ニトラセバム',
            'タゼパ': 'タゼパ',
            'コアアゼバム': 'コアアゼバム',
            'エスタソフム': 'エスタソフム',
            'スリーピン': 'スリーピン',
            'ハロゲンフム': 'ハロゲンフム',
            'フルクソフム': 'フルクソフム',
            'ブラシパハ': 'ブラシパハ',
            'ブトクトアニド': 'ブトクトアニド',
            'プロモバリン': 'プロモバリン',
            'ポエシド': 'ポエシド',
            'コサール': 'コサール',
            'ビル': 'ビル',
            'バター': 'バター',
            'プログラム': 'プログラム',
            'ペー': 'ペー',
            'フル': 'フル'
        }
        
        # 完全一致の修正
        if name in corrections:
            return corrections[name]
        
        # 部分一致の修正
        for wrong, correct in corrections.items():
            if wrong in name or name in wrong:
                return correct
        
        # 語尾パターンによる推測
        if name.endswith('ゼパム'):
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
            'keggid', 'kegg_id', 'kegg id', 'keggid', 'kegg_id', 'kegg id'
        }
        
        # 行ごとに分割
        lines = text.split('\n')
        
        for line in lines:
            # 行全体から薬剤名を抽出（1列目だけでなく）
            line_drug_names = []
            
            # ひらがな薬剤名パターン（カタカナに変換）
            hiragana_pattern = r'([あ-んー]{2,}(ぱむ|らむ|ろん|じあぜぱむ|ぴおん|ぺーと|じん|ぞらむ|ばみる|ばみど|ばみん|ーる|とーる|ちじん)?)'
            for match in re.finditer(hiragana_pattern, line):
                name = match.group(0)
                if name and name not in exclude_words:
                    # ひらがなをカタカナに変換してから正規化
                    katakana_name = self._hiragana_to_katakana(name)
                    normalized_name = self._normalize_drug_name(katakana_name)
                    line_drug_names.append(normalized_name)
            
            # カタカナ薬剤名パターン
            katakana_pattern = r'([ァ-ヶー]{2,}(パム|ラム|ロン|ジアゼパム|ピオン|ペート|ジン|ゾラム|バミル|バミド|バミン|ール|トール|チジン)?)'
            for match in re.finditer(katakana_pattern, line):
                name = match.group(0)
                if name and name not in exclude_words:
                    # 薬剤名の正規化・補完
                    normalized_name = self._normalize_drug_name(name)
                    line_drug_names.append(normalized_name)
            
            # 既存の他パターンも適用
            # ひらがな2文字以上の単語
            for n in re.findall(r'[あ-んー]{2,}', line):
                if n not in exclude_words and n not in line_drug_names:
                    # ひらがなをカタカナに変換してから正規化
                    katakana_name = self._hiragana_to_katakana(n)
                    normalized_name = self._normalize_drug_name(katakana_name)
                    line_drug_names.append(normalized_name)
            
            # カタカナ2文字以上の単語
            for n in re.findall(r'[ァ-ヶー]{2,}', line):
                if n not in exclude_words and n not in line_drug_names:
                    # 薬剤名の正規化・補完
                    normalized_name = self._normalize_drug_name(n)
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