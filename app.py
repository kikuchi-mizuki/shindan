import os
import logging
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, MessagingApiBlob, TextMessage, ReplyMessageRequest, PushMessageRequest

from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# LINE Bot設定
configuration = Configuration(access_token=os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
messaging_blob_api = MessagingApiBlob(api_client)
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# ユーザーごとの薬剤名バッファ
user_drug_buffer = {}

# サービスの遅延初期化
_services_initialized = False
ocr_service = None
drug_service = None
response_service = None
ai_extractor = None
classifier = None

def initialize_services():
    """サービスの初期化（遅延実行）"""
    global ocr_service, drug_service, response_service, ai_extractor, classifier, _services_initialized
    
    if _services_initialized:
        return True
    
    try:
        logger.info("Initializing services...")
        
        try:
            from services.ocr_service import OCRService
            ocr_service = OCRService()
            logger.info("OCRService initialized")
        except Exception as e:
            logger.error(f"OCRService initialization failed: {e}")
            return False
        
        try:
            from services.drug_service import DrugService
            drug_service = DrugService()
            logger.info("DrugService initialized")
        except Exception as e:
            logger.error(f"DrugService initialization failed: {e}")
            return False
        
        try:
            from services.response_service import ResponseService
            response_service = ResponseService()
            logger.info("ResponseService initialized")
        except Exception as e:
            logger.error(f"ResponseService initialization failed: {e}")
            return False
        
        try:
            from services.ai_extractor import AIExtractorService
            ai_extractor = AIExtractorService()
            logger.info("AIExtractorService initialized")
        except Exception as e:
            logger.error(f"AIExtractorService initialization failed: {e}")
            return False
        
        try:
            from services.classifier import Classifier
            classifier = Classifier(kegg_service=drug_service)
            logger.info("Classifier initialized")
        except Exception as e:
            logger.error(f"Classifier initialization failed: {e}")
            return False
        
        _services_initialized = True
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False

@app.route("/", methods=['GET'])
def root():
    """ルートエンドポイント"""
    try:
        return {"status": "ok", "message": "薬局サポートBot is running"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route("/health", methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        return {"status": "healthy", "message": "ok"}, 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}, 500

@app.route("/test", methods=['GET'])
def test():
    """テストエンドポイント"""
    try:
        # 基本的な環境変数チェック
        access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
        channel_secret = os.getenv('LINE_CHANNEL_SECRET')
        
        return {
            "status": "ok",
            "access_token_exists": bool(access_token),
            "channel_secret_exists": bool(channel_secret),
            "message": "LINE Bot configuration test passed"
        }, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route("/callback", methods=['POST'])
def callback():
    """LINE Webhook コールバック"""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    """テキストメッセージの処理"""
    try:
        user_id = event.source.user_id
        user_message = event.message.text
        
        # サービス初期化チェック
        if not _services_initialized:
            if not initialize_services():
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="サービス初期化エラーが発生しました。しばらく時間をおいて再度お試しください。")]
                    )
                )
                return
        
        # 基本的なテキストメッセージ処理
        if user_message.lower() in ['診断', 'しんだん', 'diagnosis']:
            if user_id in user_drug_buffer and user_drug_buffer[user_id]:
                # 診断中のメッセージを送信
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="🔍 診断中です…")]
                    )
                )
                
                # 診断処理
                # 薬剤データの形式を判定して薬剤名を抽出
                drug_list = user_drug_buffer[user_id]
                if drug_list and isinstance(drug_list[0], dict):
                    # AI抽出結果の詳細情報から薬剤名を抽出
                    drug_names = [drug.get('name', '') for drug in drug_list if drug.get('name')]
                else:
                    # 従来の薬剤名リスト
                    drug_names = drug_list
                
                if drug_names:
                    drug_info = drug_service.get_drug_interactions(drug_names)
                    response_text = response_service.generate_response(drug_info)
                else:
                    response_text = "薬剤情報が正しく取得できませんでした。"
                
                # 診断結果を送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
            else:
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="薬剤が登録されていません。\n\n📝 **薬剤名を直接入力して診断する方法：**\n\n薬剤名をスペース区切りで入力してください。\n例：クラリスロマイシン ベルソムラ デビゴ\n\nまたは、画像を送信して薬剤を登録してください。")]
                    )
                )
        
        elif user_message.lower() in ['リスト確認', 'りすとかくにん', 'list']:
            if user_id in user_drug_buffer and user_drug_buffer[user_id]:
                # 薬剤データの形式を判定して表示
                drug_list = user_drug_buffer[user_id]
                if drug_list and isinstance(drug_list[0], dict):
                    # AI抽出結果の詳細情報を表示
                    drug_display = []
                    for drug in drug_list:
                        name = drug.get('name', '')
                        strength = drug.get('strength', '')
                        final_classification = drug.get('final_classification', '')
                        class_hint = drug.get('class_hint', '')
                        kegg_category = drug.get('kegg_category', '')
                        
                        # カテゴリの優先順位
                        if final_classification and final_classification != '分類未設定':
                            category = final_classification
                        elif kegg_category:
                            category = kegg_category
                        elif class_hint:
                            category = f"{class_hint}（AI推定）"
                        else:
                            category = '不明'
                        
                        display_text = f"• {name}"
                        if strength:
                            display_text += f" {strength}"
                        display_text += f" (分類: {category})"
                        drug_display.append(display_text)
                    drug_list_text = "\n".join(drug_display)
                else:
                    # 従来の薬剤名リスト
                    drug_list_text = "\n".join([f"• {drug}" for drug in drug_list])
                
                response_text = f"📋 **現在の薬剤リスト**\n\n{drug_list_text}\n\n💡 「診断」で飲み合わせチェックを実行できます"
            else:
                response_text = "薬剤が登録されていません。画像を送信して薬剤を登録してください。"
            
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=response_text)]
                )
            )
        
        elif user_message.lower() in ['ヘルプ', 'へるぷ', 'help', '使い方', 'つかいかた']:
            help_text = """🏥 **薬局サポートBot 使い方**

**📸 薬剤の登録方法（画像）：**
1. 処方箋の画像を送信
2. 検出された薬剤を確認
3. 「診断」で飲み合わせチェック

**📝 薬剤の登録方法（テキスト）：**
薬剤名をスペース区切りで直接入力
例：クラリスロマイシン ベルソムラ デビゴ

**📋 撮影のコツ：**
• 真上から撮影（左右2ページは分割）
• 影・反射を避ける
• 処方名の行がはっきり写るように
• **iOSの「書類をスキャン」推奨**
• 文字の向きを正しく（横向きはNG）

**🔧 改善方法：**
• 明るい場所で撮影
• カメラを安定させる
• 処方箋全体が画面に入るように
• ピントを合わせてから撮影

**💊 コマンド一覧：**
• 診断 - 飲み合わせチェック
• リスト確認 - 現在の薬剤リスト
• 薬剤追加：〇〇 - 薬剤を手動追加
• ヘルプ - この使い方表示

**💡 便利な使い方：**
• 画像が読み取れない場合は薬剤名を直接入力
• 複数の薬剤を一度に入力可能
• 入力後すぐに診断実行

**⚠️ 注意事項：**
• 最終判断は医師・薬剤師にご相談ください
• 画質が悪い場合は再撮影をお願いします
• 手書きの処方箋は読み取り精度が低下する場合があります"""
            
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=help_text)]
                )
            )
        
        elif user_message.startswith('薬剤追加：'):
            drug_name = user_message.replace('薬剤追加：', '').strip()
            if drug_name:
                if user_id not in user_drug_buffer:
                    user_drug_buffer[user_id] = []
                
                # 薬剤データの形式を判定
                if user_drug_buffer[user_id] and isinstance(user_drug_buffer[user_id][0], dict):
                    # 辞書形式の場合、薬剤名の重複チェック
                    existing_names = [drug.get('name', '') for drug in user_drug_buffer[user_id]]
                    if drug_name not in existing_names:
                        # 分類サービスで分類を取得
                        temp_drug = {'generic': drug_name, 'raw': drug_name}
                        classification = classifier.classify_one(temp_drug)
                        
                        # 新しい薬剤を辞書形式で追加
                        new_drug = {
                            'name': drug_name,
                            'raw': drug_name,
                            'strength': '',
                            'dose': '',
                            'freq': '',
                            'days': None,
                            'class_hint': '',
                            'final_classification': classification or '分類未設定',
                            'kegg_category': '',
                            'kegg_id': ''
                        }
                        user_drug_buffer[user_id].append(new_drug)
                        response_text = f"✅ 薬剤「{drug_name}」を追加しました。"
                    else:
                        response_text = f"薬剤「{drug_name}」は既に登録されています。"
                else:
                    # 従来の文字列リストの場合
                    if drug_name not in user_drug_buffer[user_id]:
                        user_drug_buffer[user_id].append(drug_name)
                        response_text = f"✅ 薬剤「{drug_name}」を追加しました。"
                    else:
                        response_text = f"薬剤「{drug_name}」は既に登録されています。"
            else:
                response_text = "薬剤名を入力してください。例：薬剤追加：アムロジピン"
            
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=response_text)]
                )
            )
        
        # 薬剤名を直接入力して診断する機能
        elif (' ' in user_message or '\n' in user_message) and len(user_message.replace('\n', ' ').split()) >= 2:
            # スペース区切りまたは改行区切りの薬剤名を解析
            drug_names = user_message.replace('\n', ' ').split()
            logger.info(f"Direct drug input detected: {drug_names}")
            
            # 診断中のメッセージを送信
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text="🔍 診断中です…")]
                )
            )
            
            # 薬剤名を正規化してマッチング
            matched_drugs = drug_service.match_to_database(drug_names)
            logger.info(f"Matched drugs: {matched_drugs}")
            
            if matched_drugs:
                # ユーザーバッファに追加（文字列形式で保存）
                if user_id not in user_drug_buffer:
                    user_drug_buffer[user_id] = []
                
                for matched_drug in matched_drugs:
                    if matched_drug not in user_drug_buffer[user_id]:
                        user_drug_buffer[user_id].append(matched_drug)
                
                # 診断処理
                drug_info = drug_service.get_drug_interactions(matched_drugs)
                response_text = response_service.generate_response(drug_info)
                
                # 診断結果を送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
            else:
                # 薬剤が見つからない場合
                response_text = f"❌ 入力された薬剤名を認識できませんでした。\n\n入力: {' '.join(drug_names)}\n\n💡 **正しい薬剤名の例：**\n• クラリスロマイシン\n• ベルソムラ\n• デビゴ\n• ロゼレム\n• フルボキサミン\n• アムロジピン\n• エソメプラゾール\n\n薬剤名を確認して再度入力してください。"
                
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
        
        else:
            response_text = "薬局サポートBotへようこそ！\n\n画像を送信して薬剤を登録するか、以下のコマンドを使用してください：\n• 診断 - 飲み合わせチェック\n• 薬剤追加：〇〇 - 薬剤を手動追加\n• リスト確認 - 現在の薬剤リスト\n• ヘルプ - 使い方表示"
            
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=response_text)]
                )
            )
        
    except Exception as e:
        logger.error(f"Text message handling error: {e}")
        try:
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text="エラーが発生しました。しばらく時間をおいて再度お試しください。")]
                )
            )
        except:
            pass

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """画像メッセージの処理"""
    try:
        user_id = event.source.user_id
        
        # サービス初期化チェック
        if not _services_initialized:
            if not initialize_services():
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="サービス初期化エラーが発生しました。しばらく時間をおいて再度お試しください。")]
                    )
                )
                return
        
        # キャッシュクリア（前回の検出結果をリセット）
        if user_id in user_drug_buffer:
            user_drug_buffer[user_id] = []
            logger.info(f"Cleared drug buffer for user {user_id}")
        
        # 診断中のメッセージを送信
        messaging_api.reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text="🔍 診断中です…")]
            )
        )
        
        # 画像処理
        message_content = messaging_blob_api.get_message_content(event.message.id)
        
        # 画像品質評価付きOCR処理
        ocr_result = ocr_service.extract_drug_names(message_content)
        
        # 品質に応じた処理分岐
        if not ocr_result['should_process']:
            # 低品質画像の場合、ガイドを表示
            messaging_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=ocr_result['guide'])]
                )
            )
            return
        
        # OCRテキストを取得
        ocr_text = ocr_result.get('raw_text', '')
        if not ocr_text:
            # フォールバック: 従来の方法で薬剤名抽出
            drug_names = ocr_result['drug_names']
            if drug_names:
                matched_drugs = drug_service.match_to_database(drug_names)
            else:
                matched_drugs = []
        else:
            # AI抽出サービスを使用
            ai_result = ai_extractor.extract_drugs(ocr_text)
            logger.info(f"AI extraction result: {ai_result}")
            
            # 信頼度チェック
            confidence = ai_result.get('confidence', 'low')
            if confidence == 'low':
                # 信頼度が低い場合は確認メッセージを送信
                confirmation_message = ai_extractor.generate_confirmation_message(ai_result)
                if confirmation_message:
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=confirmation_message)]
                        )
                    )
                    return
            
            # AI抽出結果から薬剤名を取得
            ai_drugs = ai_result.get('drugs', [])
            matched_drugs = []
            
            # 分類サービスで薬剤分類を付与
            classified_drugs = classifier.classify_many(ai_drugs)
            
            # 薬剤名リストを作成
            drug_names = []
            for drug in classified_drugs:
                generic_name = drug.get('generic', '')
                if generic_name:
                    drug_names.append(generic_name)
            
            # post_filterでジクロフェナクの誤混入を防止
            # raw_textはOCR結果から取得
            raw_text = ocr_result.get('raw_text', '') if isinstance(ocr_result, dict) else ''
            filtered_drug_names = drug_service.post_filter(drug_names, raw_text)
            
            # フィルタリング後の薬剤のみを処理
            for drug in classified_drugs:
                generic_name = drug.get('generic', '')
                if generic_name in filtered_drug_names:
                    # KEGG照合を実行
                    kegg_info = drug_service.safe_find_kegg_info(generic_name)
                    if kegg_info:
                        # KEGG IDを付与
                        drug['kegg_id'] = kegg_info.get('kegg_id')
                        drug['kegg_category'] = kegg_info.get('category')
                    
                    # 分類情報を保持
                    drug['final_classification'] = drug.get('class_jp', '分類未設定')
                    
                    matched_drugs.append(generic_name)
        
        if matched_drugs:
            # ユーザーバッファに薬剤名を追加
            if user_id not in user_drug_buffer:
                user_drug_buffer[user_id] = []
            
            unique_input_count = len(set(matched_drugs))
            
            if matched_drugs:
                # AI抽出結果の詳細情報を保存
                if 'classified_drugs' in locals():
                    for drug in classified_drugs:
                        generic_name = drug.get('generic', '')
                        if generic_name and generic_name in matched_drugs:
                            # 薬剤の詳細情報を辞書として保存
                            drug_info = {
                                'name': generic_name,
                                'raw': drug.get('raw', ''),
                                'strength': drug.get('strength', ''),
                                'dose': drug.get('dose', ''),
                                'freq': drug.get('freq', ''),
                                'days': drug.get('days'),
                                'class_hint': drug.get('class_hint', ''),
                                'final_classification': drug.get('final_classification', ''),
                                'kegg_category': drug.get('kegg_category', ''),
                                'kegg_id': drug.get('kegg_id', '')
                            }
                            user_drug_buffer[user_id].append(drug_info)
                else:
                    # フォールバック: 従来の方法
                    for matched_drug_name in matched_drugs:
                        user_drug_buffer[user_id].append(matched_drug_name)
                
                # 検出結果の確認メッセージを表示
                response_text = response_service.generate_simple_response(matched_drugs)
                
                # 検出結果を送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
                
                # 診断ボタン付きFlex Messageを送信（薬剤が検出された場合のみ）
                try:
                    from linebot.v3.messaging import FlexMessage, FlexContainer
                    
                    flex_message = {
                        "type": "bubble",
                        "body": {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "お薬飲み合わせ・同効薬をチェック！",
                                    "weight": "bold",
                                    "size": "md",
                                    "align": "center",
                                    "margin": "md",
                                    "color": "#222222"
                                }
                            ]
                        },
                        "footer": {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {
                                    "type": "button",
                                    "action": {
                                        "type": "message",
                                        "label": "🔍 診断実行",
                                        "text": "診断"
                                    },
                                    "style": "primary",
                                    "color": "#1DB446"
                                }
                            ]
                        }
                    }
                    
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[FlexMessage(altText="診断ボタン", contents=FlexContainer.from_dict(flex_message))]
                        )
                    )
                except Exception as flex_error:
                    logger.error(f"Flex Message error: {flex_error}")
                    # Flex Messageが失敗した場合はテキストメッセージで代替
                    fallback_text = "💡 以下のコマンドを入力してください：\n• 診断 - 飲み合わせチェック\n• リスト確認 - 現在の薬剤リスト\n• ヘルプ - 使い方表示"
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=fallback_text)]
                        )
                    )
                
                # 部分的にしか読めていない場合はガイドも併送
                if len(matched_drugs) < unique_input_count:
                    partial_guide = """⚠️ 一部の薬剤名を正確に読み取れませんでした

検出結果を確認し、不足があれば手動で追加してください。

📋 撮影のコツ
• 真上から1ページずつ撮影（左右2ページは分割）
• 明るい場所で、影や反射を避ける
• 文字がはっきり写る距離でピント

💊 手動追加（例）
• 薬剤追加：アムロジピン
• 薬剤追加：エソメプラゾール
"""
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=partial_guide)]
                        )
                    )
            else:
                # 薬剤が検出されなかった場合
                response_text = "画像から薬剤名を検出できませんでした。より鮮明な画像で再度お試しください。"
                
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
        else:
            response_text = "画像から薬剤名を検出できませんでした。より鮮明な画像で再度お試しください。"
            
            messaging_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=response_text)]
                )
            )
        
    except Exception as e:
        logger.error(f"Image message handling error: {e}")
        try:
            messaging_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text="画像処理中にエラーが発生しました。しばらく時間をおいて再度お試しください。")]
                )
            )
        except:
            pass

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting complete drug interaction diagnosis system on port {port}, debug mode: {debug_mode}")
        logger.info("Complete drug interaction diagnosis system startup completed successfully")
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 