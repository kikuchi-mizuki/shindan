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
redis_service = None

def initialize_services():
    """サービスの初期化（遅延実行）"""
    global ocr_service, drug_service, response_service, redis_service, _services_initialized
    
    if _services_initialized:
        return True
    
    try:
        logger.info("Initializing services...")
        
        from services.ocr_service import OCRService
        from services.drug_service import DrugService
        from services.response_service import ResponseService
        from services.redis_service import RedisService
        
        ocr_service = OCRService()
        drug_service = DrugService()
        response_service = ResponseService()
        
        try:
            redis_service = RedisService()
        except Exception as e:
            logger.warning(f"Redis service initialization failed: {e}")
            redis_service = None
        
        _services_initialized = True
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False

@app.route("/", methods=['GET'])
def root():
    """ルートエンドポイント"""
    return {"status": "ok", "message": "薬局サポートBot is running"}, 200

@app.route("/health", methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント（最小限版）"""
    try:
        # 基本的な環境変数チェックのみ
        if not os.getenv('LINE_CHANNEL_ACCESS_TOKEN'):
            return {"status": "unhealthy", "message": "Missing LINE_CHANNEL_ACCESS_TOKEN"}, 500
        
        if not os.getenv('LINE_CHANNEL_SECRET'):
            return {"status": "unhealthy", "message": "Missing LINE_CHANNEL_SECRET"}, 500
        
        return {"status": "healthy", "message": "薬局サポートBot is running"}, 200
    except Exception as e:
        return {"status": "unhealthy", "message": f"Service error: {str(e)}"}, 500

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
                drug_info = drug_service.get_drug_interactions(user_drug_buffer[user_id])
                response_text = response_service.generate_response(drug_info)
                
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
                        messages=[TextMessage(text="薬剤が登録されていません。画像を送信して薬剤を登録してください。")]
                    )
                )
        
        elif user_message.lower() in ['リスト確認', 'りすとかくにん', 'list']:
            if user_id in user_drug_buffer and user_drug_buffer[user_id]:
                drug_list = "\n".join([f"• {drug}" for drug in user_drug_buffer[user_id]])
                response_text = f"📋 **現在の薬剤リスト**\n\n{drug_list}\n\n💡 「診断」で飲み合わせチェックを実行できます"
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

**📸 薬剤の登録方法：**
1. 処方箋の画像を送信
2. 検出された薬剤を確認
3. 「診断」で飲み合わせチェック

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
• ヘルプ - この使い方表示

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
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
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
        
        # 薬剤名抽出（品質評価済み）
        drug_names = ocr_result['drug_names']
        
        if drug_names:
            # ユーザーバッファに薬剤名を追加
            if user_id not in user_drug_buffer:
                user_drug_buffer[user_id] = []
            
            # マッチした薬剤名をバッファに追加
            matched_drugs = drug_service.match_to_database(drug_names)
            # しきい値: 読み取りが不十分と判断する最小件数
            MIN_CONFIDENT_DRUGS = 5
            if matched_drugs and len(matched_drugs) >= MIN_CONFIDENT_DRUGS:
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
            elif matched_drugs:
                # 件数が少なく精度が不十分と判断 → ガイドを送信
                guide_text = """📸 読み取り精度が十分ではありません

現在の画像からは薬剤名を一部しか認識できませんでした。より正確な診断のため、次をお試しください：

📋 撮影のコツ
• 真上から1ページずつ撮影（左右2ページは分割）
• 明るい場所で、影や反射を避ける
• 文字がはっきり写る距離でピントを合わせる
• iOSの「書類をスキャン」機能の利用を推奨

💊 手動追加（例）
• 薬剤追加：アムロジピン
• 薬剤追加：エソメプラゾール

より鮮明な画像で再度お試しください。"""
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=guide_text)]
                    )
                )
            else:
                response_text = """📸 薬剤名が検出されませんでした

現在の画像では薬剤名を正確に読み取ることができませんでした。

🔍 考えられる原因：
• 画像がぼやけている
• 文字が小さすぎる
• 影や反射で文字が見えにくい
• 処方箋が傾いている
• 複数ページが同時に写っている

📋 改善方法：
1. 真上から撮影（カメラを垂直に）
2. 明るい場所で撮影
3. 1ページずつ撮影（左右2ページは分割）
4. ピントを合わせてから撮影
5. 影や反射を避ける

💡 推奨方法：
• iOSの「書類をスキャン」機能を使用
• カメラアプリで「書類」モードを選択
• 自動補正機能を活用

⚠️ 注意：
• 手書きの処方箋は読み取り精度が低下します
• 画質が悪い場合は手動入力もご検討ください

💊 手動入力方法：
• 「薬剤追加：薬剤名」で手動追加
• 例：「薬剤追加：アムロジピン」

より鮮明な画像で再度お試しください。"""
        else:
            response_text = """📸 薬剤名が検出されませんでした

現在の画像では薬剤名を正確に読み取ることができませんでした。

🔍 考えられる原因：
• 画像がぼやけている
• 文字が小さすぎる
• 影や反射で文字が見えにくい
• 処方箋が傾いている
• 複数ページが同時に写っている

📋 改善方法：
1. 真上から撮影（カメラを垂直に）
2. 明るい場所で撮影
3. 1ページずつ撮影（左右2ページは分割）
4. ピントを合わせてから撮影
5. 影や反射を避ける

💡 推奨方法：
• iOSの「書類をスキャン」機能を使用
• カメラアプリで「書類」モードを選択
• 自動補正機能を活用

⚠️ 注意：
• 手書きの処方箋は読み取り精度が低下します
• 画質が悪い場合は手動入力もご検討ください

💊 手動入力方法：
• 「薬剤追加：薬剤名」で手動追加
• 例：「薬剤追加：アムロジピン」

より鮮明な画像で再度お試しください。"""
        
        # 検出結果を送信（薬剤が検出されなかった場合のみ）
        if not drug_names or not matched_drugs:
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
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting application on port {port}, debug mode: {debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 