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

def initialize_services():
    """サービスの初期化（遅延実行）"""
    global ocr_service, _services_initialized
    
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
        
        # 基本的な応答
        response_text = "薬局サポートBotへようこそ！\n\n画像を送信して薬剤を登録するか、以下のコマンドを使用してください：\n• 診断 - 飲み合わせチェック\n• ヘルプ - 使い方表示"
        
        messaging_api.reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text=response_text)]
            )
        )
        
    except Exception as e:
        logger.error(f"Text message handling error: {e}")

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
        
        # 薬剤名抽出（品質評価済み）
        drug_names = ocr_result['drug_names']
        
        if drug_names:
            # 基本的な検出結果を表示
            drug_list = "\n".join([f"• {drug}" for drug in drug_names])
            response_text = f"📋 **検出された薬剤**\n\n{drug_list}\n\n💡 現在、画像品質評価システムが動作中です。"
            
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
        
        logger.info(f"Starting LINE Bot application with image quality assessment on port {port}, debug mode: {debug_mode}")
        logger.info("LINE Bot application with image quality assessment startup completed successfully")
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 