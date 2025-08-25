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
        
        # 基本的な応答
        response_text = "画像を受け取りました。現在、画像処理機能は準備中です。"
        
        messaging_api.reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text=response_text)]
            )
        )
        
    except Exception as e:
        logger.error(f"Image message handling error: {e}")

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting LINE Bot application on port {port}, debug mode: {debug_mode}")
        logger.info("LINE Bot application startup completed successfully")
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 