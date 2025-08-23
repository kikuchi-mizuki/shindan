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
        
        # 基本的なテキストメッセージ処理
        if user_message.lower() in ['診断', 'しんだん', 'diagnosis']:
            if user_id in user_drug_buffer and user_drug_buffer[user_id]:
                response_text = "診断機能は準備中です。"
            else:
                response_text = "薬剤リストが空です。画像を送信して薬剤を登録してください。"
        elif user_message.startswith('薬剤追加：'):
            drug_name = user_message.replace('薬剤追加：', '').strip()
            if drug_name:
                if user_id not in user_drug_buffer:
                    user_drug_buffer[user_id] = []
                if drug_name not in user_drug_buffer[user_id]:
                    user_drug_buffer[user_id].append(drug_name)
                    response_text = f"✅ 薬剤「{drug_name}」を追加しました。"
                else:
                    response_text = f"⚠️ 薬剤「{drug_name}」は既にリストに含まれています。"
            else:
                response_text = "❌ 薬剤名を入力してください。"
        else:
            response_text = "薬局サポートBotへようこそ！\n\n画像を送信して薬剤を登録するか、以下のコマンドを使用してください：\n• 診断 - 飲み合わせチェック\n• 薬剤追加：〇〇 - 薬剤を手動追加"
        
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
        
        # 画像処理（簡略化）
        response_text = "画像処理機能は準備中です。\n\n薬剤名を手動で追加する場合は「薬剤追加：〇〇」の形式で入力してください。"
        
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
                    to=event.source.user_id,
                    messages=[TextMessage(text="画像処理エラーが発生しました。しばらく時間をおいて再度お試しください。")]
                )
            )
        except:
            pass

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting application on port {port}, debug mode: {debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 