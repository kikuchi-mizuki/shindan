import os
import logging
from flask import Flask

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/", methods=['GET'])
def root():
    """ルートエンドポイント"""
    logger.info("Root endpoint accessed")
    return {"status": "ok", "message": "Debug app is running"}, 200

@app.route("/health", methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "message": "ok"}, 200

@app.route("/test-imports", methods=['GET'])
def test_imports():
    """インポートテストエンドポイント"""
    logger.info("Testing imports...")
    
    try:
        # 基本的なインポートテスト
        logger.info("Testing basic imports...")
        import numpy as np
        logger.info("✓ numpy imported successfully")
        
        import pandas as pd
        logger.info("✓ pandas imported successfully")
        
        import cv2
        logger.info("✓ opencv imported successfully")
        
        from PIL import Image
        logger.info("✓ PIL imported successfully")
        
        return {"status": "success", "message": "All basic imports successful"}, 200
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        return {"status": "error", "message": f"Import failed: {str(e)}"}, 500

@app.route("/test-services", methods=['GET'])
def test_services():
    """サービス初期化テストエンドポイント"""
    logger.info("Testing service initialization...")
    
    try:
        # サービス初期化テスト
        logger.info("Testing OCR service...")
        from services.ocr_service import OCRService
        ocr_service = OCRService()
        logger.info("✓ OCR service initialized successfully")
        
        logger.info("Testing drug service...")
        from services.drug_service import DrugService
        drug_service = DrugService()
        logger.info("✓ Drug service initialized successfully")
        
        logger.info("Testing response service...")
        from services.response_service import ResponseService
        response_service = ResponseService()
        logger.info("✓ Response service initialized successfully")
        
        return {"status": "success", "message": "All services initialized successfully"}, 200
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return {"status": "error", "message": f"Service initialization failed: {str(e)}"}, 500

@app.route("/test-line-bot", methods=['GET'])
def test_line_bot():
    """LINE Bot設定テストエンドポイント"""
    logger.info("Testing LINE Bot configuration...")
    
    try:
        # LINE Bot設定テスト
        logger.info("Testing LINE Bot imports...")
        from linebot.v3 import WebhookHandler
        from linebot.v3.exceptions import InvalidSignatureError
        from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
        from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, MessagingApiBlob, TextMessage, ReplyMessageRequest, PushMessageRequest
        logger.info("✓ LINE Bot imports successful")
        
        # 環境変数チェック
        logger.info("Testing environment variables...")
        access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
        channel_secret = os.getenv('LINE_CHANNEL_SECRET')
        
        if not access_token:
            raise Exception("LINE_CHANNEL_ACCESS_TOKEN not found")
        if not channel_secret:
            raise Exception("LINE_CHANNEL_SECRET not found")
            
        logger.info("✓ Environment variables found")
        
        # LINE Bot設定
        logger.info("Testing LINE Bot configuration...")
        configuration = Configuration(access_token=access_token)
        api_client = ApiClient(configuration)
        messaging_api = MessagingApi(api_client)
        messaging_blob_api = MessagingApiBlob(api_client)
        handler = WebhookHandler(channel_secret)
        logger.info("✓ LINE Bot configuration successful")
        
        return {"status": "success", "message": "LINE Bot configuration successful"}, 200
        
    except Exception as e:
        logger.error(f"LINE Bot test failed: {e}")
        return {"status": "error", "message": f"LINE Bot test failed: {str(e)}"}, 500

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting debug application on port {port}, debug mode: {debug_mode}")
        logger.info("Debug application startup completed successfully")
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Debug application startup failed: {e}")
        raise
