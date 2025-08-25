import os
import logging
from flask import Flask, request, abort

from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
            "message": "Basic test passed"
        }, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting minimal application on port {port}, debug mode: {debug_mode}")
        logger.info("Minimal application startup completed successfully")
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 