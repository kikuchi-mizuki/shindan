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
    return {"status": "ok", "message": "Minimal app is running"}, 200

@app.route("/health", methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "message": "ok"}, 200

@app.route("/test", methods=['GET'])
def test():
    """テストエンドポイント"""
    logger.info("Test endpoint accessed")
    return {"status": "test", "message": "Test successful"}, 200

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting minimal application on port {port}, debug mode: {debug_mode}")
        logger.info("Minimal application startup completed successfully")
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Minimal application startup failed: {e}")
        raise
