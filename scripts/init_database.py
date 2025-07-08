#!/usr/bin/env python3
"""
薬局サポートBot データベース初期化スクリプト
"""

import sys
import os
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pmda_downloader import PMDADownloader

def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("薬局サポートBot データベース初期化を開始します...")
    
    try:
        # PMDAダウンローダーの初期化
        downloader = PMDADownloader()
        
        # データベースの更新
        processed_data = downloader.download_drug_data()
        if processed_data is not None and not processed_data.empty:
            logger.info("データベースの初期化が完了しました")
            # データベース情報の表示
            processed_file = os.path.join(downloader.data_dir, 'processed_drug_database.csv')
            logger.info(f"データベース情報:")
            logger.info(f"  ファイル: {processed_file}")
            logger.info(f"  レコード数: {len(processed_data)}")
            logger.info(f"  最終更新: {datetime.fromtimestamp(os.path.getmtime(processed_file)).isoformat()}")
        else:
            logger.error("データベースの初期化に失敗しました")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"初期化中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 