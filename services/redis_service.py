import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RedisService:
    """Redisを使用したユーザーデータ永続化サービス"""
    
    def __init__(self):
        self.redis_client = None
        self.redis_available = False
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Redisクライアントの初期化"""
        try:
            import redis
            # 環境変数からRedis接続情報を取得
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # 接続テスト
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connection established successfully")
            
        except ImportError:
            logger.warning("Redis library not available. Using in-memory storage.")
            self.redis_available = False
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_available = False
    
    def _get_user_key(self, user_id: str, data_type: str) -> str:
        """ユーザー固有のキーを生成"""
        return f"user:{user_id}:{data_type}"
    
    def save_user_drugs(self, user_id: str, drug_names: List[str]) -> bool:
        """ユーザーの薬剤リストを保存"""
        try:
            if not self.redis_available:
                logger.warning("Redis not available, skipping save")
                return False
            
            key = self._get_user_key(user_id, "drugs")
            data = {
                "drug_names": drug_names,
                "updated_at": datetime.now().isoformat(),
                "count": len(drug_names)
            }
            
            self.redis_client.setex(key, 86400 * 30, json.dumps(data))  # 30日間保存
            logger.info(f"Saved {len(drug_names)} drugs for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving user drugs: {e}")
            return False
    
    def get_user_drugs(self, user_id: str) -> List[str]:
        """ユーザーの薬剤リストを取得"""
        try:
            if not self.redis_available:
                logger.warning("Redis not available, returning empty list")
                return []
            
            key = self._get_user_key(user_id, "drugs")
            data = self.redis_client.get(key)
            
            if data:
                parsed_data = json.loads(data)
                drug_names = parsed_data.get("drug_names", [])
                logger.info(f"Retrieved {len(drug_names)} drugs for user {user_id}")
                return drug_names
            else:
                logger.info(f"No drugs found for user {user_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving user drugs: {e}")
            return []
    
    def save_diagnosis_history(self, user_id: str, diagnosis_result: Dict[str, Any]) -> bool:
        """診断履歴を保存"""
        try:
            if not self.redis_available:
                logger.warning("Redis not available, skipping save")
                return False
            
            key = self._get_user_key(user_id, "diagnosis_history")
            
            # 既存の履歴を取得
            existing_data = self.redis_client.get(key)
            history = []
            if existing_data:
                history = json.loads(existing_data)
            
            # 新しい診断結果を追加
            diagnosis_entry = {
                "timestamp": datetime.now().isoformat(),
                "drug_count": len(diagnosis_result.get("detected_drugs", [])),
                "has_interactions": len(diagnosis_result.get("interactions", [])) > 0,
                "has_warnings": len(diagnosis_result.get("warnings", [])) > 0,
                "drug_names": [drug.get("name", "") for drug in diagnosis_result.get("detected_drugs", [])]
            }
            
            history.append(diagnosis_entry)
            
            # 最新10件のみ保持
            if len(history) > 10:
                history = history[-10:]
            
            self.redis_client.setex(key, 86400 * 90, json.dumps(history))  # 90日間保存
            logger.info(f"Saved diagnosis history for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving diagnosis history: {e}")
            return False
    
    def get_diagnosis_history(self, user_id: str) -> List[Dict[str, Any]]:
        """診断履歴を取得"""
        try:
            if not self.redis_available:
                logger.warning("Redis not available, returning empty history")
                return []
            
            key = self._get_user_key(user_id, "diagnosis_history")
            data = self.redis_client.get(key)
            
            if data:
                history = json.loads(data)
                logger.info(f"Retrieved {len(history)} diagnosis records for user {user_id}")
                return history
            else:
                logger.info(f"No diagnosis history found for user {user_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving diagnosis history: {e}")
            return []
    
    def add_drug_to_user(self, user_id: str, drug_name: str) -> bool:
        """ユーザーの薬剤リストに薬剤を追加"""
        try:
            current_drugs = self.get_user_drugs(user_id)
            if drug_name not in current_drugs:
                current_drugs.append(drug_name)
                return self.save_user_drugs(user_id, current_drugs)
            return True
            
        except Exception as e:
            logger.error(f"Error adding drug to user: {e}")
            return False
    
    def remove_drug_from_user(self, user_id: str, drug_name: str) -> bool:
        """ユーザーの薬剤リストから薬剤を削除"""
        try:
            current_drugs = self.get_user_drugs(user_id)
            if drug_name in current_drugs:
                current_drugs.remove(drug_name)
                return self.save_user_drugs(user_id, current_drugs)
            return True
            
        except Exception as e:
            logger.error(f"Error removing drug from user: {e}")
            return False
    
    def clear_user_drugs(self, user_id: str) -> bool:
        """ユーザーの薬剤リストをクリア"""
        try:
            if not self.redis_available:
                return True
            
            key = self._get_user_key(user_id, "drugs")
            self.redis_client.delete(key)
            logger.info(f"Cleared drugs for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing user drugs: {e}")
            return False
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """ユーザーの統計情報を取得"""
        try:
            drugs = self.get_user_drugs(user_id)
            history = self.get_diagnosis_history(user_id)
            
            stats = {
                "total_drugs": len(drugs),
                "total_diagnoses": len(history),
                "last_diagnosis": None,
                "most_common_drugs": [],
                "interaction_count": 0
            }
            
            if history:
                stats["last_diagnosis"] = history[-1]["timestamp"]
                stats["interaction_count"] = sum(1 for h in history if h.get("has_interactions", False))
            
            # 最も頻繁に使用される薬剤を計算
            drug_counts = {}
            for entry in history:
                for drug_name in entry.get("drug_names", []):
                    drug_counts[drug_name] = drug_counts.get(drug_name, 0) + 1
            
            stats["most_common_drugs"] = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}
    
    def is_redis_available(self) -> bool:
        """Redisが利用可能かどうかを確認"""
        return self.redis_available 