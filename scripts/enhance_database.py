#!/usr/bin/env python3
"""
薬剤データベース拡充スクリプト
PMDA/KEGG等の公的データベースから薬剤情報を取得し、分類情報を充実させる
"""

import pandas as pd
import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.drug_service import AIDrugMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseEnhancer:
    """薬剤データベース拡充クラス"""
    
    def __init__(self):
        self.ai_matcher = AIDrugMatcher()
        self.kegg_api_base = "https://rest.kegg.org"
        self.enhanced_data = []
        
    def enhance_existing_database(self, input_file: str = "data/drug_database.csv", 
                                 output_file: str = "data/enhanced_drug_database.csv"):
        """既存データベースを拡充"""
        try:
            # 既存データを読み込み
            df = pd.read_csv(input_file)
            logger.info(f"既存データベース読み込み: {len(df)}件")
            
            # 分類情報が不明な薬剤を特定
            unknown_category_drugs = df[df['薬効分類'] == '不明']
            logger.info(f"分類不明の薬剤: {len(unknown_category_drugs)}件")
            
            # AI分類で補完
            enhanced_df = self._enhance_with_ai_classification(df)
            
            # KEGG情報で補完
            enhanced_df = self._enhance_with_kegg_info(enhanced_df)
            
            # 結果を保存
            enhanced_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"拡充データベース保存: {output_file}")
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"データベース拡充エラー: {e}")
            return None
    
    def _enhance_with_ai_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """AI分類で薬剤分類を補完"""
        logger.info("AI分類による補完開始")
        
        enhanced_df = df.copy()
        updated_count = 0
        
        for idx, row in enhanced_df.iterrows():
            category_value = row['薬効分類']
            if not pd.isna(category_value) and str(category_value) == '不明':
                drug_name = str(row['販売名'])
                analysis = self.ai_matcher.analyze_drug_name(drug_name)
                
                if analysis['category'] != 'unknown':
                    japanese_category = self._map_ai_category_to_japanese(analysis['category'])
                    enhanced_df.at[idx, '薬効分類'] = japanese_category
                    updated_count += 1
                    logger.info(f"AI分類更新: {drug_name} -> {japanese_category}")
        
        logger.info(f"AI分類による更新: {updated_count}件")
        return enhanced_df
    
    def _enhance_with_kegg_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """KEGG情報で薬剤情報を補完"""
        logger.info("KEGG情報による補完開始")
        
        enhanced_df = df.copy()
        updated_count = 0
        
        for idx, row in enhanced_df.iterrows():
            drug_name = str(row['販売名'])
            
            # 分類が不明または基本的な場合のみKEGG検索
            category_value = row['薬効分類']
            if not pd.isna(category_value) and str(category_value) in ['不明', '解熱鎮痛薬', '下剤']:
                kegg_info = self._search_kegg_drug(drug_name)
                if kegg_info:
                    enhanced_df.at[idx, '薬効分類'] = kegg_info['category']
                    generic_value = row['一般名']
                    if not pd.isna(generic_value) and str(generic_value) == '不明':
                        enhanced_df.at[idx, '一般名'] = kegg_info['generic_name']
                    updated_count += 1
                    logger.info(f"KEGG情報更新: {drug_name} -> {kegg_info['category']}")
                
                # レート制限
                time.sleep(0.5)
        
        logger.info(f"KEGG情報による更新: {updated_count}件")
        return enhanced_df
    
    def _search_kegg_drug(self, drug_name: str) -> Optional[Dict[str, str]]:
        """KEGG APIで薬剤情報を検索"""
        try:
            # 英語名を生成
            analysis = self.ai_matcher.analyze_drug_name(drug_name)
            english_variants = analysis.get('english_variants', [])
            
            # 検索パターンを生成
            search_patterns = [drug_name] + english_variants[:3]
            
            for pattern in search_patterns:
                if len(pattern) < 3:
                    continue
                
                search_url = f"{self.kegg_api_base}/find/drug/{pattern}"
                response = requests.get(search_url, timeout=10)
                
                if response.status_code == 200 and response.text.strip():
                    lines = response.text.strip().split('\n')
                    if lines and lines[0]:
                        # 最初の結果を使用
                        kegg_id, description = lines[0].split('\t', 1)
                        
                        # 詳細情報を取得
                        detail_url = f"{self.kegg_api_base}/get/{kegg_id}"
                        detail_response = requests.get(detail_url, timeout=10)
                        
                        if detail_response.status_code == 200:
                            return self._parse_kegg_detail(detail_response.text, drug_name)
                
                time.sleep(0.2)  # レート制限
            
            return None
            
        except Exception as e:
            logger.warning(f"KEGG検索エラー ({drug_name}): {e}")
            return None
    
    def _parse_kegg_detail(self, kegg_text: str, original_name: str) -> Dict[str, str]:
        """KEGG詳細情報を解析"""
        try:
            lines = kegg_text.strip().split('\n')
            
            # 基本情報を抽出
            generic_name = original_name
            category = '不明'
            
            for line in lines:
                if line.startswith('NAME'):
                    # 英語名を抽出
                    name_part = line.split('NAME', 1)[1].strip()
                    if ';' in name_part:
                        generic_name = name_part.split(';')[0].strip()
                
                elif line.startswith('CLASS'):
                    # 分類情報を抽出
                    class_part = line.split('CLASS', 1)[1].strip()
                    category = self._extract_category_from_kegg_class(class_part)
            
            return {
                'generic_name': generic_name,
                'category': category
            }
            
        except Exception as e:
            logger.warning(f"KEGG詳細解析エラー: {e}")
            return {
                'generic_name': original_name,
                'category': '不明'
            }
    
    def _extract_category_from_kegg_class(self, kegg_class: str) -> str:
        """KEGG分類から日本語分類を抽出"""
        kegg_class_lower = kegg_class.lower()
        
        # 分類マッピング
        category_mapping = {
            'benzodiazepine': 'ベンゾジアゼピン系',
            'barbiturate': 'バルビツール酸系',
            'opioid': 'オピオイド系',
            'nsaid': 'NSAIDs',
            'statin': 'スタチン系',
            'ace inhibitor': 'ACE阻害薬',
            'angiotensin': 'ARB',
            'beta blocker': 'β遮断薬',
            'calcium channel': 'カルシウム拮抗薬',
            'diuretic': '利尿薬',
            'antihistamine': '抗ヒスタミン薬',
            'antacid': '制酸薬',
            'analgesic': '解熱鎮痛薬',
            'antibiotic': '抗生物質',
            'antiviral': '抗ウイルス薬',
            'antifungal': '抗真菌薬',
            'antidepressant': '抗うつ薬',
            'antipsychotic': '抗精神病薬',
            'antianxiety': '抗不安薬',
            'sedative': '鎮静薬',
            'hypnotic': '睡眠薬'
        }
        
        for english_key, japanese_category in category_mapping.items():
            if english_key in kegg_class_lower:
                return japanese_category
        
        return '不明'
    
    def _map_ai_category_to_japanese(self, ai_category: str) -> str:
        """AI分類を日本語に変換"""
        category_mapping = {
            'benzodiazepines': 'ベンゾジアゼピン系',
            'barbiturates': 'バルビツール酸系',
            'opioids': 'オピオイド系',
            'nsaids': 'NSAIDs',
            'statins': 'スタチン系',
            'ace_inhibitors': 'ACE阻害薬',
            'arb': 'ARB',
            'beta_blockers': 'β遮断薬',
            'calcium_blockers': 'カルシウム拮抗薬',
            'diuretics': '利尿薬',
            'antihistamines': '抗ヒスタミン薬',
            'antacids': '制酸薬',
            'unknown': '不明'
        }
        return category_mapping.get(ai_category, ai_category)
    
    def create_sample_enhanced_data(self):
        """サンプル拡充データの作成"""
        sample_data = [
            # ベンゾジアゼピン系
            {'販売名': 'アルプラゾラム', '一般名': 'アルプラゾラム', '薬効分類': 'ベンゾジアゼピン系', '相互作用': 'アルコール,他の向精神薬'},
            {'販売名': 'ジアゼパム', '一般名': 'ジアゼパム', '薬効分類': 'ベンゾジアゼピン系', '相互作用': 'アルコール,他の向精神薬'},
            {'販売名': 'クロナゼパム', '一般名': 'クロナゼパム', '薬効分類': 'ベンゾジアゼピン系', '相互作用': 'アルコール,他の向精神薬'},
            {'販売名': 'ロラゼパム', '一般名': 'ロラゼパム', '薬効分類': 'ベンゾジアゼピン系', '相互作用': 'アルコール,他の向精神薬'},
            {'販売名': 'エスタゾラム', '一般名': 'エスタゾラム', '薬効分類': 'ベンゾジアゼピン系', '相互作用': 'アルコール,他の向精神薬'},
            
            # ARB
            {'販売名': 'バルサルタン', '一般名': 'バルサルタン', '薬効分類': 'ARB', '相互作用': 'カリウム保持性利尿薬,NSAIDs'},
            {'販売名': 'ロサルタン', '一般名': 'ロサルタン', '薬効分類': 'ARB', '相互作用': 'カリウム保持性利尿薬,NSAIDs'},
            {'販売名': 'カンデサルタン', '一般名': 'カンデサルタン', '薬効分類': 'ARB', '相互作用': 'カリウム保持性利尿薬,NSAIDs'},
            
            # 利尿薬
            {'販売名': 'スピロノラクトン', '一般名': 'スピロノラクトン', '薬効分類': '利尿薬', '相互作用': 'カリウム製剤,ACE阻害薬'},
            {'販売名': 'フロセミド', '一般名': 'フロセミド', '薬効分類': '利尿薬', '相互作用': 'ジギタリス,アミノグリコシド系抗生物質'},
            
            # NSAIDs
            {'販売名': 'イブプロフェン', '一般名': 'イブプロフェン', '薬効分類': 'NSAIDs', '相互作用': 'ワルファリン,抗凝固薬'},
            {'販売名': 'ロキソプロフェン', '一般名': 'ロキソプロフェン', '薬効分類': 'NSAIDs', '相互作用': 'ワルファリン,抗凝固薬'},
            {'販売名': 'ジクロフェナク', '一般名': 'ジクロフェナク', '薬効分類': 'NSAIDs', '相互作用': 'ワルファリン,抗凝固薬'},
            
            # スタチン系
            {'販売名': 'シンバスタチン', '一般名': 'シンバスタチン', '薬効分類': 'スタチン系', '相互作用': 'グレープフルーツジュース,抗真菌薬'},
            {'販売名': 'アトルバスタチン', '一般名': 'アトルバスタチン', '薬効分類': 'スタチン系', '相互作用': 'グレープフルーツジュース,抗真菌薬'},
            
            # β遮断薬
            {'販売名': 'プロプラノロール', '一般名': 'プロプラノロール', '薬効分類': 'β遮断薬', '相互作用': 'カルシウム拮抗薬,抗不整脈薬'},
            {'販売名': 'アテノロール', '一般名': 'アテノロール', '薬効分類': 'β遮断薬', '相互作用': 'カルシウム拮抗薬,抗不整脈薬'},
        ]
        
        df = pd.DataFrame(sample_data)
        output_file = "data/enhanced_drug_database.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"サンプル拡充データ作成: {output_file}")
        return df

def main():
    """メイン処理"""
    enhancer = DatabaseEnhancer()
    
    # 既存データベースの拡充
    if os.path.exists("data/drug_database.csv"):
        logger.info("既存データベースの拡充を開始")
        enhanced_df = enhancer.enhance_existing_database()
        if enhanced_df is not None:
            logger.info(f"拡充完了: {len(enhanced_df)}件")
        else:
            logger.error("拡充に失敗しました")
    else:
        logger.info("既存データベースが見つからないため、サンプルデータを作成")
        enhancer.create_sample_enhanced_data()

if __name__ == "__main__":
    main() 