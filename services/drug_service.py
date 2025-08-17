import pandas as pd
import logging
import os
import requests
import json
from typing import List, Dict, Any, Optional
import difflib
import re
from collections import defaultdict
import itertools
import time  # 追加：レート制限用

logger = logging.getLogger(__name__)

# rapidfuzzの代替実装（既存のまま）
def fuzz_ratio(s1, s2):
    if not s1 or not s2:
        return 0
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 100
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0
    # 簡易的な類似度計算
    common_chars = sum(1 for c in s1 if c in s2)
    return int((common_chars * 2) / (len1 + len2) * 100)

def fuzz_partial_ratio(s1, s2):
    if not s1 or not s2:
        return 0
    s1, s2 = s1.lower(), s2.lower()
    if s1 in s2 or s2 in s1:
        return 100
    return fuzz_ratio(s1, s2)

def fuzz_token_sort_ratio(s1, s2):
    if not s1 or not s2:
        return 0
    # 簡易的なトークンソート
    tokens1 = sorted(s1.lower().split())
    tokens2 = sorted(s2.lower().split())
    return fuzz_ratio(' '.join(tokens1), ' '.join(tokens2))

def fuzz_token_set_ratio(s1, s2):
    if not s1 or not s2:
        return 0
    # 簡易的なトークンセット
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0
    return int(len(intersection) / len(union) * 100)

class fuzz:
    ratio = fuzz_ratio
    partial_ratio = fuzz_partial_ratio
    token_sort_ratio = fuzz_token_sort_ratio
    token_set_ratio = fuzz_token_set_ratio

def process_func(query, choices, limit=5):
    if not query or not choices:
        return []
    results = []
    for choice in choices:
        score = fuzz.ratio(query, choice)
        if score > 0:
            results.append((choice, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]

class process:
    @staticmethod
    def extractOne(query, choices, scorer=None):
        if not query or not choices:
            return None
        if scorer is None:
            scorer = fuzz.ratio
        best_score = 0
        best_choice = None
        for choice in choices:
            score = scorer(query, choice)
            if score > best_score:
                best_score = score
                best_choice = choice
        return (best_choice, best_score) if best_choice else None

class AIDrugMatcher:
    """AIを活用した薬剤マッチングシステム"""
    
    def __init__(self):
        # 分析結果のキャッシュ
        self.analysis_cache = {}
        self.cache_max_size = 1000  # 最大キャッシュサイズ
        
        self.drug_patterns = {
            'benzodiazepines': [
                'パム', 'ラム', 'ゾラム', 'ジアゼパム', 'クロナゼパム', 'アルプラゾラム', 'ロラゼパム',
                'テマゼパム', 'ミダゾラム', 'エスタゾラム', 'フルラゼパム', 'ニトラゼパム', 'ブロマゼパム',
                'クロチアゼパム', 'クロキサゾラム', 'ハロキサゾラム', 'メキサゾラム', 'オキサゼパム',
                'オキサゾラム', 'プラゼパム', 'トリアゾラム', 'エチゾラム', 'フルニトラゼパム',
                'ブロチゾラム', 'ジアゼ', 'クロナ', 'アルプラ', 'ロラ', 'テマ', 'ミダ', 'エスタ',
                'フルラ', 'ニトラ', 'ブロマ', 'クロチ', 'クロキ', 'ハロキ', 'メキ', 'オキサ',
                'プラ', 'トリア', 'エチ', 'フルニ', 'ブロチ', 'xanax', 'valium', 'klonopin',
                'ativan', 'restoril', 'versed', 'prosom', 'dalmane', 'mogadon', 'lexotan',
                'lendormin', 'rize', 'sepazon', 'somelin', 'melex', 'serenal', 'halcion',
                'sedekopan'
            ],
            'barbiturates': [
                'バルビタール', 'バルビタル', 'フェノバルビタール', 'アモバルビタール',
                'ペントバルビタール', 'チオペンタール', 'バルビ', 'フェノバル', 'アモバル',
                'ペントバル', 'チオペン', 'phenobarbital', 'amobarbital', 'pentobarbital',
                'thiopental'
            ],
            'opioids': [
                'モルヒネ', 'コデイン', 'フェンタニル', 'オキシコドン', 'ヒドロコドン',
                'トラマドール', 'ペンタゾシン', 'ブプレノルフィン', 'メタドン', 'モル',
                'コデ', 'フェンタ', 'オキシ', 'ヒドロ', 'トラマ', 'ペンタ', 'ブプレ',
                'メタ', 'morphine', 'codeine', 'fentanyl', 'oxycodone', 'hydrocodone',
                'tramadol', 'pentazocine', 'buprenorphine', 'methadone'
            ],
            'nsaids': [
                'アスピリン', 'イブプロフェン', 'ロキソプロフェン', 'ジクロフェナク',
                'メフェナム酸', 'インドメタシン', 'ナプロキセン', 'ケトプロフェン',
                'セレコキシブ', 'メロキシカム', 'アスピ', 'イブプロ', 'ロキソ', 'ジクロ',
                'メフェ', 'インド', 'ナプロ', 'ケトプロ', 'セレコ', 'メロキ', 'aspirin',
                'ibuprofen', 'loxoprofen', 'diclofenac', 'mefenamic', 'indomethacin',
                'naproxen', 'ketoprofen', 'celecoxib', 'meloxicam'
            ],
            'statins': [
                'スタチン', 'シンバスタチン', 'アトルバスタチン', 'プラバスタチン',
                'ロスバスタチン', 'フルバスタチン', 'ピタバスタチン', 'シンバ',
                'アトルバ', 'プラバ', 'ロスバ', 'フルバ', 'ピタバ', 'simvastatin',
                'atorvastatin', 'pravastatin', 'rosuvastatin', 'fluvastatin',
                'pitavastatin'
            ],
            'ace_inhibitors': [
                'プリル', 'カプトプリル', 'エナラプリル', 'リシノプリル', 'ペリンドプリル',
                'キナプリル', 'トランドラプリル', 'カプト', 'エナラ', 'リシノ', 'ペリンド',
                'キナ', 'トランドラ', 'captopril', 'enalapril', 'lisinopril',
                'perindopril', 'quinapril', 'trandolapril'
            ],
            'arb': [
                'サルタン', 'ロサルタン', 'カンデサルタン', 'バルサルタン', 'イルベサルタン',
                'オルメサルタン', 'テルミサルタン', 'アジルサルタン', 'ロサ', 'カンデ',
                'バルサ', 'イルベ', 'オルメ', 'テルミ', 'アジル', 'losartan',
                'candesartan', 'valsartan', 'irbesartan', 'olmesartan', 'telmisartan',
                'azilsartan'
            ],
            'beta_blockers': [
                'ロール', 'プロプラノロール', 'アテノロール', 'ビソプロロール',
                'メトプロロール', 'カルベジロール', 'ネビボロール', 'プロプラ',
                'アテノ', 'ビソプロ', 'メトプロ', 'カルベジ', 'ネビボ', 'propranolol',
                'atenolol', 'bisoprolol', 'metoprolol', 'carvedilol', 'nebivolol'
            ],
            'calcium_blockers': [
                'ジピン', 'ニフェジピン', 'アムロジピン', 'ベラパミル', 'ジルチアゼム',
                'ニカルジピン', 'ニソルジピン', 'ニフェ', 'アムロ', 'ベラパ', 'ジルチ',
                'ニカル', 'ニソル', 'nifedipine', 'amlodipine', 'verapamil',
                'diltiazem', 'nicardipine', 'nisoldipine'
            ],
            'diuretics': [
                'サイド', 'フロセミド', 'ヒドロクロロチアジド', 'スピロノラクトン',
                'トリアムテレン', 'アミロライド', 'ブメタニド', 'フロセ', 'ヒドロ',
                'スピロ', 'トリアム', 'アミロ', 'ブメタ', 'furosemide',
                'hydrochlorothiazide', 'spironolactone', 'triamterene',
                'amiloride', 'bumetanide'
            ],
            'antihistamines': [
                'フェニラミン', 'クロルフェニラミン', 'ジフェンヒドラミン',
                'セチリジン', 'ロラタジン', 'フェキソフェナジン', 'フェニラ',
                'クロルフェニ', 'ジフェン', 'セチリ', 'ロラタ', 'フェキソ',
                'pheniramine', 'chlorpheniramine', 'diphenhydramine',
                'cetirizine', 'loratadine', 'fexofenadine'
            ],
            'antacids': [
                'アルミニウム', 'マグネシウム', 'カルシウム', '水酸化', '炭酸',
                'aluminum', 'magnesium', 'calcium', 'hydroxide', 'carbonate'
            ]
        }
        
        self.common_suffixes = [
            'パム', 'ラム', 'ゾラム', 'ロン', 'ピン', 'サルタン', 'プリル', 'ロール', 'サイド',
            'フェン', 'ゾール', 'プロフェン', 'シン', 'タン', 'ジン', 'ミン', 'リン', 'チン'
        ]
        self.common_prefixes = [
            'アセト', 'メト', 'プロ', 'ジ', 'トリア', 'クロナ', 'ジアゼ', 'アルプラ',
            'ロラ', 'テマ', 'ミダ', 'エスタ', 'フルラ', 'ニトラ', 'ブロマ', 'クロチ',
            'クロキ', 'ハロキ', 'メキ', 'オキサ', 'プラ', 'エチ', 'フルニ', 'ブロチ'
        ]
        
    def analyze_drug_name(self, drug_name: str) -> Dict[str, Any]:
        """薬剤名をAI的に分析（キャッシュ対応版）"""
        # キャッシュチェック
        if drug_name in self.analysis_cache:
            logger.info(f"Using cached analysis for '{drug_name}'")
            return self.analysis_cache[drug_name]
        
        analysis = {
            'original': drug_name,
            'normalized': self._normalize_name(drug_name),
            'category': self._predict_category(drug_name),
            'confidence': 0.0,
            'search_priority': [],
            'english_variants': self._generate_english_variants(drug_name)
        }
        
        # 信頼度スコアの計算
        analysis['confidence'] = self._calculate_confidence(drug_name, analysis)
        
        # 検索優先度の決定
        analysis['search_priority'] = self._determine_search_priority(drug_name, analysis)
        
        # キャッシュに保存
        self._cache_analysis(drug_name, analysis)
        
        return analysis
    
    def _cache_analysis(self, drug_name: str, analysis: Dict[str, Any]):
        """分析結果をキャッシュに保存"""
        # キャッシュサイズ制限
        if len(self.analysis_cache) >= self.cache_max_size:
            # 最も古いエントリを削除（簡易的なLRU）
            oldest_key = next(iter(self.analysis_cache))
            del self.analysis_cache[oldest_key]
        
        self.analysis_cache[drug_name] = analysis
    
    def _normalize_name(self, name: str) -> str:
        """薬剤名の正規化（強化版）"""
        if not name:
            return ''
        
        # 基本的な正規化
        normalized = name.strip()
        
        # 全角→半角変換
        normalized = re.sub(r'[Ａ-Ｚａ-ｚ０-９]', lambda x: chr(ord(x.group(0)) - 0xFEE0), normalized)
        
        # ひらがな→カタカナ変換（薬剤名はカタカナ表記が標準）
        hiragana_to_katakana = str.maketrans({
            'あ': 'ア', 'い': 'イ', 'う': 'ウ', 'え': 'エ', 'お': 'オ',
            'か': 'カ', 'き': 'キ', 'く': 'ク', 'け': 'ケ', 'こ': 'コ',
            'さ': 'サ', 'し': 'シ', 'す': 'ス', 'せ': 'セ', 'そ': 'ソ',
            'た': 'タ', 'ち': 'チ', 'つ': 'ツ', 'て': 'テ', 'と': 'ト',
            'な': 'ナ', 'に': 'ニ', 'ぬ': 'ヌ', 'ね': 'ネ', 'の': 'ノ',
            'は': 'ハ', 'ひ': 'ヒ', 'ふ': 'フ', 'へ': 'ヘ', 'ほ': 'ホ',
            'ま': 'マ', 'み': 'ミ', 'む': 'ム', 'め': 'メ', 'も': 'モ',
            'や': 'ヤ', 'ゆ': 'ユ', 'よ': 'ヨ',
            'ら': 'ラ', 'り': 'リ', 'る': 'ル', 'れ': 'レ', 'ろ': 'ロ',
            'わ': 'ワ', 'を': 'ヲ', 'ん': 'ン',
            'が': 'ガ', 'ぎ': 'ギ', 'ぐ': 'グ', 'げ': 'ゲ', 'ご': 'ゴ',
            'ざ': 'ザ', 'じ': 'ジ', 'ず': 'ズ', 'ぜ': 'ゼ', 'ぞ': 'ゾ',
            'だ': 'ダ', 'ぢ': 'ヂ', 'づ': 'ヅ', 'で': 'デ', 'ど': 'ド',
            'ば': 'バ', 'び': 'ビ', 'ぶ': 'ブ', 'べ': 'ベ', 'ぼ': 'ボ',
            'ぱ': 'パ', 'ぴ': 'ピ', 'ぷ': 'プ', 'ぺ': 'ペ', 'ぽ': 'ポ',
            'ゃ': 'ャ', 'ゅ': 'ュ', 'ょ': 'ョ', 'っ': 'ッ'
        })
        normalized = normalized.translate(hiragana_to_katakana)
        
        # 括弧とその中身を除去
        normalized = re.sub(r'[\(\)（）].*?[\(\)（）]', '', normalized)
        normalized = re.sub(r'[\(\)（）]', '', normalized)
        
        # 剤形の除去（より詳細）
        dosage_forms = [
            '錠', 'カプセル', '散', '液', '注射', '軟膏', 'クリーム', 'テープ', '坐剤', '点眼',
            'mg', 'g', 'ml', 'L', 'μg', 'mcg', 'IU', '単位'
        ]
        for form in dosage_forms:
            # 剤形の前の数字も含めて除去
            normalized = re.sub(rf'\d*{re.escape(form)}', '', normalized, flags=re.IGNORECASE)
        
        # 製薬会社名の除去
        company_patterns = [
            r'「[^」]*」',  # 「」で囲まれた文字列
            r'\[[^\]]*\]',  # []で囲まれた文字列
            r'\([^)]*\)',   # ()で囲まれた文字列
        ]
        for pattern in company_patterns:
            normalized = re.sub(pattern, '', normalized)
        
        # 記号・空白の除去
        normalized = re.sub(r'[\s\-・,，、.。()（）【】［］\[\]{}｛｝<>《》"\'\-―ー=+*/\\]', '', normalized)
        
        # 複数スペースを単一に
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _predict_category(self, drug_name: str) -> str:
        """薬剤カテゴリの予測（修正版）"""
        drug_lower = drug_name.lower()
        normalized_name = self._normalize_name(drug_name).lower()
        
        # 正確な薬剤分類マッピング
        exact_drug_mapping = {
            # PDE5阻害薬
            'タダラフィル': 'pde5_inhibitor',
            'シルデナフィル': 'pde5_inhibitor',
            'バルデナフィル': 'pde5_inhibitor',
            'アバナフィル': 'pde5_inhibitor',
            'ウデナフィル': 'pde5_inhibitor',
            
            # 硝酸薬
            'ニコランジル': 'nitrate',
            'ニトログリセリン': 'nitrate',
            'イソソルビド': 'nitrate',
            'ニトロプルシド': 'nitrate',
            
            # ARNI（心不全治療薬）
            'エンレスト': 'arni',
            'サクビトリル': 'arni',
            'バルサルタン': 'arb',  # ARNIの成分
            
            # Ca拮抗薬+ARB配合剤
            'テラムロ': 'ca_antagonist_arb_combination',
            'アムロジピン': 'ca_antagonist',
            'テルミサルタン': 'arb',
            
            # ACE阻害薬
            'エナラプリル': 'ace_inhibitor',
            'カプトプリル': 'ace_inhibitor',
            'リシノプリル': 'ace_inhibitor',
            'ペリンドプリル': 'ace_inhibitor',
            
            # P-CAB（胃薬）
            'タケキャブ': 'p_cab',
            'ボノプラザン': 'p_cab',
            'フォノプラザン': 'p_cab',
            
            # PPI（胃薬）
            'ランソプラゾール': 'ppi',
            'オメプラゾール': 'ppi',
            'エソメプラゾール': 'ppi',
            'ラベプラゾール': 'ppi',
            'パントプラゾール': 'ppi',
            
            # ベンゾジアゼピン系（正確な分類）
            'ジアゼパム': 'benzodiazepine',
            'クロナゼパム': 'benzodiazepine',
            'アルプラゾラム': 'benzodiazepine',
            'ロラゼパム': 'benzodiazepine',
            'テマゼパム': 'benzodiazepine',
            'ミダゾラム': 'benzodiazepine',
            'エスタゾラム': 'benzodiazepine',
            'フルラゼパム': 'benzodiazepine',
            'ニトラゼパム': 'benzodiazepine',
            'ブロマゼパム': 'benzodiazepine',
            'クロチアゼパム': 'benzodiazepine',
            'クロキサゾラム': 'benzodiazepine',
            'ハロキサゾラム': 'benzodiazepine',
            'メキサゾラム': 'benzodiazepine',
            'オキサゼパム': 'benzodiazepine',
            'オキサゾラム': 'benzodiazepine',
            'プラゼパム': 'benzodiazepine',
            'トリアゾラム': 'benzodiazepine',
            'エチゾラム': 'benzodiazepine',
            'フルニトラゼパム': 'benzodiazepine',
            'ブロチゾラム': 'benzodiazepine',
            
            # バルビツール酸系
            'フェノバルビタール': 'barbiturate',
            'アモバルビタール': 'barbiturate',
            'ペントバルビタール': 'barbiturate',
            'チオペンタール': 'barbiturate',
            'セコバルビタール': 'barbiturate',
            
            # オピオイド
            'モルヒネ': 'opioid',
            'コデイン': 'opioid',
            'フェンタニル': 'opioid',
            'オキシコドン': 'opioid',
            'ヒドロコドン': 'opioid',
            'トラマドール': 'opioid',
            'ペンタゾシン': 'opioid',
            'ブプレノルフィン': 'opioid',
            'メタドン': 'opioid',
            
            # NSAIDs
            'アスピリン': 'nsaid',
            'イブプロフェン': 'nsaid',
            'ロキソプロフェン': 'nsaid',
            'ジクロフェナク': 'nsaid',
            'メフェナム酸': 'nsaid',
            'インドメタシン': 'nsaid',
            'ナプロキセン': 'nsaid',
            'ケトプロフェン': 'nsaid',
            'セレコキシブ': 'nsaid',
            'メロキシカム': 'nsaid',
            'アセトアミノフェン': 'nsaid',
            
            # スタチン
            'シンバスタチン': 'statin',
            'アトルバスタチン': 'statin',
            'プラバスタチン': 'statin',
            'ロスバスタチン': 'statin',
            'フルバスタチン': 'statin',
            'ピタバスタチン': 'statin',
            
            # ARB
            'ロサルタン': 'arb',
            'カンデサルタン': 'arb',
            'イルベサルタン': 'arb',
            'オルメサルタン': 'arb',
            'アジルサルタン': 'arb',
            
            # β遮断薬
            'プロプラノロール': 'beta_blocker',
            'アテノロール': 'beta_blocker',
            'ビソプロロール': 'beta_blocker',
            'メトプロロール': 'beta_blocker',
            'カルベジロール': 'beta_blocker',
            'ネビボロール': 'beta_blocker',
            
            # Ca拮抗薬
            'ニフェジピン': 'ca_antagonist',
            'ベラパミル': 'ca_antagonist',
            'ジルチアゼム': 'ca_antagonist',
            'ニカルジピン': 'ca_antagonist',
            'ニソルジピン': 'ca_antagonist',
            'ベニジピン': 'ca_antagonist',
            'シルニジピン': 'ca_antagonist',
            
            # 利尿薬
            'フロセミド': 'diuretic',
            'ヒドロクロロチアジド': 'diuretic',
            'スピロノラクトン': 'diuretic',
            'トリアムテレン': 'diuretic',
            'アミロライド': 'diuretic',
            'ブメタニド': 'diuretic',
            
            # 抗ヒスタミン薬
            'フェニラミン': 'antihistamine',
            'クロルフェニラミン': 'antihistamine',
            'ジフェンヒドラミン': 'antihistamine',
            'セチリジン': 'antihistamine',
            'ロラタジン': 'antihistamine',
            'フェキソフェナジン': 'antihistamine',
            
            # 制酸薬
            'アルミニウム': 'antacid',
            'マグネシウム': 'antacid',
            'カルシウム': 'antacid',
            
            # 抗凝固薬
            'ワルファリン': 'anticoagulant',
            'ダビガトラン': 'anticoagulant',
            'リバーロキサバン': 'anticoagulant',
            'アピキサバン': 'anticoagulant',
            'エドキサバン': 'anticoagulant',
            'ヘパリン': 'anticoagulant',
            'エノキサパリン': 'anticoagulant',
            
            # 糖尿病治療薬
            'メトホルミン': 'diabetes_medication',
            'インスリン': 'diabetes_medication',
            'グリクラジド': 'diabetes_medication',
            'グリメピリド': 'diabetes_medication',
            'ピオグリタゾン': 'diabetes_medication',
            'シタグリプチン': 'diabetes_medication',
            'ビルダグリプチン': 'diabetes_medication',
            'リナグリプチン': 'diabetes_medication',
            'アログリプチン': 'diabetes_medication',
            'テネリグリプチン': 'diabetes_medication',
            
            # 抗生物質
            'アモキシシリン': 'antibiotic',
            'セファレキシン': 'antibiotic',
            'エリスロマイシン': 'antibiotic',
            'クラリスロマイシン': 'antibiotic',
            'アジスロマイシン': 'antibiotic',
            'ドキシサイクリン': 'antibiotic',
            'ミノサイクリン': 'antibiotic',
            'レボフロキサシン': 'antibiotic',
            'シプロフロキサシン': 'antibiotic',
            'ノルフロキサシン': 'antibiotic',
            'バンコマイシン': 'antibiotic',
            'テイコプラニン': 'antibiotic',
            'メロペネム': 'antibiotic',
            'イミペネム': 'antibiotic',
            'セフトリアキソン': 'antibiotic',
            
            # 抗うつ薬
            'フルオキセチン': 'antidepressant',
            'パロキセチン': 'antidepressant',
            'セルトラリン': 'antidepressant',
            'エスシタロプラム': 'antidepressant',
            'ベンラファキシン': 'antidepressant',
            'デュロキセチン': 'antidepressant',
            
            # 抗精神病薬
            'アリピプラゾール': 'antipsychotic',
            'リスペリドン': 'antipsychotic',
            'オランザピン': 'antipsychotic',
            'クエチアピン': 'antipsychotic',
            
            # 気管支拡張薬
            'テオフィリン': 'bronchodilator',
            'サルブタモール': 'bronchodilator',
            'フォルモテロール': 'bronchodilator',
            'サルメテロール': 'bronchodilator',
            
            # 吸入ステロイド薬
            'ブデソニド': 'inhaled_corticosteroid',
            'フルチカゾン': 'inhaled_corticosteroid',
            
            # ロイコトリエン受容体拮抗薬
            'モンテルカスト': 'leukotriene_receptor_antagonist',
            'ザフィルルカスト': 'leukotriene_receptor_antagonist',
            
            # 去痰薬
            'アセチルシステイン': 'mucolytic',
            'カルボシステイン': 'mucolytic',
            
            # 前立腺肥大症治療薬
            'フィナステリド': 'bph_medication',
            'デュタステリド': 'bph_medication',
            'タムスロシン': 'bph_medication',
            'シルドシン': 'bph_medication',
            'ナフトピジル': 'bph_medication',
            'ウラピジル': 'bph_medication',
            
            # その他の薬剤
            'ジゴキシン': 'cardiac_glycoside',
            'アミオダロン': 'antiarrhythmic',
            'メトトレキサート': 'antirheumatic',
            'プレドニゾロン': 'corticosteroid',
            'シクロスポリン': 'immunosuppressant',
            'タクロリムス': 'immunosuppressant',
            'アザチオプリン': 'immunosuppressant',
            'ミコフェノール酸': 'immunosuppressant',
            'レフルノミド': 'antirheumatic',
            'サラゾスルファピリジン': 'antirheumatic',
            'ブシラミン': 'antirheumatic',
            'ペニシラミン': 'antirheumatic'
        }
        
        # 1. 完全一致チェック（最も信頼性が高い）
        for drug, category in exact_drug_mapping.items():
            if drug.lower() in drug_lower or drug.lower() in normalized_name:
                return category
        
        # 2. 部分一致チェック（より厳密な条件）
        best_category = 'unknown'
        best_score = 0
        
        for drug, category in exact_drug_mapping.items():
            drug_lower_match = drug.lower()
            
            # より厳密な部分一致チェック
            if (drug_lower_match in drug_lower or 
                drug_lower_match in normalized_name or
                drug_lower in drug_lower_match):
                
                # 類似度スコアを計算
                score = self._calculate_pattern_similarity(drug_lower, drug_lower_match)
                if score > best_score and score >= 0.8:  # 80%以上の類似度に引き上げ
                    best_score = score
                    best_category = category
        
        return best_category
    
    def _calculate_pattern_similarity(self, drug_name: str, pattern: str) -> float:
        """パターンとの類似度を計算"""
        if not drug_name or not pattern:
            return 0.0
        
        # 完全一致
        if drug_name == pattern:
            return 1.0
        
        # 部分一致
        if pattern in drug_name or drug_name in pattern:
            return 0.9
        
        # 文字レベルでの類似度
        common_chars = sum(1 for c in pattern if c in drug_name)
        if len(pattern) > 0:
            char_similarity = common_chars / len(pattern)
        else:
            char_similarity = 0.0
        
        # 接頭辞・接尾辞の一致
        prefix_bonus = 0.0
        suffix_bonus = 0.0
        
        if drug_name.startswith(pattern[:3]) and len(pattern) >= 3:
            prefix_bonus = 0.3
        
        if drug_name.endswith(pattern[-3:]) and len(pattern) >= 3:
            suffix_bonus = 0.3
        
        return min(char_similarity + prefix_bonus + suffix_bonus, 1.0)
    
    def _calculate_confidence(self, drug_name: str, analysis: Dict[str, Any]) -> float:
        """マッチング信頼度の計算"""
        confidence = 0.0
        
        # 長さによる基本スコア
        if len(drug_name) >= 4:
            confidence += 0.2
        
        # カテゴリ予測によるスコア
        if analysis['category'] != 'unknown':
            confidence += 0.3
        
        # 一般的な接尾辞/接頭辞によるスコア
        for suffix in self.common_suffixes:
            if drug_name.endswith(suffix):
                confidence += 0.2
                break
        
        for prefix in self.common_prefixes:
            if drug_name.startswith(prefix):
                confidence += 0.1
                break
        
        # 英語名の存在によるスコア
        if analysis['english_variants']:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _generate_english_variants(self, drug_name: str) -> List[str]:
        """英語名の変形を生成"""
        variants = []
        
        # 一般的な薬剤名の英語変換パターン
        japanese_to_english = {
            'トリアゾラム': ['triazolam', 'halcion'],
            'ジアゼパム': ['diazepam', 'valium'],
            'クロナゼパム': ['clonazepam', 'klonopin'],
            'アルプラゾラム': ['alprazolam', 'xanax'],
            'ロラゼパム': ['lorazepam', 'ativan'],
            'テマゼパム': ['temazepam', 'restoril'],
            'ミダゾラム': ['midazolam', 'versed'],
            'エスタゾラム': ['estazolam', 'prosom'],
            'フルラゼパム': ['flurazepam', 'dalmane'],
            'ニトラゼパム': ['nitrazepam', 'mogadon'],
            'ブロマゼパム': ['bromazepam', 'lexotan'],
            'クロチアゼパム': ['clotiazepam', 'rize'],
            'クロキサゾラム': ['cloxazolam', 'sepazon'],
            'エチゾラム': ['etizolam', 'sedekopan'],
            'ブロチゾラム': ['brotizolam', 'lendormin'],
            'ハロキサゾラム': ['haloxazolam', 'somelin'],
            'メキサゾラム': ['mexazolam', 'melex'],
            'オキサゾラム': ['oxazolam', 'serenal'],
            'フルタゾラム': ['flutazolam', 'coreminal'],
            'アスピリン': ['aspirin', 'acetylsalicylic acid'],
            'イブプロフェン': ['ibuprofen', 'advil', 'motrin'],
            'ロキソプロフェン': ['loxoprofen', 'loxonin'],
            'ジクロフェナク': ['diclofenac', 'voltaren'],
            'アセトアミノフェン': ['acetaminophen', 'paracetamol', 'tylenol'],
            'ワルファリン': ['warfarin', 'coumadin'],
            'ダビガトラン': ['dabigatran', 'pradaxa'],
            'リバーロキサバン': ['rivaroxaban', 'xarelto'],
            'アピキサバン': ['apixaban', 'eliquis'],
            'シンバスタチン': ['simvastatin', 'zocor'],
            'アトルバスタチン': ['atorvastatin', 'lipitor'],
            'プラバスタチン': ['pravastatin', 'pravachol'],
            'メトホルミン': ['metformin', 'glucophage'],
            'インスリン': ['insulin'],
            'プレドニゾロン': ['prednisolone', 'prednisone'],
            'アミオダロン': ['amiodarone', 'cordarone'],
            'ジゴキシン': ['digoxin', 'lanoxin'],
            'メトトレキサート': ['methotrexate', 'trexall'],
            'フェノバルビタール': ['phenobarbital', 'luminal'],
            'セコバルビタール': ['secobarbital', 'seconal'],
            'ブトクタミド': ['butoctamide', 'listomin'],
            'エソピクロン': ['eszopiclone', 'lunesta'],
            'ゾルピデム': ['zolpidem', 'ambien'],
            'ゾピクロン': ['zopiclone', 'imovane']
        }
        
        # 直接マッチング
        if drug_name in japanese_to_english:
            variants.extend(japanese_to_english[drug_name])
        
        # 部分マッチング
        for japanese, english_list in japanese_to_english.items():
            if drug_name in japanese or japanese in drug_name:
                variants.extend(english_list)
        
        # 接尾辞ベースの推測
        if drug_name.endswith('パム'):
            base = drug_name[:-2]
            variants.append(f"{base.lower()}pam")
        elif drug_name.endswith('ラム'):
            base = drug_name[:-2]
            variants.append(f"{base.lower()}lam")
        elif drug_name.endswith('ゾラム'):
            base = drug_name[:-3]
            variants.append(f"{base.lower()}zolam")
        
        return list(set(variants))  # 重複除去
    
    def _determine_search_priority(self, drug_name: str, analysis: Dict[str, Any]) -> List[str]:
        """検索優先度の決定（AI最適化版）"""
        priority = []
        
        # 信頼度が低すぎる場合は早期終了
        if analysis['confidence'] < 0.2:
            return []
        
        # 1. 元の薬剤名（高信頼度の場合のみ）
        if analysis['confidence'] >= 0.4:
            priority.append(drug_name)
        
        # 2. 正規化された名前（信頼度に応じて）
        if analysis['normalized'] != drug_name and analysis['confidence'] >= 0.3:
            priority.append(analysis['normalized'])
        
        # 3. 英語名（高信頼度のもののみ）
        english_variants = analysis['english_variants']
        if english_variants and analysis['confidence'] >= 0.5:
            # 一般的な英語名を優先
            common_names = ['aspirin', 'ibuprofen', 'warfarin', 'insulin', 'metformin']
            for name in english_variants:
                if name.lower() in common_names:
                    priority.insert(1, name)  # 高優先度で挿入
                else:
                    priority.append(name)
        
        # 4. カテゴリベースの検索（信頼度が高い場合のみ）
        if analysis['category'] != 'unknown' and analysis['confidence'] >= 0.6:
            # カテゴリ固有の検索パターンを追加
            category_patterns = {
                'benzodiazepines': ['pam', 'lam', 'zolam'],
                'barbiturates': ['barbital', 'barbital'],
                'opioids': ['morphine', 'codeine', 'fentanyl'],
                'nsaids': ['profen', 'fenac', 'aspirin'],
                'statins': ['statin'],
                'ace_inhibitors': ['pril'],
                'arb': ['sartan'],
                'beta_blockers': ['olol'],
                'calcium_blockers': ['dipine'],
                'diuretics': ['ide']
            }
            
            if analysis['category'] in category_patterns:
                for pattern in category_patterns[analysis['category']]:
                    if pattern not in priority:
                        priority.append(pattern)
        
        # 5. 一般的な接尾辞（信頼度が高い場合のみ）
        if analysis['confidence'] >= 0.7:
            for suffix in self.common_suffixes:
                if drug_name.endswith(suffix) and suffix not in priority:
                    priority.append(suffix)
        
        # 信頼度に基づいて検索パターン数を制限
        max_patterns = min(3, int(analysis['confidence'] * 5))  # 信頼度に応じて1-3個
        return priority[:max_patterns]

class DrugService:
    def __init__(self):
        self.drug_database = None
        self.therapeutic_categories = {}
        self.same_effect_drugs = {}
        self.interaction_rules = {}
        self.diagnosis_templates = {}
        self.category_mapping = {}
        self.local_db_cache = {}
        self.kegg_cache = {}
        self.normalization_cache = {}
        self.kegg_api_base = "https://rest.kegg.jp"
        self.ai_matcher = AIDrugMatcher()  # AIマッチャーを追加
        self.last_api_call = 0  # API呼び出しの時間制限用
        self.api_call_interval = 0.5  # 0.5秒間隔
        self.diagnosis_cache = {}
        
        # データベースとルールの読み込み
        self._load_drug_database()
        self._load_therapeutic_categories()
        self._load_same_effect_drugs()
        self._load_interaction_rules()
        self._create_diagnosis_templates()
        self._create_category_mapping()
        
        # 初期化完了のログ
        logger.info(f"DrugService initialized successfully - Database: {len(self.drug_database) if self.drug_database is not None else 0} records")

    def _rate_limit_api_call(self):
        """API呼び出しのレート制限"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.api_call_interval:
            sleep_time = self.api_call_interval - time_since_last_call
            time.sleep(sleep_time)
        self.last_api_call = time.time()

    def _load_drug_database(self):
        """薬剤データベースを読み込み"""
        try:
            # データベースファイルの存在確認
            import os
            db_paths = [
                "data/processed_drug_database.csv",
                "data/drug_database.csv",
                "data/enhanced_drug_database.csv"
            ]
            
            for db_path in db_paths:
                if os.path.exists(db_path):
                    self.drug_database = pd.read_csv(db_path)
                    logger.info(f"Loaded drug database from {db_path}: {len(self.drug_database)} records")
                    return
            
            # どのファイルも見つからない場合
            logger.warning("No drug database file found, creating sample database")
            self._create_sample_database()
            
        except Exception as e:
            logger.error(f"Error loading drug database: {e}")
            logger.info("Creating sample database as fallback")
            self._create_sample_database()
    
    def _load_therapeutic_categories(self):
        """治療分類による同効薬グループを定義"""
        self.therapeutic_categories = {
            '解熱鎮痛薬': {
                'NSAIDs': ['アスピリン', 'イブプロフェン', 'ロキソプロフェン', 'ジクロフェナク', 'メフェナム酸'],
                'アセトアミノフェン系': ['アセトアミノフェン', 'カロナール'],
                'オピオイド系': ['コデイン', 'モルヒネ', 'フェンタニル']
            },
            '抗凝固薬': {
                'ワルファリン系': ['ワルファリン', 'ワーファリン'],
                'DOAC': ['ダビガトラン', 'リバーロキサバン', 'アピキサバン', 'エドキサバン'],
                'ヘパリン系': ['ヘパリン', 'ダルテパリン', 'エノキサパリン']
            },
            '脂質異常症治療薬': {
                'スタチン系': ['シンバスタチン', 'アトルバスタチン', 'プラバスタチン', 'ロスバスタチン', 'ピタバスタチン'],
                'フィブラート系': ['ベザフィブラート', 'フェノフィブラート'],
                'コレステロール吸収阻害薬': ['エゼチミブ']
            },
            '糖尿病治療薬': {
                'ビグアナイド系': ['メトホルミン', 'ブホルミン'],
                'スルホニルウレア系': ['グリメピリド', 'グリクラジド', 'トルブタミド'],
                'DPP-4阻害薬': ['シタグリプチン', 'ビルダグリプチン', 'リナグリプチン', 'アログリプチン'],
                'SGLT2阻害薬': ['ダパグリフロジン', 'カナグリフロジン', 'エンパグリフロジン']
            },
            '高血圧治療薬': {
                'カルシウム拮抗薬': ['ニフェジピン', 'アムロジピン', 'ベラパミル', 'ジルチアゼム'],
                'ACE阻害薬': ['カプトプリル', 'エナラプリル', 'リシノプリル'],
                'ARB': ['ロサルタン', 'カンデサルタン', 'バルサルタン', 'オルメサルタン'],
                'β遮断薬': ['プロプラノロール', 'アテノロール', 'ビソプロロール']
            }
        }
        logger.info(f"Loaded therapeutic categories: {len(self.therapeutic_categories)} categories")
    
    def _load_same_effect_drugs(self):
        """同効薬の詳細マッピング"""
        self.same_effect_drugs = {
            'アスピリン': {
                'same_effect': ['イブプロフェン', 'ロキソプロフェン', 'ジクロフェナク'],
                'mechanism': 'COX-1/COX-2阻害',
                'risk_level': 'high'
            },
            'ワルファリン': {
                'same_effect': ['ダビガトラン', 'リバーロキサバン', 'アピキサバン'],
                'mechanism': '抗凝固作用',
                'risk_level': 'critical'
            },
            'シンバスタチン': {
                'same_effect': ['アトルバスタチン', 'プラバスタチン', 'ロスバスタチン'],
                'mechanism': 'HMG-CoA還元酵素阻害',
                'risk_level': 'high'
            },
            'メトホルミン': {
                'same_effect': ['ブホルミン'],
                'mechanism': 'ビグアナイド系',
                'risk_level': 'medium'
            }
        }
        logger.info(f"Loaded same effect drugs: {len(self.same_effect_drugs)} mappings")
    
    def _create_sample_database(self):
        """サンプル薬剤データベースを作成"""
        sample_data = {
            'drug_name': [
                'アスピリン', 'ワルファリン', 'ジゴキシン', 'アミオダロン',
                'シンバスタチン', 'アトルバスタチン', 'メトトレキサート',
                'プレドニゾロン', 'インスリン', 'メトホルミン', 'イブプロフェン',
                'ロキソプロフェン', 'ダビガトラン', 'リバーロキサバン', 'プラバスタチン'
            ],
            'generic_name': [
                'アセチルサリチル酸', 'ワルファリンカリウム', 'ジゴキシン',
                'アミオダロン塩酸塩', 'シンバスタチン', 'アトルバスタチンカルシウム',
                'メトトレキサート', 'プレドニゾロン', 'インスリン', 'メトホルミン塩酸塩',
                'イブプロフェン', 'ロキソプロフェンナトリウム', 'ダビガトランエテキシラート',
                'リバーロキサバン', 'プラバスタチンナトリウム'
            ],
            'category': [
                '解熱鎮痛薬', '抗凝固薬', '強心薬', '抗不整脈薬',
                '脂質異常症治療薬', '脂質異常症治療薬', '抗リウマチ薬',
                '副腎皮質ホルモン', '糖尿病治療薬', '糖尿病治療薬', '解熱鎮痛薬',
                '解熱鎮痛薬', '抗凝固薬', '抗凝固薬', '脂質異常症治療薬'
            ],
            'interactions': [
                'ワルファリン,抗凝固薬', 'アスピリン,NSAIDs', 'アミオダロン,ジゴキシン',
                'ジゴキシン,アミオダロン', 'アミオダロン,シンバスタチン', 'シンバスタチン,アミオダロン',
                'メトトレキサート,葉酸', 'プレドニゾロン,インスリン', 'インスリン,メトホルミン',
                'メトホルミン,インスリン', 'アスピリン,イブプロフェン', 'アスピリン,ロキソプロフェン',
                'ワルファリン,ダビガトラン', 'ワルファリン,リバーロキサバン', 'シンバスタチン,プラバスタチン'
            ]
        }
        self.drug_database = pd.DataFrame(sample_data)
        logger.info(f"Created sample drug database: {len(self.drug_database)} records")
    
    def _load_interaction_rules(self):
        """飲み合わせルールを読み込む（詳細版）"""
        self.interaction_rules = {
            'ワルファリン': {
                'アスピリン': {'risk': 'high', 'description': '出血リスク増加', 'mechanism': '血小板機能阻害の重複'},
                'NSAIDs': {'risk': 'high', 'description': '出血リスク増加', 'mechanism': '抗凝固作用の増強'},
                'ジゴキシン': {'risk': 'medium', 'description': '薬物相互作用の可能性', 'mechanism': 'タンパク結合競合'},
                'ダビガトラン': {'risk': 'critical', 'description': '重複投与による出血リスク', 'mechanism': '抗凝固作用の重複'}
            },
            'アミオダロン': {
                'ジゴキシン': {'risk': 'high', 'description': 'ジゴキシン血中濃度上昇', 'mechanism': 'CYP3A4阻害'},
                'シンバスタチン': {'risk': 'high', 'description': '横紋筋融解症リスク増加', 'mechanism': 'CYP3A4阻害'},
                'ワルファリン': {'risk': 'high', 'description': 'ワルファリン効果増強', 'mechanism': 'CYP2C9阻害'}
            },
            'シンバスタチン': {
                'アミオダロン': {'risk': 'high', 'description': '横紋筋融解症リスク増加', 'mechanism': 'CYP3A4阻害'},
                'アトルバスタチン': {'risk': 'medium', 'description': '横紋筋融解症リスク増加', 'mechanism': '同系統薬剤の重複'},
                'プラバスタチン': {'risk': 'medium', 'description': '横紋筋融解症リスク増加', 'mechanism': '同系統薬剤の重複'}
            },
            'メトトレキサート': {
                '葉酸': {'risk': 'medium', 'description': '葉酸補充が必要', 'mechanism': '葉酸拮抗作用'},
                'NSAIDs': {'risk': 'high', 'description': '骨髄抑制リスク増加', 'mechanism': '腎排泄競合'}
            }
        }
        logger.info(f"Loaded interaction rules: {len(self.interaction_rules)} drug mappings")
    
    def _deduplicate_drugs(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """薬剤リストの重複除去（共通関数）"""
        unique_drugs = []
        seen_names = set()
        
        for drug in drugs:
            drug_name = drug.get('name', '').strip()
            if drug_name and drug_name not in seen_names:
                seen_names.add(drug_name)
                unique_drugs.append(drug)
            elif drug_name:
                logger.info(f"Duplicate drug removed: {drug_name}")
        
        logger.info(f"After deduplication: {len(unique_drugs)} unique drugs found")
        return unique_drugs

    def _create_drug_info_from_row(self, row: pd.Series, drug_name: str) -> Dict[str, Any]:
        """行データから薬剤情報を作成（共通関数）"""
        drug_name_str = str(row['drug_name']).strip()
        generic_name_str = str(row['generic_name']).strip()
        
        interactions_value = row['interactions']
        try:
            if pd.isna(interactions_value):
                interactions_str = ''
            else:
                interactions_str = str(interactions_value)
        except:
            interactions_str = ''
        
        return {
            'name': drug_name_str,
            'generic_name': generic_name_str,
            'category': str(row['category']),
            'interactions': interactions_str.split(',') if interactions_str else []
        }

    def _find_drug_info(self, drug_name: str) -> Dict[str, Any]:
        # --- ローカルDBキャッシュ ---
        if drug_name in self.local_db_cache:
            return self.local_db_cache[drug_name]
        
        # AIマッチャーで薬剤名を分析
        analysis = self.ai_matcher.analyze_drug_name(drug_name)
        
        # データベースから検索（効率化：1回のループで完全一致と部分一致を同時チェック）
        exact_match = None
        partial_match = None
        
        for _, row in self.drug_database.iterrows():
            drug_name_str = str(row['drug_name']).strip()
            generic_name_str = str(row['generic_name']).strip()
            
            # 完全一致チェック
            if drug_name == drug_name_str or drug_name == generic_name_str:
                exact_match = self._create_drug_info_from_row(row, drug_name)
                break
            
            # 部分一致チェック（完全一致が見つからない場合のみ）
            if not exact_match and len(drug_name) >= 3:
                if (drug_name in drug_name_str) or (drug_name in generic_name_str):
                    partial_match = self._create_drug_info_from_row(row, drug_name)
        
        # 結果を決定
        if exact_match or partial_match:
            result = exact_match or partial_match
            if result is None:  # 型安全性のため
                result = {
            'name': drug_name,
            'generic_name': '不明',
            'category': '不明',
                    'interactions': []
                }
        else:
            # データベースにマッチしない場合、AI分析結果を使用
            ai_category = analysis.get('category', 'unknown') if analysis else 'unknown'
            result = {
                'name': drug_name,
                'generic_name': '不明',
                'category': self._map_ai_category_to_japanese(ai_category),
            'interactions': []
        }
        
        # AI分析結果を追加
        if analysis:
            result['ai_analysis'] = analysis
        
        self.local_db_cache[drug_name] = result
        return result
    
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
            'unknown': '不明'
        }
        return category_mapping.get(ai_category, ai_category)

    def _find_partial_match(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """部分一致による薬剤検索"""
        if not self.drug_database is not None:
            return None
        
        best_match = None
        best_score = 0
        
        for _, row in self.drug_database.iterrows():
            drug_name_str = str(row['drug_name']).strip()
            generic_name_str = str(row['generic_name']).strip()
            
            # 類似度スコアを計算
            score1 = fuzz.ratio(drug_name, drug_name_str)
            score2 = fuzz.ratio(drug_name, generic_name_str)
            max_score = max(score1, score2)
            
            if max_score > best_score and max_score >= 60:  # 60%以上の類似度
                best_score = max_score
                best_match = self._create_drug_info_from_row(row, drug_name)
        
        return best_match

    def _check_interactions(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """薬剤間の相互作用をチェック（itertools.combinationsで最適化）"""
        interactions = []
        
        # 全組み合わせを効率的に生成
        for drug1, drug2 in itertools.combinations(drugs, 2):
            interaction = self._check_interaction_rule(drug1['name'], drug2['name'])
            if interaction:
                interactions.append({
                    'drug1': drug1['name'],
                    'drug2': drug2['name'],
                    'risk': interaction['risk'],
                    'description': interaction['description']
                })
        
        return interactions

    def _check_same_effect_drugs(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """同効薬の重複チェック（itertools.combinationsで最適化）"""
        warnings = []
        
        # 全組み合わせを効率的に生成
        for drug1, drug2 in itertools.combinations(drugs, 2):
            same_effect_info = self._check_same_effect(drug1['name'], drug2['name'])
            if same_effect_info:
                warnings.append({
                    'drug1': drug1['name'],
                    'drug2': drug2['name'],
                    'mechanism': same_effect_info['mechanism'],
                    'risk_level': same_effect_info['risk_level'],
                    'description': f"同効薬の重複投与: {same_effect_info['mechanism']}"
                })
        
        return warnings

    def _check_category_duplicates(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """薬剤分類による重複チェック（defaultdictで最適化）"""
        duplicates = []
        category_groups = defaultdict(list)
        
        # 薬剤を分類別にグループ化（defaultdictで簡潔化）
        for drug in drugs:
            category_groups[drug['category']].append(drug)
        
        # 同一分類内で複数の薬剤がある場合をチェック
        for category, drug_list in category_groups.items():
            if len(drug_list) > 1:
                duplicates.append({
                    'category': category,
                    'drugs': [drug['name'] for drug in drug_list],
                    'count': len(drug_list),
                    'description': f"{category}の薬剤が{len(drug_list)}種類検出されました"
                })
        
        return duplicates

    def normalize_name(self, name: str) -> str:
        """薬剤名の正規化（キャッシュ付き）"""
        if not name:
            return ''
        
        # キャッシュチェック
        if name in self.normalization_cache:
            return self.normalization_cache[name]
        
        # 正規化処理
        normalized = name
        # 全角→半角
        normalized = re.sub(r'[Ａ-Ｚａ-ｚ０-９]', lambda x: chr(ord(x.group(0)) - 0xFEE0), normalized)
        # ひらがな→カタカナ変換（薬剤名はカタカナ表記が標準）
        hiragana_to_katakana = str.maketrans({
            'あ': 'ア', 'い': 'イ', 'う': 'ウ', 'え': 'エ', 'お': 'オ',
            'か': 'カ', 'き': 'キ', 'く': 'ク', 'け': 'ケ', 'こ': 'コ',
            'さ': 'サ', 'し': 'シ', 'す': 'ス', 'せ': 'セ', 'そ': 'ソ',
            'た': 'タ', 'ち': 'チ', 'つ': 'ツ', 'て': 'テ', 'と': 'ト',
            'な': 'ナ', 'に': 'ニ', 'ぬ': 'ヌ', 'ね': 'ネ', 'の': 'ノ',
            'は': 'ハ', 'ひ': 'ヒ', 'ふ': 'フ', 'へ': 'ヘ', 'ほ': 'ホ',
            'ま': 'マ', 'み': 'ミ', 'む': 'ム', 'め': 'メ', 'も': 'モ',
            'や': 'ヤ', 'ゆ': 'ユ', 'よ': 'ヨ',
            'ら': 'ラ', 'り': 'リ', 'る': 'ル', 'れ': 'レ', 'ろ': 'ロ',
            'わ': 'ワ', 'を': 'ヲ', 'ん': 'ン',
            'が': 'ガ', 'ぎ': 'ギ', 'ぐ': 'グ', 'げ': 'ゲ', 'ご': 'ゴ',
            'ざ': 'ザ', 'じ': 'ジ', 'ず': 'ズ', 'ぜ': 'ゼ', 'ぞ': 'ゾ',
            'だ': 'ダ', 'ぢ': 'ヂ', 'づ': 'ヅ', 'で': 'デ', 'ど': 'ド',
            'ば': 'バ', 'び': 'ビ', 'ぶ': 'ブ', 'べ': 'ベ', 'ぼ': 'ボ',
            'ぱ': 'パ', 'ぴ': 'ピ', 'ぷ': 'プ', 'ぺ': 'ペ', 'ぽ': 'ポ',
            'ゃ': 'ャ', 'ゅ': 'ュ', 'ょ': 'ョ', 'っ': 'ッ'
        })
        normalized = normalized.translate(hiragana_to_katakana)
        # 記号・空白除去
        normalized = re.sub(r'[\s\-・,，、.。()（）【】［］\[\]{}｛｝<>《》"\'\-―ー=+*/\\]', '', normalized)
        
        # キャッシュに保存
        self.normalization_cache[name] = normalized
        return normalized

    def _generate_search_patterns_optimized(self, drug_name: str, english_names: List[str] = None) -> set:
        """AIベースの検索パターン生成（非推奨 - AIマッチャーを使用）"""
        # このメソッドは非推奨です。代わりにAIマッチャーを使用してください
        analysis = self.ai_matcher.analyze_drug_name(drug_name)
        return set(analysis['search_priority'])

    def match_to_database(self, ocr_names: list) -> list:
        """OCRで検出された薬剤名をデータベースにマッチング（改善版：薬剤の漏れを防ぐ）"""
        if not ocr_names:
            return []
            
        # AIベースのバッチ処理最適化（制限を緩和）
        optimized_names = self._ai_batch_optimize(ocr_names)
        
        results = []
        for drug_name in optimized_names:
            # ローカルDBキャッシュチェック
            if drug_name in self.local_db_cache:
                cached_result = self.local_db_cache[drug_name]
                if cached_result is not None:
                    results.append(drug_name)  # 薬剤名のみを追加
                    continue
            
            # 薬剤情報を検索
            drug_info = self._find_drug_info(drug_name)
            
            # キャッシュに保存
            self.local_db_cache[drug_name] = drug_info
            
            # より寛容な条件で薬剤を追加
            if drug_info is not None:
                results.append(drug_name)  # 薬剤名のみを追加
            else:
                # データベースにない場合でも、信頼度が高い場合は追加
                analysis = self.ai_matcher.analyze_drug_name(drug_name)
                if analysis['confidence'] >= 0.2:  # 信頼度の閾値を下げる
                    results.append(drug_name)
                    logger.info(f"Added drug '{drug_name}' with confidence {analysis['confidence']}")
        
        # 結果が空の場合は元のリストをそのまま返す
        if not results:
            results = ocr_names
            logger.warning(f"No drugs matched to database, returning original OCR names: {ocr_names}")
        
        logger.info(f"Matched {len(results)} drugs from {len(ocr_names)} OCR names")
        return results
    
    def _ai_batch_optimize(self, drug_names: List[str]) -> List[str]:
        """AIベースのバッチ処理最適化（改善版：薬剤の漏れを防ぐ）"""
        optimized_names = []
        
        # 1. 重複除去
        unique_names = list(set(drug_names))
        
        # 2. 信頼度による優先度付け
        name_priorities = []
        for name in unique_names:
            analysis = self.ai_matcher.analyze_drug_name(name)
            priority_score = analysis['confidence']
            name_priorities.append((name, priority_score))
        
        # 3. 信頼度の高い順にソート
        name_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 制限を緩和してより多くの薬剤を含める
        # 高信頼度のもの（信頼度0.3以上）をすべて含める
        high_confidence_names = [name for name, score in name_priorities if score >= 0.3]
        
        # 低信頼度のものも含める（最大10個まで）
        low_confidence_names = [name for name, score in name_priorities if score < 0.3][:10]
        
        optimized_names = high_confidence_names + low_confidence_names
        
        # 5. 結果が空の場合は元のリストをそのまま返す
        if not optimized_names:
            optimized_names = unique_names
        
        logger.info(f"AI batch optimization: {len(drug_names)} -> {len(optimized_names)} names")
        return optimized_names

    def _try_kegg_matching(self, drug_name: str) -> Optional[str]:
        """AIベースの効率的なKEGG API検索（無効化版）"""
        # KEGG検索を無効化してタイムアウトを回避
        logger.info(f"KEGG search disabled for '{drug_name}' to prevent timeout")
        return None

    def get_drug_interactions(self, drug_names: List[str]) -> Dict[str, Any]:
        """薬剤名リストから飲み合わせ情報を取得（AI強化版）"""
        try:
            logger.info(f"Starting drug interaction analysis for {len(drug_names)} drugs: {drug_names}")
            
            # AI診断の実行
            ai_analysis = self.analyze_drug_interactions_ai(drug_names)
            logger.info(f"AI analysis completed successfully")
            
            results = {
                'detected_drugs': [],
                'interactions': [],
                'same_effect_warnings': [],
                'category_duplicates': [],
                'kegg_info': [],
                'warnings': [],
                'recommendations': [],
                'diagnosis_details': [],
                'ai_analysis': ai_analysis,  # AI分析結果を追加
                'clinical_risks': ai_analysis['risk_summary'],  # 臨床リスク情報
                'drug_categories': ai_analysis['drug_categories']  # 薬剤カテゴリ情報
            }
            
            # 検出された薬剤の情報を取得
            detected_drugs = []
            for drug_name in drug_names:
                drug_info = self._find_drug_info(drug_name)
                if drug_info is not None:
                    # AI分析で得られたカテゴリ情報を追加
                    category = ai_analysis['drug_categories'].get(drug_name, 'unknown')
                    drug_info['ai_category'] = category
                    detected_drugs.append(drug_info)
                    logger.info(f"Found drug info for {drug_name}: {category}")
                else:
                    logger.warning(f"No drug info found for {drug_name}")
            
            # 重複除去
            results['detected_drugs'] = self._deduplicate_drugs(detected_drugs)
            logger.info(f"Deduplicated drugs: {len(results['detected_drugs'])}")
            
            # 従来の相互作用チェック（補完的）
            if len(results['detected_drugs']) > 1:
                interactions = self._check_interactions(results['detected_drugs'])
                results['interactions'] = interactions
                logger.info(f"Found {len(interactions)} interactions")
            
            # 同効薬チェック
            same_effect_warnings = self._check_same_effect_drugs(results['detected_drugs'])
            results['same_effect_warnings'] = same_effect_warnings
            logger.info(f"Found {len(same_effect_warnings)} same effect warnings")
            
            # 薬剤分類による重複チェック
            results['category_duplicates'] = self._check_category_duplicates(results['detected_drugs'])
            logger.info(f"Found {len(results['category_duplicates'])} category duplicates")
            
            # KEGG情報の取得
            results['kegg_info'] = self._get_kegg_info(results['detected_drugs'])
            logger.info(f"Found {len(results['kegg_info'])} KEGG info entries")
            
            # AI分析に基づく警告と推奨事項を生成
            results['warnings'] = self._generate_ai_warnings(ai_analysis)
            results['recommendations'] = ai_analysis['risk_summary']['clinical_recommendations']
            logger.info(f"Generated {len(results['warnings'])} warnings and {len(results['recommendations'])} recommendations")
            
            # 診断詳細の生成
            results['diagnosis_details'] = self._generate_ai_diagnosis_details(ai_analysis)
            logger.info(f"Generated {len(results['diagnosis_details'])} diagnosis details")
            
            # 診断結果をキャッシュ
            norm_names = [self.normalize_name(n) for n in drug_names]
            cache_key = tuple(sorted(norm_names))
            self.diagnosis_cache[cache_key] = results
            
            logger.info(f"Drug interaction analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error getting drug interactions: {e}", exc_info=True)
            # より詳細なエラー情報を提供
            return {
                'detected_drugs': [],
                'interactions': [],
                'same_effect_warnings': [],
                'category_duplicates': [],
                'kegg_info': [],
                'warnings': [f'薬剤情報の取得中にエラーが発生しました: {str(e)}'],
                'recommendations': ['薬剤師にご相談ください', 'システムエラーが発生しました'],
                'ai_analysis': {},
                'clinical_risks': {},
                'drug_categories': {},
                'diagnosis_details': []
            }
    
    def _generate_ai_warnings(self, ai_analysis: Dict) -> List[str]:
        """AI分析に基づく警告の生成"""
        warnings = []
        
        # 高リスクの警告
        if ai_analysis['high_risk_count'] > 0:
            warnings.append(f"🚨 高リスク相互作用が{ai_analysis['high_risk_count']}件検出されました")
        
        # 中リスクの警告
        if ai_analysis['medium_risk_count'] > 0:
            warnings.append(f"⚠️ 中リスク相互作用が{ai_analysis['medium_risk_count']}件検出されました")
        
        # 低リスクの警告
        if ai_analysis['low_risk_count'] > 0:
            warnings.append(f"ℹ️ 低リスク相互作用が{ai_analysis['low_risk_count']}件検出されました")
        
        # 薬剤カテゴリの警告
        drug_categories = ai_analysis['drug_categories']
        
        # 血圧降下薬の多剤併用
        bp_meds = [drug for drug, cat in drug_categories.items() 
                  if cat in ['ace_inhibitor', 'arb', 'ca_antagonist', 'beta_blocker', 'diuretic', 'nitrate', 'arni']]
        if len(bp_meds) >= 3:
            warnings.append(f"💊 血圧降下薬が{len(bp_meds)}剤検出されました（低血圧リスク）")
        
        # 胃薬の重複
        gastric_meds = [drug for drug, cat in drug_categories.items() 
                       if cat in ['ppi', 'p_cab']]
        if len(gastric_meds) >= 2:
            warnings.append(f"🫀 胃酸分泌抑制薬が{len(gastric_meds)}剤検出されました（重複投与リスク）")
        
        # 向精神薬の多剤併用
        psychotropics = [drug for drug, cat in drug_categories.items() 
                        if cat in ['benzodiazepine', 'barbiturate', 'antidepressant', 'antipsychotic']]
        if len(psychotropics) >= 2:
            warnings.append(f"🧠 向精神薬が{len(psychotropics)}剤検出されました（中枢抑制作用増強リスク）")
        
        return warnings
    
    def _generate_ai_diagnosis_details(self, ai_analysis: Dict) -> List[Dict]:
        """AI分析に基づく診断詳細の生成"""
        diagnosis_details = []
        
        # 検出されたリスクの詳細
        for risk in ai_analysis['detected_risks']:
            diagnosis_details.append({
                'type': 'AI検出リスク',
                'risk_level': risk['risk_level'],
                'drugs': risk['involved_drugs'],
                'category': risk['risk_name'],
                'reason': risk['description'],
                'symptoms': risk['clinical_impact'],
                'recommendation': risk['recommendation']
            })
        
        # 薬剤カテゴリ別の詳細
        drug_categories = ai_analysis['drug_categories']
        category_groups = {}
        
        for drug, category in drug_categories.items():
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(drug)
        
        # 同一カテゴリの薬剤が複数ある場合の警告
        for category, drugs in category_groups.items():
            if len(drugs) >= 2:
                diagnosis_details.append({
                    'type': '同一カテゴリ重複',
                    'risk_level': 'medium',
                    'drugs': drugs,
                    'category': category,
                    'reason': f"{category}カテゴリの薬剤が{len(drugs)}剤検出されました",
                    'symptoms': '同効薬の重複投与による副作用リスクの増加',
                    'recommendation': '重複投与を避け、必要に応じて剤形を変更してください'
                })
        
        return diagnosis_details

    def _check_interaction_rule(self, drug1: str, drug2: str) -> Optional[Dict[str, str]]:
        """2つの薬剤間の相互作用ルールをチェック"""
        # 相互作用ルールをチェック
        for drug, interactions in self.interaction_rules.items():
            if drug1 == drug and drug2 in interactions:
                interaction_info = interactions[drug2]
                if isinstance(interaction_info, dict):
                    return {
                        'risk': interaction_info.get('risk', 'high'),
                        'description': interaction_info.get('description', '相互作用あり'),
                        'mechanism': interaction_info.get('mechanism', '不明')
                    }
                else:
                    return {
                        'risk': 'high',
                        'description': str(interaction_info),
                        'mechanism': '不明'
                    }
            elif drug2 == drug and drug1 in interactions:
                interaction_info = interactions[drug1]
                if isinstance(interaction_info, dict):
                    return {
                        'risk': interaction_info.get('risk', 'high'),
                        'description': interaction_info.get('description', '相互作用あり'),
                        'mechanism': interaction_info.get('mechanism', '不明')
                    }
                else:
                    return {
                        'risk': 'high',
                        'description': str(interaction_info),
                        'mechanism': '不明'
                    }
        
        return None
    

    
    def _check_same_effect(self, drug1: str, drug2: str) -> Optional[Dict[str, str]]:
        """2つの薬剤が同効薬かチェック"""
        for drug, info in self.same_effect_drugs.items():
            if drug1 == drug and drug2 in info['same_effect']:
                return info
            elif drug2 == drug and drug1 in info['same_effect']:
                return info
        return None
    
    def _get_kegg_info(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """KEGGデータベースから薬剤情報を取得（無効化版）"""
        # KEGG情報取得を無効化してタイムアウトを回避
        logger.info(f"KEGG info retrieval disabled for {len(drugs)} drugs to prevent timeout")
        return []
    
    def _convert_to_english_names(self, drug_name: str) -> List[str]:
        """日本語薬剤名を英語名に変換"""
        try:
            # 既存の英語名変換辞書を使用
            english_names = self._get_english_drug_names(drug_name)
            logger.info(f"English names for '{drug_name}': {english_names}")
            return english_names
        except Exception as e:
            logger.warning(f"Failed to convert '{drug_name}' to English names: {e}")
            return []

    def _search_kegg_api(self, drug_name, english_names=None):
        """KEGG API検索（非推奨 - AIベースのシステムを使用してください）"""
        logger.warning(f"DEPRECATED: _search_kegg_api called for '{drug_name}'. Use AI-based system instead.")
        # AIベースのシステムに委譲
        return self._try_kegg_matching(drug_name)

    def _calculate_similarity_score(self, drug_name, description, english_names=None):
        """類似度計算の改善（部分一致強化）"""
        # 基本スコア
        exact_score = self._calculate_exact_match_score(drug_name, description)
        partial_score = self._calculate_partial_match_score(drug_name, description)
        token_score = self._calculate_token_match_score(drug_name, description)
        
        # 英語名ボーナス
        english_bonus = self._calculate_english_bonus(drug_name, description, english_names)
        
        # 部分一致ボーナス（強化）
        partial_bonus = self._calculate_partial_bonus(drug_name, description)
        
        # 語尾パターンボーナス
        suffix_bonus = self._calculate_suffix_bonus(drug_name, description)
        
        # 重み付け計算
        total_score = (
            exact_score * 0.4 +
            partial_score * 0.3 +
            token_score * 0.2 +
            english_bonus * 0.05 +
            partial_bonus * 0.03 +
            suffix_bonus * 0.02
        )
        
        return total_score

    def _calculate_suffix_bonus(self, drug_name, description):
        """語尾パターンボーナス"""
        suffixes = ['パム', 'ラム', 'ロン', 'ジアゼパム', 'ピオン', 'ペート', 'ジン', 'ゾラム', 'バミル', 'バミド', 'バミン']
        
        for suffix in suffixes:
            if drug_name.endswith(suffix) and suffix in description:
                return 15  # 語尾一致ボーナス
        
        return 0

    def _generate_search_patterns(self, drug_name, english_names=None):
        """検索パターン生成（非推奨 - AIベースのシステムを使用してください）"""
        logger.warning(f"DEPRECATED: _generate_search_patterns called for '{drug_name}'. Use AI-based system instead.")
        # AIベースのシステムに委譲
        analysis = self.ai_matcher.analyze_drug_name(drug_name)
        return analysis['search_priority'][:3]  # 最大3パターンのみ

    def _calculate_exact_match_score(self, drug_name, description):
        """正確なマッチスコア"""
        return fuzz.ratio(drug_name, description)

    def _calculate_partial_match_score(self, drug_name, description):
        """部分マッチスコア"""
        return fuzz.partial_ratio(drug_name, description)

    def _calculate_token_match_score(self, drug_name, description):
        """トークンマッチスコア"""
        return fuzz.token_sort_ratio(drug_name, description)

    def _calculate_english_bonus(self, drug_name, description, english_names):
        """英語名ボーナススコア"""
        bonus = 0
        for english_name in english_names:
            if english_name.lower() in description.lower():
                bonus = 15
                break
        return bonus

    def _calculate_partial_bonus(self, drug_name, description):
        """部分マッチボーナススコア"""
        # 部分一致のボーナス計算
        if drug_name in description or description in drug_name:
            return 10
        return 0

    def _extract_kegg_category(self, kegg_text: str) -> str:
        """KEGGテキストから薬効分類（CLASS/CATEGORY/ATC）を抽出"""
        lines = kegg_text.split('\n')
        category = None
        for line in lines:
            if line.startswith('CLASS') and len(line.split()) > 1:
                category = line.split(' ', 1)[1].strip()
                break
        if not category:
            for line in lines:
                if line.startswith('CATEGORY') and len(line.split()) > 1:
                    category = line.split(' ', 1)[1].strip()
                    break
        if not category:
            for line in lines:
                if line.startswith('ATC') and len(line.split()) > 1:
                    category = line.split(' ', 1)[1].strip()
                    break
        return category if category else 'Others'

    def _fetch_kegg_drug_info(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """KEGG APIから薬剤情報を取得（改善版）"""
        try:
            # 複数の検索パターンを試行
            search_patterns = [
                drug_name,
                self.normalize_name(drug_name),
                self._remove_dosage_form(drug_name),
                self._remove_dosage_form(self.normalize_name(drug_name))
            ]
            
            for pattern in search_patterns:
                if not pattern or len(pattern) < 2:
                    continue
                    
                # KEGG APIで検索
                search_url = f"{self.kegg_api_base}/find/drug/{pattern}"
                response = requests.get(search_url, timeout=5)
                
                if response.status_code == 200 and response.text.strip():
                    lines = response.text.strip().split('\n')
                    if lines and lines[0]:
                        # 最適なマッチを選択
                        best_match = self._select_best_kegg_match(lines, drug_name)
                        if best_match:
                            kegg_id = best_match['kegg_id']
                            
                            # 薬剤の詳細情報を取得
                            info_url = f"{self.kegg_api_base}/get/{kegg_id}"
                            info_response = requests.get(info_url, timeout=5)
                            
                            if info_response.status_code == 200:
                                kegg_text = info_response.text
                                return {
                                    'kegg_id': kegg_id,
                                    'category': self._extract_kegg_category(kegg_text),
                                    'pathways': self._extract_pathways(kegg_text),
                                    'targets': self._extract_targets(kegg_text),
                                    'matched_name': best_match['kegg_name']
                                }
            
            return None
            
        except requests.exceptions.Timeout:
            logger.warning(f"KEGG API タイムアウト: {drug_name}")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"KEGG API 接続エラー: {drug_name}")
            return None
        except Exception as e:
            logger.warning(f"KEGG API エラー ({drug_name}): {e}")
            return None
    
    def _select_best_kegg_match(self, kegg_lines: List[str], original_name: str) -> Optional[Dict[str, str]]:
        """KEGG検索結果から最適なマッチを選択"""
        try:
            candidates = []
            original_lower = original_name.lower()
            original_norm = self.normalize_name(original_name)
            
            # 英語名変換を取得
            english_names = self._get_english_drug_names(original_name)
            
            # 医薬品以外の成分を除外するキーワード
            exclude_keywords = {
                'earth', 'wax', 'oil', 'acid', 'silica', 'silicon', 'aluminum', 'magnesium',
                'calcium', 'sodium', 'potassium', 'chloride', 'sulfate', 'carbonate',
                'phosphate', 'citrate', 'stearate', 'palmitate', 'oleate', 'laurate',
                'glycerin', 'propylene', 'polyethylene', 'cellulose', 'starch', 'lactose',
                'sucrose', 'sorbitol', 'mannitol', 'xylitol', 'maltitol', 'erythritol',
                'vanillin', 'menthol', 'camphor', 'eucalyptus', 'peppermint', 'anise',
                'caraway', 'cardamom', 'cinnamon', 'clove', 'ginger', 'lemon', 'lime',
                'orange', 'rose', 'lavender', 'jasmine', 'ylang', 'patchouli', 'sandalwood',
                'cedar', 'pine', 'fir', 'spruce', 'birch', 'maple', 'oak', 'walnut',
                'almond', 'peanut', 'soybean', 'corn', 'wheat', 'rice', 'barley', 'oat',
                'rye', 'millet', 'sorghum', 'quinoa', 'buckwheat', 'amaranth', 'teff',
                'chickpea', 'lentil', 'bean', 'pea', 'carrot', 'potato', 'tomato',
                'onion', 'garlic', 'ginger', 'turmeric', 'cinnamon', 'nutmeg', 'clove',
                'black pepper', 'white pepper', 'red pepper', 'chili', 'paprika',
                'saffron', 'safflower', 'sunflower', 'sesame', 'poppy', 'flax', 'chia',
                'hemp', 'coconut', 'palm', 'olive', 'avocado', 'grapeseed', 'cottonseed',
                'rapeseed', 'canola', 'mustard', 'castor', 'jojoba', 'argan', 'shea',
                'cocoa', 'coffee', 'tea', 'chamomile', 'peppermint', 'spearmint',
                'rosemary', 'thyme', 'oregano', 'basil', 'sage', 'marjoram', 'tarragon',
                'dill', 'fennel', 'coriander', 'cumin', 'cardamom', 'fenugreek',
                'asafoetida', 'sumac', 'zaatar', 'harissa', 'berbere', 'ras el hanout',
                'garam masala', 'curry', 'turmeric', 'ginger', 'galangal', 'lemongrass',
                'kaffir lime', 'makrut lime', 'calamansi', 'yuzu', 'sudachi', 'kabosu',
                'shikuwasa', 'tachibana', 'mikan', 'satsuma', 'mandarin', 'clementine',
                'tangerine', 'pomelo', 'grapefruit', 'lime', 'lemon', 'citron',
                'buddha hand', 'fingered citron', 'bergamot', 'bitter orange',
                'sweet orange', 'blood orange', 'valencia orange', 'navel orange',
                'seville orange', 'moro orange', 'tarocco orange', 'sanguinello orange'
            }
            
            for line in kegg_lines:
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        kegg_id = parts[0]
                        kegg_name = parts[1]
                        kegg_lower = kegg_name.lower()
                        
                        # 医薬品以外の成分を除外
                        if any(keyword in kegg_lower for keyword in exclude_keywords):
                            continue
                        
                        # 複数の類似度スコアを計算
                        exact_ratio = fuzz.ratio(original_lower, kegg_lower)
                        partial_ratio = fuzz.partial_ratio(original_lower, kegg_lower)
                        token_sort_ratio = fuzz.token_sort_ratio(original_lower, kegg_lower)
                        token_set_ratio = fuzz.token_set_ratio(original_lower, kegg_lower)
                        
                        # 正規化名での比較
                        norm_ratio = fuzz.ratio(original_norm.lower(), kegg_lower)
                        
                        # 英語名との比較（ボーナススコア）
                        english_bonus = 0
                        for english_name in english_names:
                            if english_name.lower() in kegg_lower:
                                english_bonus = 25  # 英語名一致のボーナスを増加
                                break
                        
                        # 部分一致ボーナス
                        partial_bonus = 0
                        if len(original_name) > 3:
                            for i in range(len(original_name) - 2):
                                substring = original_name[i:i+3].lower()
                                if substring in kegg_lower:
                                    partial_bonus = 15  # 部分一致ボーナスを増加
                                    break
                        
                        # 語尾パターンボーナス（ベンゾジアゼピン系など）
                        suffix_bonus = 0
                        if original_name.endswith('ゾラム') and 'alprazolam' in kegg_lower:
                            suffix_bonus = 20
                        elif original_name.endswith('パム') and 'pam' in kegg_lower:
                            suffix_bonus = 20
                        elif original_name.endswith('ラム') and 'lam' in kegg_lower:
                            suffix_bonus = 20
                        
                        # 総合スコア計算（英語名ボーナスを重視）
                        base_score = max(exact_ratio, partial_ratio, token_sort_ratio, token_set_ratio, norm_ratio)
                        total_score = base_score + english_bonus + partial_bonus + suffix_bonus
                        
                        candidates.append({
                            'kegg_id': kegg_id,
                            'kegg_name': kegg_name,
                            'similarity': total_score,
                            'exact_ratio': exact_ratio,
                            'partial_ratio': partial_ratio,
                            'token_sort_ratio': token_sort_ratio,
                            'token_set_ratio': token_set_ratio,
                            'norm_ratio': norm_ratio,
                            'english_bonus': english_bonus,
                            'partial_bonus': partial_bonus,
                            'suffix_bonus': suffix_bonus
                        })
            
            if candidates:
                # スコアでソート
                candidates.sort(key=lambda x: x['similarity'], reverse=True)
                best_match = candidates[0]
                
                logger.info(f"Best KEGG match for '{original_name}': {best_match['kegg_name']} (score: {best_match['similarity']})")
                logger.info(f"  - Exact: {best_match['exact_ratio']}, Partial: {best_match['partial_ratio']}, Token: {best_match['token_sort_ratio']}")
                logger.info(f"  - English bonus: {best_match['english_bonus']}, Partial bonus: {best_match['partial_bonus']}, Suffix bonus: {best_match['suffix_bonus']}")
                
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in _select_best_kegg_match: {e}")
            return None
    
    def _extract_pathways(self, kegg_text: str) -> List[str]:
        """KEGGテキストからパスウェイ情報を抽出"""
        pathways = []
        lines = kegg_text.split('\n')
        
        for line in lines:
            if line.startswith('PATHWAY'):
                parts = line.split()
                if len(parts) >= 2:
                    pathways.append(parts[1])
        
        return pathways[:5]  # 最大5個まで
    
    def _extract_targets(self, kegg_text: str) -> List[str]:
        """KEGGテキストからターゲット情報を抽出"""
        targets = []
        lines = kegg_text.split('\n')
        
        for line in lines:
            if line.startswith('TARGET'):
                parts = line.split()
                if len(parts) >= 2:
                    targets.append(parts[1])
        
        return targets[:3]  # 最大3個まで
    
    def _generate_warnings(self, results: Dict[str, Any]) -> List[str]:
        """警告メッセージを生成"""
        warnings = []
        
        if results['interactions']:
            warnings.append("⚠️ 薬剤間の相互作用が検出されました")
        
        if results['same_effect_warnings']:
            warnings.append("⚠️ 同効薬の重複投与が検出されました")
        
        if results['category_duplicates']:
            warnings.append("⚠️ 薬剤分類による重複が検出されました")
        
        if results['kegg_info']:
            warnings.append("⚠️ KEGG情報の取得に成功しました")
        
        if len(results['detected_drugs']) == 0:
            warnings.append("⚠️ 薬剤名を特定できませんでした")
        
        return warnings
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        if results['interactions']:
            recommendations.append("・薬剤師に必ずご相談ください")
            recommendations.append("・定期的な血液検査が必要な場合があります")
        
        if results['same_effect_warnings']:
            recommendations.append("・同効薬の重複投与を避けるように注意してください")
        
        if results['category_duplicates']:
            recommendations.append("・薬剤分類による重複を避けるように注意してください")
        
        if results['kegg_info']:
            recommendations.append("・KEGG情報を参考にしてください")
        
        recommendations.append("・この情報は参考情報です。最終判断は薬剤師にお任せください")
        
        return recommendations

    def _remove_dosage_form(self, drug_name: str) -> str:
        """剤形（錠、カプセル等）を除去して短縮名を取得"""
        import re
        if not drug_name:
            return ""
        # 剤形パターン
        dosage_patterns = [
            r'錠\d*', r'カプセル\d*', r'散\d*', r'液\d*', r'注射用\d*', r'軟膏\d*', 
            r'点眼\d*', r'坐剤\d*', r'内用液\d*', r'外用液\d*', r'顆粒\d*', r'シロップ\d*',
            r'\d+mg', r'\d+g', r'\d+ml', r'\d+L', r'\d+%', r'\d+単位'
        ]
        result = drug_name
        for pattern in dosage_patterns:
            result = re.sub(pattern, '', result)
        # 前後の空白を除去
        result = result.strip()
        return result if result else drug_name

    def _get_english_drug_names(self, japanese_name: str) -> List[str]:
        """日本語薬剤名を英語名に変換（正規化比較対応）"""
        drug_name_mapping = {
            # ベンゾジアゼピン系抗不安薬・睡眠薬
            'アルプラゾラム': ['alprazolam', 'xanax', 'U-31889'],
            'ブロマゼパム': ['bromazepam', 'lexotanil', 'ro-5-3350'],
            'クロルジアゼポキシド': ['chlordiazepoxide', 'librium', 'ro-5-0690'],
            'クロバザム': ['clobazam', 'frisium', 'HR-376'],
            'クロナゼパム': ['clonazepam', 'klonopin', 'ro-5-4023'],
            'ジアゼパム': ['diazepam', 'valium', 'ro-5-2807'],
            'エチゾラム': ['etizolam', 'depas', 'U-33400'],
            'フルニトラゼパム': ['flunitrazepam', 'rohypnol', 'ro-5-4200'],
            'フルラゼパム': ['flurazepam', 'dalmane', 'ro-5-6901'],
            'ハロキサゾラム': ['haloxazolam', 'somelin', 'CS-430'],
            'ロラゼパム': ['lorazepam', 'ativan', 'wy-4036'],
            'ミダゾラム': ['midazolam', 'versed', 'ro-21-3981'],
            'ニトラゼパム': ['nitrazepam', 'mogadon', 'ro-4-5360'],
            'オキサゼパム': ['oxazepam', 'serax', 'ro-5-6789'],
            'テマゼパム': ['temazepam', 'restoril', 'ro-5-5345'],
            'トリアゾラム': ['triazolam', 'halcion', 'U-33030'],
            
            # 非ベンゾジアゼピン系睡眠薬
            'ゾルピデム': ['zolpidem', 'ambien', 'stilnox', 'SL-800750'],
            'ゾピクロン': ['zopiclone', 'imovane', 'RP-27267'],
            'ゾレピデム': ['zaleplon', 'sonata', 'CL-284846'],
            'エスゾピクロン': ['eszopiclone', 'lunesta', 'S-20098'],
            
            # オレキシン受容体拮抗薬
            'ラメルテオン': ['ramelteon', 'rozerem', 'TAK-375'],
            'スボレキサント': ['suvorexant', 'belsomra', 'MK-4305'],
            'レンボレキサント': ['lemborexant', 'dayvigo', 'E-2006'],
            'ダロレキサント': ['daridorexant', 'quviviq', 'ACT-541468'],
            
            # バルビツール酸系
            'ペントバルビタール': ['pentobarbital', 'nembutal', 'ethyl-5-phenylbarbituric acid'],
            'フェノバルビタール': ['phenobarbital', 'luminal', 'phenobarbitone'],
            'セコバルビタール': ['secobarbital', 'seconal', 'quinalbarbitone'],
            'アモバルビタール': ['amobarbital', 'amytal', 'amylobarbitone'],
            'バルビタール': ['barbital', 'veronal', 'diethylbarbituric acid'],
            
            # その他の睡眠薬・抗不安薬
            'メタゼパム': ['metazepam', 'meptin', 'ro-5-5345'],
            'ロフラゼペート': ['lofrazepate', 'meilax', 'ro-5-3590'],
            'メキサゾラム': ['mexazolam', 'melex', 'CS-386'],
            'オキサゾラム': ['oxazolam', 'serenal', 'CS-370'],
            'リルマザフォン': ['rilmazafone', 'rhythmy', '450-191'],
            'プラゼパム': ['prazepam', 'centrax', 'ro-5-4082'],
            'タンドスピロン': ['tandospirone', 'sediel', 'SM-3997'],
            'トフィンパム': ['tofisopam', 'grandaxin', 'GR-2'],
            
            # 追加のベンゾジアゼピン系薬剤
            'クロチアゼパム': ['clotiazepam', 'veredorm', 'YB-2'],
            'エスタゾラム': ['estazolam', 'prosom', 'D-40TA'],
            'クロキサゾラム': ['cloxazolam', 'olcadil', 'MT-141'],
            'エスゾピクロン': ['eszopiclone', 'lunesta', 'S-20098'],
            'ハロキサゾラム': ['haloxazolam', 'somelin', 'CS-430'],
            'フルラゼパム': ['flurazepam', 'dalmane', 'ro-5-6901'],
            'フルニトラゼパム': ['flunitrazepam', 'rohypnol', 'ro-5-4200'],
            'ニトラゼパム': ['nitrazepam', 'mogadon', 'ro-4-5360'],
            'オキサゼパム': ['oxazepam', 'serax', 'ro-5-6789'],
            'ロラゼパム': ['lorazepam', 'ativan', 'wy-4036'],
            
            # バルビツール酸系薬剤
            'アモバルビタール': ['amobarbital', 'amytal', 'amylobarbitone'],
            'ブトクトアミド': ['butoctamide', 'listomin-s', 'BA-1'],
            'クロラール': ['chloral', 'chloral hydrate', 'trichloroacetaldehyde'],
            'ブロモバレリル': ['bromovaleryl', 'bromovalerylurea', 'bromisoval'],
            'ブロチゾラム': ['brotizolam', 'lendormin', 'WE-941'],
            
            # 漢方薬・生薬
            'トケイソウエキス': ['passiflora incarnata', 'passion flower', 'maypop'],
            
            # 既存の薬剤
            'ロゼレム': ['ramelteon', 'rozerem', 'TAK-375'],
            'ベルソムラ': ['suvorexant', 'belsomra', 'MK-4305'],
            'フルボキサミン': ['fluvoxamine', 'luvox', 'fluvoxamine maleate'],
            'タケキャブ': ['vonoprazan', 'takecab', 'TAK-438'],
            'デュビゴ': ['linaclotide', 'linzess', 'MD-1100'],
            'ランソプラゾール': ['lansoprazole', 'prevacid', 'AG-1749'],
            'クラリスロマイシン': ['clarithromycin', 'biaxin', 'TE-031'],
            'カルベジロール': ['carvedilol', 'coreg', 'BM-14190'],
            'バルサルタン': ['valsartan', 'diovan', 'CGP-48933'],
            'アスピリン': ['aspirin', 'acetylsalicylic acid', 'ASA'],
            'ワルファリン': ['warfarin', 'coumadin', 'warfarin sodium'],
            'シンバスタチン': ['simvastatin', 'zocor', 'MK-733'],
            'メトホルミン': ['metformin', 'glucophage', 'dimethylbiguanide'],
            'アトルバスタチン': ['atorvastatin', 'lipitor', 'CI-981'],
            'プラバスタチン': ['pravastatin', 'pravachol', 'CS-514'],
            'ロスバスタチン': ['rosuvastatin', 'crestor', 'ZD-4522'],
            'ピタバスタチン': ['pitavastatin', 'livalo', 'NK-104'],
            'ダビガトラン': ['dabigatran', 'pradaxa', 'BIBR-1048'],
            'リバーロキサバン': ['rivaroxaban', 'xarelto', 'BAY-59-7939'],
            'アピキサバン': ['apixaban', 'eliquis', 'BMS-562247'],
            'エドキサバン': ['edoxaban', 'savaysa', 'DU-176b'],
            'イブプロフェン': ['ibuprofen', 'advil', 'motrin'],
            'ロキソプロフェン': ['loxoprofen', 'loxonin', 'CS-600'],
            'ジクロフェナク': ['diclofenac', 'voltaren', 'GP-45840'],
            'アセトアミノフェン': ['acetaminophen', 'paracetamol', 'tylenol'],
            'カプトプリル': ['captopril', 'capoten', 'SQ-14225'],
            'エナラプリル': ['enalapril', 'vasotec', 'MK-421'],
            'リシノプリル': ['lisinopril', 'prinivil', 'MK-521'],
            'ロサルタン': ['losartan', 'cozaar', 'MK-954'],
            'カンデサルタン': ['candesartan', 'atacand', 'TCV-116'],
            'オルメサルタン': ['olmesartan', 'benicar', 'CS-866'],
            'プロプラノロール': ['propranolol', 'inderal', 'AY-64043'],
            'アテノロール': ['atenolol', 'tenormin', 'ICI-66082'],
            'ビソプロロール': ['bisoprolol', 'zebeta', 'EMD-33512'],
            'ニフェジピン': ['nifedipine', 'adalat', 'BAY-a-1040'],
            'アムロジピン': ['amlodipine', 'norvasc', 'UK-48340'],
            'ベラパミル': ['verapamil', 'calan', 'CP-16533'],
            'ジルチアゼム': ['diltiazem', 'cardizem', 'CRD-401'],
            'グリメピリド': ['glimepiride', 'amaryl', 'HOE-490'],
            'グリクラジド': ['gliclazide', 'diamicron', 'S-1702'],
            'シタグリプチン': ['sitagliptin', 'januvia', 'MK-0431'],
            'ビルダグリプチン': ['vildagliptin', 'galvus', 'LAF-237'],
            'リナグリプチン': ['linagliptin', 'tradjenta', 'BI-1356'],
            'アログリプチン': ['alogliptin', 'nesina', 'SYR-322'],
            'ダパグリフロジン': ['dapagliflozin', 'forxiga', 'BMS-512148'],
            'カナグリフロジン': ['canagliflozin', 'invokana', 'JNJ-28431754'],
            'エンパグリフロジン': ['empagliflozin', 'jardiance', 'BI-10773'],
            'エゼチミブ': ['ezetimibe', 'zetia', 'SCH-58235'],
            'ベザフィブラート': ['bezafibrate', 'bezalip', 'BM-15075'],
            'フェノフィブラート': ['fenofibrate', 'tricor', 'procetofen'],
            'ヘパリン': ['heparin', 'heparin sodium'],
            'ダルテパリン': ['dalteparin', 'fragmin', 'FR-860'],
            'エノキサパリン': ['enoxaparin', 'lovenox', 'PK-10169']
        }
        # 正規化関数
        def normalize(name):
            import re
            name = re.sub(r'（.*?）', '', name)
            name = re.sub(r'\(.*?\)', '', name)
            name = name.replace('販売中止', '')
            name = name.replace(' ', '').replace('　', '')
            name = re.sub(r'[0-9０-９.．・,，、.。()（）【】［］\[\]{}｛｝<>《》"\'\-―ー=+*/\\]', '', name)
            name = name.strip()
            return name
        norm_jp = normalize(japanese_name)
        for k, v in drug_name_mapping.items():
            if normalize(k) == norm_jp:
                return v
        return []

    def _create_diagnosis_templates(self):
        """診断カテゴリごとの理由・症状テンプレートを作成"""
        self.diagnosis_templates = {
            '睡眠薬': {
                'reason': 'どちらも睡眠を助けるお薬ですが、種類が異なり、同じ目的で使われることがあります。効果が重なり、必要以上に効きすぎてしまう可能性があります。',
                'symptoms': '日中の眠気やふらつきが強く出たり、ぼんやりすることが増えたりする可能性があります。'
            },
            '胃酸分泌抑制薬': {
                'reason': 'これらのお薬は、どちらも胃酸の分泌を抑える働きがあります。同じような働きのお薬を一緒に飲むことで、効果が強くなりすぎたり、副作用が出やすくなることがあります。',
                'symptoms': 'お腹の調子が悪くなる（下痢や便秘など）や、長期にわたる使用では栄養吸収に影響が出ることがあります。'
            },
            '抗生物質': {
                'reason': '複数の抗生物質を同時に使用すると、効果が強くなりすぎたり、副作用が増えることがあります。',
                'symptoms': '下痢や吐き気、腸内細菌バランスの乱れなどが起こることがあります。'
            },
            '併用禁忌': {
                'reason': '併用することで一方または両方の薬の作用が強くなりすぎたり、重篤な副作用が発生する危険性があります。',
                'symptoms': '強い眠気、意識障害、重篤な副作用（例：出血、肝障害など）が起こる可能性があります。'
            },
            '併用注意': {
                'reason': '併用することで薬の効果や副作用が変化することがあります。',
                'symptoms': '副作用が出やすくなったり、薬の効果が強く/弱くなることがあります。'
            }
        }

    def _create_category_mapping(self):
        """細分類→大分類マッピングを作成"""
        self.category_mapping = {
            'オレキシン受容体拮抗薬': '睡眠薬',
            'ベンゾジアゼピン系': '睡眠薬',
            'メラトニン受容体作動薬': '睡眠薬',
            'プロトンポンプ阻害薬': '胃酸分泌抑制薬',
            'カリウムイオン競合型アシッドブロッカー': '胃酸分泌抑制薬',
            'マクロライド系抗生物質': '抗生物質',
            'ニューキノロン系抗生物質': '抗生物質',
            # 必要に応じて追加
        }

    def analyze_drug_interactions_ai(self, drug_names: List[str]) -> Dict[str, Any]:
        """AIを活用した薬剤相互作用分析（臨床的に重要なリスクを特定）"""
        
        # 薬剤カテゴリの取得
        drug_categories = {}
        for drug in drug_names:
            category = self._predict_category(drug)
            drug_categories[drug] = category
        
        # 臨床的に重要な相互作用パターンの定義
        clinical_risks = {
            'blood_pressure_medications': {
                'categories': ['ace_inhibitor', 'arb', 'ca_antagonist', 'beta_blocker', 'diuretic', 'nitrate', 'arni'],
                'risk_level': 'high',
                'description': '血圧降下薬の多剤併用による低血圧リスク',
                'clinical_impact': 'めまい、失神、腎機能障害のリスク増加',
                'recommendation': '血圧の定期的なモニタリングが必要'
            },
            'gastric_medications': {
                'categories': ['ppi', 'p_cab'],
                'risk_level': 'medium',
                'description': '胃酸分泌抑制薬の重複投与',
                'clinical_impact': '胃酸分泌の過度な抑制、栄養吸収障害のリスク',
                'recommendation': '重複投与を避け、必要に応じて剤形を変更'
            },
            'anticoagulants': {
                'categories': ['anticoagulant', 'nsaid'],
                'risk_level': 'high',
                'description': '抗凝固薬とNSAIDsの併用による出血リスク',
                'clinical_impact': '消化管出血、脳出血のリスク増加',
                'recommendation': '出血の兆候に注意し、定期的な血液検査が必要'
            },
            'diabetes_medications': {
                'categories': ['diabetes_medication', 'corticosteroid'],
                'risk_level': 'medium',
                'description': '糖尿病治療薬とステロイドの併用',
                'clinical_impact': '血糖値の上昇、糖尿病コントロールの悪化',
                'recommendation': '血糖値の頻回モニタリングが必要'
            },
            'psychotropic_medications': {
                'categories': ['benzodiazepine', 'barbiturate', 'antidepressant', 'antipsychotic'],
                'risk_level': 'high',
                'description': '向精神薬の多剤併用による中枢抑制作用の増強',
                'clinical_impact': '過度の眠気、呼吸抑制、認知機能障害のリスク',
                'recommendation': '段階的な投与開始と慎重な用量調整が必要'
            },
            'cardiac_medications': {
                'categories': ['cardiac_glycoside', 'diuretic', 'ace_inhibitor'],
                'risk_level': 'high',
                'description': '心臓薬の多剤併用による電解質異常リスク',
                'clinical_impact': '低カリウム血症、ジゴキシン中毒のリスク増加',
                'recommendation': '電解質の定期的なモニタリングが必要'
            },
            'statin_interactions': {
                'categories': ['statin', 'antibiotic'],
                'risk_level': 'medium',
                'description': 'スタチンと抗生物質の併用による筋障害リスク',
                'clinical_impact': '横紋筋融解症、CK上昇のリスク増加',
                'recommendation': '筋肉痛などの症状に注意し、CK値のモニタリング'
            }
        }
        
        # リスク分析の実行
        detected_risks = []
        risk_summary = {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': [],
            'clinical_recommendations': []
        }
        
        # 各リスクパターンのチェック
        for risk_name, risk_info in clinical_risks.items():
            matching_categories = []
            matching_drugs = []
            
            for drug, category in drug_categories.items():
                if category in risk_info['categories']:
                    matching_categories.append(category)
                    matching_drugs.append(drug)
            
            # リスク条件の判定
            if len(matching_drugs) >= 2:  # 2剤以上の併用でリスク判定
                risk_detail = {
                    'risk_name': risk_name,
                    'risk_level': risk_info['risk_level'],
                    'description': risk_info['description'],
                    'clinical_impact': risk_info['clinical_impact'],
                    'recommendation': risk_info['recommendation'],
                    'involved_drugs': matching_drugs,
                    'involved_categories': matching_categories
                }
                
                detected_risks.append(risk_detail)
                
                # リスクレベル別に分類
                if risk_info['risk_level'] == 'high':
                    risk_summary['high_risk'].append(risk_detail)
                elif risk_info['risk_level'] == 'medium':
                    risk_summary['medium_risk'].append(risk_detail)
                else:
                    risk_summary['low_risk'].append(risk_detail)
        
        # 臨床推奨事項の生成
        clinical_recommendations = self._generate_clinical_recommendations(risk_summary, drug_categories)
        risk_summary['clinical_recommendations'] = clinical_recommendations
        
        return {
            'drug_categories': drug_categories,
            'detected_risks': detected_risks,
            'risk_summary': risk_summary,
            'total_risks': len(detected_risks),
            'high_risk_count': len(risk_summary['high_risk']),
            'medium_risk_count': len(risk_summary['medium_risk']),
            'low_risk_count': len(risk_summary['low_risk'])
        }
    
    def _generate_clinical_recommendations(self, risk_summary: Dict, drug_categories: Dict) -> List[str]:
        """臨床推奨事項の生成"""
        recommendations = []
        
        # 高リスクの推奨事項
        if risk_summary['high_risk']:
            recommendations.append("🚨 **高リスク相互作用が検出されました**")
            recommendations.append("・薬剤師への緊急相談が必要です")
            recommendations.append("・患者の状態を慎重にモニタリングしてください")
        
        # 血圧降下薬の多剤併用
        bp_meds = [drug for drug, cat in drug_categories.items() 
                  if cat in ['ace_inhibitor', 'arb', 'ca_antagonist', 'beta_blocker', 'diuretic', 'nitrate', 'arni']]
        if len(bp_meds) >= 3:
            recommendations.append("💊 **血圧降下薬の多剤併用**")
            recommendations.append("・低血圧によるめまい、失神のリスク")
            recommendations.append("・腎機能の定期的なモニタリングが必要")
            recommendations.append("・段階的な投与開始を推奨")
        
        # 胃薬の重複
        gastric_meds = [drug for drug, cat in drug_categories.items() 
                       if cat in ['ppi', 'p_cab']]
        if len(gastric_meds) >= 2:
            recommendations.append("🫀 **胃酸分泌抑制薬の重複投与**")
            recommendations.append("・PPIとP-CABの併用は避けるべき")
            recommendations.append("・剤形の変更を検討してください")
        
        # 抗凝固薬との併用
        anticoagulants = [drug for drug, cat in drug_categories.items() 
                         if cat == 'anticoagulant']
        nsaids = [drug for drug, cat in drug_categories.items() 
                 if cat == 'nsaid']
        if anticoagulants and nsaids:
            recommendations.append("🩸 **抗凝固薬とNSAIDsの併用**")
            recommendations.append("・出血リスクの増加")
            recommendations.append("・消化管出血の兆候に注意")
            recommendations.append("・定期的な血液検査が必要")
        
        # 向精神薬の多剤併用
        psychotropics = [drug for drug, cat in drug_categories.items() 
                        if cat in ['benzodiazepine', 'barbiturate', 'antidepressant', 'antipsychotic']]
        if len(psychotropics) >= 2:
            recommendations.append("🧠 **向精神薬の多剤併用**")
            recommendations.append("・中枢抑制作用の増強")
            recommendations.append("・呼吸抑制のリスク")
            recommendations.append("・段階的な投与開始を推奨")
        
        # 一般的な推奨事項
        recommendations.append("📋 **一般的な注意事項**")
        recommendations.append("・この情報は参考情報です")
        recommendations.append("・最終判断は薬剤師にお任せください")
        recommendations.append("・患者の状態変化に注意してください")
        
        return recommendations

    def _predict_category(self, drug_name: str) -> str:
        """薬剤カテゴリの予測（DrugService版）"""
        drug_lower = drug_name.lower()
        normalized_name = self._normalize_name(drug_name).lower()
        
        # 正確な薬剤分類マッピング
        exact_drug_mapping = {
            # PDE5阻害薬
            'タダラフィル': 'pde5_inhibitor',
            'シルデナフィル': 'pde5_inhibitor',
            'バルデナフィル': 'pde5_inhibitor',
            'アバナフィル': 'pde5_inhibitor',
            'ウデナフィル': 'pde5_inhibitor',
            
            # 硝酸薬
            'ニコランジル': 'nitrate',
            'ニトログリセリン': 'nitrate',
            'イソソルビド': 'nitrate',
            'ニトロプルシド': 'nitrate',
            
            # ARNI（心不全治療薬）
            'エンレスト': 'arni',
            'サクビトリル': 'arni',
            'バルサルタン': 'arb',  # ARNIの成分
            
            # Ca拮抗薬+ARB配合剤
            'テラムロ': 'ca_antagonist_arb_combination',
            'アムロジピン': 'ca_antagonist',
            'テルミサルタン': 'arb',
            
            # ACE阻害薬
            'エナラプリル': 'ace_inhibitor',
            'カプトプリル': 'ace_inhibitor',
            'リシノプリル': 'ace_inhibitor',
            'ペリンドプリル': 'ace_inhibitor',
            
            # P-CAB（胃薬）
            'タケキャブ': 'p_cab',
            'ボノプラザン': 'p_cab',
            'フォノプラザン': 'p_cab',
            
            # PPI（胃薬）
            'ランソプラゾール': 'ppi',
            'オメプラゾール': 'ppi',
            'エソメプラゾール': 'ppi',
            'ラベプラゾール': 'ppi',
            'パントプラゾール': 'ppi',
            
            # ベンゾジアゼピン系（正確な分類）
            'ジアゼパム': 'benzodiazepine',
            'クロナゼパム': 'benzodiazepine',
            'アルプラゾラム': 'benzodiazepine',
            'ロラゼパム': 'benzodiazepine',
            'テマゼパム': 'benzodiazepine',
            'ミダゾラム': 'benzodiazepine',
            'エスタゾラム': 'benzodiazepine',
            'フルラゼパム': 'benzodiazepine',
            'ニトラゼパム': 'benzodiazepine',
            'ブロマゼパム': 'benzodiazepine',
            'クロチアゼパム': 'benzodiazepine',
            'クロキサゾラム': 'benzodiazepine',
            'ハロキサゾラム': 'benzodiazepine',
            'メキサゾラム': 'benzodiazepine',
            'オキサゼパム': 'benzodiazepine',
            'オキサゾラム': 'benzodiazepine',
            'プラゼパム': 'benzodiazepine',
            'トリアゾラム': 'benzodiazepine',
            'エチゾラム': 'benzodiazepine',
            'フルニトラゼパム': 'benzodiazepine',
            'ブロチゾラム': 'benzodiazepine',
            
            # バルビツール酸系
            'フェノバルビタール': 'barbiturate',
            'アモバルビタール': 'barbiturate',
            'ペントバルビタール': 'barbiturate',
            'チオペンタール': 'barbiturate',
            'セコバルビタール': 'barbiturate',
            
            # オピオイド
            'モルヒネ': 'opioid',
            'コデイン': 'opioid',
            'フェンタニル': 'opioid',
            'オキシコドン': 'opioid',
            'ヒドロコドン': 'opioid',
            'トラマドール': 'opioid',
            'ペンタゾシン': 'opioid',
            'ブプレノルフィン': 'opioid',
            'メタドン': 'opioid',
            
            # NSAIDs
            'アスピリン': 'nsaid',
            'イブプロフェン': 'nsaid',
            'ロキソプロフェン': 'nsaid',
            'ジクロフェナク': 'nsaid',
            'メフェナム酸': 'nsaid',
            'インドメタシン': 'nsaid',
            'ナプロキセン': 'nsaid',
            'ケトプロフェン': 'nsaid',
            'セレコキシブ': 'nsaid',
            'メロキシカム': 'nsaid',
            'アセトアミノフェン': 'nsaid',
            
            # スタチン
            'シンバスタチン': 'statin',
            'アトルバスタチン': 'statin',
            'プラバスタチン': 'statin',
            'ロスバスタチン': 'statin',
            'フルバスタチン': 'statin',
            'ピタバスタチン': 'statin',
            
            # ARB
            'ロサルタン': 'arb',
            'カンデサルタン': 'arb',
            'イルベサルタン': 'arb',
            'オルメサルタン': 'arb',
            'アジルサルタン': 'arb',
            
            # β遮断薬
            'プロプラノロール': 'beta_blocker',
            'アテノロール': 'beta_blocker',
            'ビソプロロール': 'beta_blocker',
            'メトプロロール': 'beta_blocker',
            'カルベジロール': 'beta_blocker',
            'ネビボロール': 'beta_blocker',
            
            # Ca拮抗薬
            'ニフェジピン': 'ca_antagonist',
            'ベラパミル': 'ca_antagonist',
            'ジルチアゼム': 'ca_antagonist',
            'ニカルジピン': 'ca_antagonist',
            'ニソルジピン': 'ca_antagonist',
            'ベニジピン': 'ca_antagonist',
            'シルニジピン': 'ca_antagonist',
            
            # 利尿薬
            'フロセミド': 'diuretic',
            'ヒドロクロロチアジド': 'diuretic',
            'スピロノラクトン': 'diuretic',
            'トリアムテレン': 'diuretic',
            'アミロライド': 'diuretic',
            'ブメタニド': 'diuretic',
            
            # 抗ヒスタミン薬
            'フェニラミン': 'antihistamine',
            'クロルフェニラミン': 'antihistamine',
            'ジフェンヒドラミン': 'antihistamine',
            'セチリジン': 'antihistamine',
            'ロラタジン': 'antihistamine',
            'フェキソフェナジン': 'antihistamine',
            
            # 制酸薬
            'アルミニウム': 'antacid',
            'マグネシウム': 'antacid',
            'カルシウム': 'antacid',
            
            # 抗凝固薬
            'ワルファリン': 'anticoagulant',
            'ダビガトラン': 'anticoagulant',
            'リバーロキサバン': 'anticoagulant',
            'アピキサバン': 'anticoagulant',
            'エドキサバン': 'anticoagulant',
            'ヘパリン': 'anticoagulant',
            'エノキサパリン': 'anticoagulant',
            
            # 糖尿病治療薬
            'メトホルミン': 'diabetes_medication',
            'インスリン': 'diabetes_medication',
            'グリクラジド': 'diabetes_medication',
            'グリメピリド': 'diabetes_medication',
            'ピオグリタゾン': 'diabetes_medication',
            'シタグリプチン': 'diabetes_medication',
            'ビルダグリプチン': 'diabetes_medication',
            'リナグリプチン': 'diabetes_medication',
            'アログリプチン': 'diabetes_medication',
            'テネリグリプチン': 'diabetes_medication',
            
            # 抗生物質
            'アモキシシリン': 'antibiotic',
            'セファレキシン': 'antibiotic',
            'エリスロマイシン': 'antibiotic',
            'クラリスロマイシン': 'antibiotic',
            'アジスロマイシン': 'antibiotic',
            'ドキシサイクリン': 'antibiotic',
            'ミノサイクリン': 'antibiotic',
            'レボフロキサシン': 'antibiotic',
            'シプロフロキサシン': 'antibiotic',
            'ノルフロキサシン': 'antibiotic',
            'バンコマイシン': 'antibiotic',
            'テイコプラニン': 'antibiotic',
            'メロペネム': 'antibiotic',
            'イミペネム': 'antibiotic',
            'セフトリアキソン': 'antibiotic',
            
            # 抗うつ薬
            'フルオキセチン': 'antidepressant',
            'パロキセチン': 'antidepressant',
            'セルトラリン': 'antidepressant',
            'エスシタロプラム': 'antidepressant',
            'ベンラファキシン': 'antidepressant',
            'デュロキセチン': 'antidepressant',
            
            # 抗精神病薬
            'アリピプラゾール': 'antipsychotic',
            'リスペリドン': 'antipsychotic',
            'オランザピン': 'antipsychotic',
            'クエチアピン': 'antipsychotic',
            
            # 気管支拡張薬
            'テオフィリン': 'bronchodilator',
            'サルブタモール': 'bronchodilator',
            'フォルモテロール': 'bronchodilator',
            'サルメテロール': 'bronchodilator',
            
            # 吸入ステロイド薬
            'ブデソニド': 'inhaled_corticosteroid',
            'フルチカゾン': 'inhaled_corticosteroid',
            
            # ロイコトリエン受容体拮抗薬
            'モンテルカスト': 'leukotriene_receptor_antagonist',
            'ザフィルルカスト': 'leukotriene_receptor_antagonist',
            
            # 去痰薬
            'アセチルシステイン': 'mucolytic',
            'カルボシステイン': 'mucolytic',
            
            # 前立腺肥大症治療薬
            'フィナステリド': 'bph_medication',
            'デュタステリド': 'bph_medication',
            'タムスロシン': 'bph_medication',
            'シルドシン': 'bph_medication',
            'ナフトピジル': 'bph_medication',
            'ウラピジル': 'bph_medication',
            
            # その他の薬剤
            'ジゴキシン': 'cardiac_glycoside',
            'アミオダロン': 'antiarrhythmic',
            'メトトレキサート': 'antirheumatic',
            'プレドニゾロン': 'corticosteroid',
            'シクロスポリン': 'immunosuppressant',
            'タクロリムス': 'immunosuppressant',
            'アザチオプリン': 'immunosuppressant',
            'ミコフェノール酸': 'immunosuppressant',
            'レフルノミド': 'antirheumatic',
            'サラゾスルファピリジン': 'antirheumatic',
            'ブシラミン': 'antirheumatic',
            'ペニシラミン': 'antirheumatic'
        }
        
        # 1. 完全一致チェック（最も信頼性が高い）
        for drug, category in exact_drug_mapping.items():
            if drug.lower() in drug_lower or drug.lower() in normalized_name:
                return category
        
        # 2. 部分一致チェック（より厳密な条件）
        best_category = 'unknown'
        best_score = 0
        
        for drug, category in exact_drug_mapping.items():
            drug_lower_match = drug.lower()
            
            # より厳密な部分一致チェック
            if (drug_lower_match in drug_lower or 
                drug_lower_match in normalized_name or
                drug_lower in drug_lower_match):
                
                # 類似度スコアを計算
                score = self._calculate_pattern_similarity(drug_lower, drug_lower_match)
                if score > best_score and score >= 0.8:  # 80%以上の類似度に引き上げ
                    best_score = score
                    best_category = category
        
        return best_category

    def _calculate_pattern_similarity(self, drug_name: str, pattern: str) -> float:
        """パターンとの類似度を計算"""
        if not drug_name or not pattern:
            return 0.0
        
        # 完全一致
        if drug_name == pattern:
            return 1.0
        
        # 部分一致
        if pattern in drug_name or drug_name in pattern:
            return 0.9
        
        # 文字レベルでの類似度
        common_chars = sum(1 for c in pattern if c in drug_name)
        if len(pattern) > 0:
            char_similarity = common_chars / len(pattern)
        else:
            char_similarity = 0.0
        
        # 接頭辞・接尾辞の一致
        prefix_bonus = 0.0
        suffix_bonus = 0.0
        
        if drug_name.startswith(pattern[:3]) and len(pattern) >= 3:
            prefix_bonus = 0.3
        
        if drug_name.endswith(pattern[-3:]) and len(pattern) >= 3:
            suffix_bonus = 0.3
        
        return min(char_similarity + prefix_bonus + suffix_bonus, 1.0)

    def _normalize_name(self, drug_name: str) -> str:
        """薬剤名の正規化"""
        if not drug_name:
            return ""
        
        # 基本的な正規化
        normalized = drug_name.strip()
        
        # 剤形の除去
        dosage_forms = ['錠', 'カプセル', '散剤', '液剤', '注射剤', '軟膏', 'クリーム', '貼付剤', '吸入剤', '点眼剤', '点鼻剤']
        for form in dosage_forms:
            normalized = normalized.replace(form, '')
        
        # 数字と単位の除去
        normalized = re.sub(r'\d+\.?\d*\s*(mg|g|ml|μg|mcg)', '', normalized)
        normalized = re.sub(r'\d+', '', normalized)
        
        # 製薬会社名の除去（括弧内）
        normalized = re.sub(r'[（(].*?[）)]', '', normalized)
        
        # 特殊文字の除去
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized.strip()



 