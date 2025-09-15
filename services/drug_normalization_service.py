import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class DrugNormalizationService:
    """薬剤名正規化サービス（辞書＋あいまい一致＋信頼度）"""
    
    def __init__(self):
        # OCR誤読の正規化辞書
        self.ocr_aliases = {
            "ロ腔": "口腔",
            "口腔内崩壊錠": "口腔内崩壊",  # 揺れ吸収
            "OD錠": "口腔内崩壊",
            "cap": "カプセル",
            "錠剤": "錠",
            "カプセル剤": "カプセル",
            "顆粒剤": "顆粒",
            "散剤": "散",
            "液剤": "液",
            "ゲル剤": "ゲル",
            "軟膏剤": "軟膏",
            "クリーム剤": "クリーム",
            "貼付剤": "貼付",
            "テープ剤": "テープ",
        }
        
        # 相互作用判定用のタグ辞書
        self.interaction_tags = {
            # RAAS系
            "バルサルタン": ["ARB", "RAAS"],
            "テルミサルタン": ["ARB", "RAAS"],
            "カンデサルタン": ["ARB", "RAAS"],
            "オルメサルタン": ["ARB", "RAAS"],
            "イルベサルタン": ["ARB", "RAAS"],
            "ロサルタン": ["ARB", "RAAS"],
            "アジルサルタン": ["ARB", "RAAS"],
            "サクビトリル/バルサルタン": ["ARNI", "RAAS"],  # エンレスト
            "エンレスト": ["ARNI", "RAAS"],
            
            # CCB系
            "アムロジピン": ["CCB"],
            "ニフェジピン": ["CCB"],
            "ジルチアゼム": ["CCB"],
            "ベラパミル": ["CCB"],
            
            # 刺激性下剤
            "センノシド": ["STIM_LAX"],
            "センナ": ["STIM_LAX"],
            "ピコスルファートナトリウム": ["STIM_LAX"],
            "ラキソベロン": ["STIM_LAX"],
            
            # 抗血小板薬
            "アスピリン": ["ANTI_PLATELET"],
            "クロピドグレル": ["ANTI_PLATELET"],
            "プラスグレル": ["ANTI_PLATELET"],
            
            # 鎮痛薬
            "トラマドール": ["OPIOID"],
            "アセトアミノフェン": ["ANALGESIC"],
        }
        # 国内向け薬剤辞書（一般名・商品名・別名・剤形）
        self.drug_dictionary = {
            # マクロライド系抗生物質
            'クラリスロマイシン': {
                'normalized': 'クラリスロマイシン',
                'generic_name': 'クラリスロマイシン',
                'category': 'macrolide_antibiotic_cyp3a4_inhibitor',
                'aliases': ['クラリス', 'クラリシッド', 'クラリシッド錠', 'クラリスロマイシン錠'],
                'confidence': 1.0
            },
            
            # 睡眠薬
            'ベルソムラ': {
                'normalized': 'スボレキサント',
                'generic_name': 'スボレキサント',
                'category': 'orexin_receptor_antagonist',
                'aliases': ['スボレキサント', 'ベルソムラ錠', 'ベルソムラOD錠'],
                'confidence': 1.0
            },
            'デビゴ': {
                'normalized': 'レンボレキサント',
                'generic_name': 'レンボレキサント',
                'category': 'orexin_receptor_antagonist',
                'aliases': ['レンボレキサント', 'デビゴ錠', 'デビゴOD錠'],
                'confidence': 1.0
            },
            'ロゼレム': {
                'normalized': 'ラメルテオン',
                'generic_name': 'ラメルテオン',
                'category': 'melatonin_receptor_agonist',
                'aliases': ['ラメルテオン', 'ロゼレム錠'],
                'confidence': 1.0
            },
            
            # 抗うつ薬
            'フルボキサミン': {
                'normalized': 'フルボキサミン',
                'generic_name': 'フルボキサミン',
                'category': 'ssri_antidepressant',
                'aliases': ['フルボキサミン', 'ルボックス', 'ルボックス錠'],
                'confidence': 1.0
            },
            
            # 降圧薬
            'アムロジピン': {
                'normalized': 'アムロジピン',
                'generic_name': 'アムロジピン',
                'category': 'ca_antagonist',
                'aliases': ['アムロジピン', 'ノルバスク', 'ノルバスク錠'],
                'confidence': 1.0
            },
            
            # 胃薬
            'エソメプラゾール': {
                'normalized': 'エソメプラゾール',
                'generic_name': 'エソメプラゾール',
                'category': 'ppi',
                'aliases': ['エソメプラゾール', 'ネキシウム', 'ネキシウム錠', 'ネキシウムOD錠'],
                'confidence': 1.0
            },
            
            # PDE5阻害薬
            'タダラフィル': {
                'normalized': 'タダラフィル',
                'generic_name': 'タダラフィル',
                'category': 'pde5_inhibitor',
                'aliases': ['タダラフィル', 'シアリス', 'シアリス錠'],
                'confidence': 1.0
            },
            
            # 硝酸薬
            'ニコランジル': {
                'normalized': 'ニコランジル',
                'generic_name': 'ニコランジル',
                'category': 'nitrate',
                'aliases': ['ニコランジル', 'シグマート', 'シグマート錠'],
                'confidence': 1.0
            },
            
            # ARNI
            'エンレスト': {
                'normalized': 'サクビトリル/バルサルタン',
                'generic_name': 'サクビトリル/バルサルタン',
                'category': 'arni',
                'aliases': ['サクビトリル/バルサルタン', 'エンレスト錠'],
                'confidence': 1.0
            },
            
            # 配合剤
            'テラムロAP': {
                'normalized': 'テルミサルタン/アムロジピン',
                'generic_name': 'テルミサルタン/アムロジピン',
                'category': 'ca_antagonist_arb_combination',
                'aliases': ['テルミサルタン/アムロジピン', 'テラムロAP錠'],
                'confidence': 1.0
            },
            
            # ACE阻害薬
            'エナラプリル': {
                'normalized': 'エナラプリル',
                'generic_name': 'エナラプリル',
                'category': 'ace_inhibitor',
                'aliases': ['エナラプリル', 'レニベース', 'レニベース錠'],
                'confidence': 1.0
            },
            
            # P-CAB
            'タケキャブ': {
                'normalized': 'ボノプラザン',
                'generic_name': 'ボノプラザン',
                'category': 'p_cab',
                'aliases': ['ボノプラザン', 'タケキャブ錠'],
                'confidence': 1.0
            },
            
            # PPI
            'ランソプラゾール': {
                'normalized': 'ランソプラゾール',
                'generic_name': 'ランソプラゾール',
                'category': 'ppi',
                'aliases': ['ランソプラゾール', 'タケプロン', 'タケプロン錠', 'タケプロンOD錠'],
                'confidence': 1.0
            },
            
            # スタチン
            'アトルバスタチン': {
                'normalized': 'アトルバスタチン',
                'generic_name': 'アトルバスタチン',
                'category': 'statin',
                'aliases': ['アトルバスタチン', 'リピトール', 'リピトール錠'],
                'confidence': 1.0
            },
            
            # 抗血小板薬
            'クロピドグレル': {
                'normalized': 'クロピドグレル',
                'generic_name': 'クロピドグレル',
                'category': 'antiplatelet',
                'aliases': ['クロピドグレル', 'プラビックス', 'プラビックス錠'],
                'confidence': 1.0
            },
            
            # β遮断薬
            'ビソプロロール': {
                'normalized': 'ビソプロロール',
                'generic_name': 'ビソプロロール',
                'category': 'beta_blocker',
                'aliases': ['ビソプロロール', 'メインテート', 'メインテート錠'],
                'confidence': 1.0
            },
            
            # 鉄剤
            'フェロベンゾン': {
                'normalized': 'フェロベンゾン',
                'generic_name': 'フェロベンゾン',
                'category': 'iron_supplement',
                'aliases': ['フェロベンゾン', 'フェロベンゾン散'],
                'confidence': 1.0
            },
            
            # 鎮痛薬
            'ロキソニン': {
                'normalized': 'ロキソプロフェン',
                'generic_name': 'ロキソプロフェン',
                'category': 'nsaid',
                'aliases': ['ロキソプロフェン', 'ロキソニン', 'ロキソニン錠', 'ロキソニンテープ'],
                'confidence': 1.0
            },
            
            # ビタミン剤
            'アスパラCA': {
                'normalized': 'アスパラギン酸カリウム・マグネシウム',
                'generic_name': 'アスパラギン酸カリウム・マグネシウム',
                'category': 'vitamin_supplement',
                'aliases': ['アスパラギン酸カリウム・マグネシウム', 'アスパラCA', 'アスパラCA錠'],
                'confidence': 1.0
            }
        }
        
        # 信頼度閾値（実用精度向上のため厳格化）
        self.confidence_thresholds = {
            'high': 0.85,    # 高信頼度（確定）
            'medium': 0.75,  # 中信頼度（要確認）
            'low': 0.5       # 低信頼度（不明）
        }
    
    def normalize_drug_name(self, drug_name: str) -> Dict[str, Any]:
        """薬剤名の正規化（辞書＋あいまい一致＋信頼度）"""
        try:
            # 前処理
            cleaned_name = self._preprocess_drug_name(drug_name)
            logger.info(f"薬剤名正規化開始: '{drug_name}' -> '{cleaned_name}'")
            
            # 完全一致チェック
            if cleaned_name in self.drug_dictionary:
                result = self.drug_dictionary[cleaned_name].copy()
                result['original'] = drug_name
                result['cleaned'] = cleaned_name
                result['match_type'] = 'exact'
                logger.info(f"完全一致: {drug_name} -> {result['normalized']}")
                return result
            
            # あいまい一致チェック
            fuzzy_matches = self._fuzzy_match(cleaned_name)
            
            if fuzzy_matches:
                best_match = fuzzy_matches[0]
                confidence = best_match['confidence']
                
                if confidence >= self.confidence_thresholds['high']:
                    # 高信頼度：確定
                    result = self.drug_dictionary[best_match['name']].copy()
                    result['original'] = drug_name
                    result['cleaned'] = cleaned_name
                    result['confidence'] = confidence
                    result['match_type'] = 'fuzzy_high'
                    result['candidates'] = [m['name'] for m in fuzzy_matches[:3]]
                    logger.info(f"高信頼度マッチ: {drug_name} -> {result['normalized']} (信頼度: {confidence:.2f})")
                    return result
                
                elif confidence >= self.confidence_thresholds['medium']:
                    # 中信頼度：要確認
                    result = {
                        'normalized': best_match['name'],
                        'original': drug_name,
                        'cleaned': cleaned_name,
                        'confidence': confidence,
                        'match_type': 'fuzzy_medium',
                        'candidates': [m['name'] for m in fuzzy_matches[:3]],
                        'category': 'unknown',
                        'generic_name': '要確認',
                        'aliases': []
                    }
                    logger.info(f"中信頼度マッチ: {drug_name} -> {best_match['name']} (信頼度: {confidence:.2f})")
                    return result
            
            # 低信頼度またはマッチなし
            result = {
                'normalized': drug_name,
                'original': drug_name,
                'cleaned': cleaned_name,
                'confidence': 0.0,
                'match_type': 'unknown',
                'category': 'unknown',
                'generic_name': '不明',
                'aliases': []
            }
            logger.warning(f"マッチなし: {drug_name}")
            return result
            
        except Exception as e:
            logger.error(f"薬剤名正規化エラー: {e}")
            return {
                'normalized': drug_name,
                'original': drug_name,
                'cleaned': drug_name,
                'confidence': 0.0,
                'match_type': 'error',
                'category': 'unknown',
                'generic_name': 'エラー',
                'aliases': []
            }
    
    def fix_ocr_aliases(self, drug_name: str) -> str:
        """OCR誤読の正規化"""
        if not drug_name:
            return ""
        
        cleaned = drug_name
        for ocr_error, correct in self.ocr_aliases.items():
            cleaned = cleaned.replace(ocr_error, correct)
        
        return cleaned
    
    def get_interaction_tags(self, generic_name: str) -> set:
        """相互作用判定用のタグを取得"""
        if not generic_name:
            return set()
        
        # 直接マッチ
        if generic_name in self.interaction_tags:
            return set(self.interaction_tags[generic_name])
        
        # 部分マッチ（配合剤など）
        tags = set()
        for drug, drug_tags in self.interaction_tags.items():
            if drug in generic_name:
                tags.update(drug_tags)
        
        return tags
    
    def _preprocess_drug_name(self, drug_name: str) -> str:
        """薬剤名の前処理"""
        if not drug_name:
            return ""
        
        # OCR誤読の修正
        cleaned = self.fix_ocr_aliases(drug_name)
        
        # 空白除去
        cleaned = cleaned.strip()
        
        # 剤形表記の正規化
        cleaned = re.sub(r'錠$', '', cleaned)
        cleaned = re.sub(r'カプセル$', '', cleaned)
        cleaned = re.sub(r'散$', '', cleaned)
        cleaned = re.sub(r'液$', '', cleaned)
        cleaned = re.sub(r'テープ$', '', cleaned)
        
        # 用量表記の除去
        cleaned = re.sub(r'\d+mg', '', cleaned)
        cleaned = re.sub(r'\d+g', '', cleaned)
        cleaned = re.sub(r'\d+ml', '', cleaned)
        cleaned = re.sub(r'\d+μg', '', cleaned)
        
        # 特殊文字の除去
        cleaned = re.sub(r'[()（）]', '', cleaned)
        
        return cleaned.strip()
    
    def _fuzzy_match(self, drug_name: str) -> List[Dict[str, Any]]:
        """あいまい一致による薬剤名検索"""
        matches = []
        
        for dict_name, drug_info in self.drug_dictionary.items():
            # 1. 辞書名との類似度
            name_similarity = self._calculate_similarity(drug_name, dict_name)
            
            # 2. 別名との類似度
            alias_similarities = []
            for alias in drug_info.get('aliases', []):
                alias_similarity = self._calculate_similarity(drug_name, alias)
                alias_similarities.append(alias_similarity)
            
            # 最高類似度を取得
            max_alias_similarity = max(alias_similarities) if alias_similarities else 0.0
            best_similarity = max(name_similarity, max_alias_similarity)
            
            if best_similarity > 0.85:  # 実用精度向上のため閾値を0.85に厳格化
                matches.append({
                    'name': dict_name,
                    'confidence': best_similarity,
                    'similarity_type': 'name' if name_similarity > max_alias_similarity else 'alias'
                })
        
        # 信頼度でソート
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """文字列類似度の計算（Jaro-Winkler + Levenshtein）"""
        if not str1 or not str2:
            return 0.0
        
        # 正規化
        s1, s2 = str1.lower(), str2.lower()
        
        # 完全一致
        if s1 == s2:
            return 1.0
        
        # 部分一致
        if s1 in s2 or s2 in s1:
            return 0.9
        
        # SequenceMatcher（Ratcliff/Obershelp）
        sequence_similarity = SequenceMatcher(None, s1, s2).ratio()
        
        # 文字レベルの類似度
        char_similarity = self._char_level_similarity(s1, s2)
        
        # 重み付き平均
        weighted_similarity = (sequence_similarity * 0.7) + (char_similarity * 0.3)
        
        return weighted_similarity
    
    def _char_level_similarity(self, str1: str, str2: str) -> float:
        """文字レベルの類似度計算"""
        if not str1 or not str2:
            return 0.0
        
        # 文字セットの類似度
        set1, set2 = set(str1), set(str2)
        intersection = set1 & set2
        union = set1 | set2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def get_confidence_level(self, confidence: float) -> str:
        """信頼度レベルの取得"""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def validate_drug_list(self, drug_names: List[str]) -> Dict[str, Any]:
        """薬剤リストの検証"""
        results = {
            'valid_drugs': [],
            'uncertain_drugs': [],
            'invalid_drugs': [],
            'overall_confidence': 0.0
        }
        
        total_confidence = 0.0
        valid_count = 0
        
        for drug_name in drug_names:
            normalized = self.normalize_drug_name(drug_name)
            confidence = normalized.get('confidence', 0.0)
            
            if confidence >= self.confidence_thresholds['high']:
                results['valid_drugs'].append(normalized)
                total_confidence += confidence
                valid_count += 1
            elif confidence >= self.confidence_thresholds['medium']:
                results['uncertain_drugs'].append(normalized)
            else:
                results['invalid_drugs'].append(normalized)
        
        # 全体の信頼度を計算
        if valid_count > 0:
            results['overall_confidence'] = total_confidence / valid_count
        
        return results
