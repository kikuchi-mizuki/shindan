import logging
import re
from typing import List, Any, Tuple, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class DrugNormalizationService:
    """薬剤名正規化サービス（辞書＋あいまい一致＋信頼度）"""
    
    def __init__(self):
        # 永続学習ストア（誤読・同義語・タグなど）
        try:
            from .normalize_store import fix_misread, SYN, TAGS
            self._fix_misread = fix_misread
            self._dynamic_syn = SYN
            self._dynamic_tags = TAGS
        except Exception:
            self._fix_misread = lambda s: s
            self._dynamic_syn = {}
            self._dynamic_tags = {}
        # OCR誤読の正規化辞書（強化版）
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
        # 新規追加：画像照合で発見された誤認識パターン（最終版）
        "テラムロジン": "テラムロAP",  # 存在しない薬剤名→配合剤
        "テラムロプリド": "テラムロAP",  # OCR誤読パターン（AP→プリド）
        "テラムロリウム": "テラムロAP",  # OCR誤読パターン（AP→リウム）
        "ラベプラゾール": "ボノプラザン",  # PPI→P-CABの誤認識
        "ラベプラゾールナトリウム": "ボノプラザン",  # 同様の誤認識
        "タケキャブ": "ボノプラザン",  # 商品名→一般名
        # 睡眠薬ブランド→一般名
        "デエビゴ": "レンボレキサント",
        "デビゴ": "レンボレキサント",
        # OCRゆれ
        "デエビコ": "レンボレキサント",
        "デェビゴ": "レンボレキサント",
        # 電解質/ミネラル製剤（取り違え対策）
        "アスパラ-CA": "L-アスパラギン酸カルシウム",
        "アスパラーCA": "L-アスパラギン酸カルシウム",
        "アスパラーCA錠200": "L-アスパラギン酸カルシウム",
        "アスパラCA": "L-アスパラギン酸カルシウム",
        "アスパラＣＡ": "L-アスパラギン酸カルシウム",
        "アスパラK": "L-アスパラギン酸カリウム・L-アスパラギン酸マグネシウム",
        }
        
        # 相互作用判定用のタグ辞書（強化版）
        self.interaction_tags = {
            # RAAS系
            "サクビトリル": ["RAAS", "ARNI"],
            "バルサルタン": ["RAAS", "ARB"],
            "テルミサルタン": ["RAAS", "ARB"],
            "カンデサルタン": ["RAAS", "ARB"],
            "オルメサルタン": ["RAAS", "ARB"],
            "イルベサルタン": ["RAAS", "ARB"],
            "ロサルタン": ["RAAS", "ARB"],
            "アジルサルタン": ["RAAS", "ARB"],
            "エナラプリル": ["RAAS", "ACEI"],
            "リシノプリル": ["RAAS", "ACEI"],
            "サクビトリル/バルサルタン": ["ARNI", "RAAS"],  # エンレスト
            "エンレスト": ["ARNI", "RAAS"],
            # ACE阻害薬
            "エナラプリル": ["ACEI", "RAAS"],
            "リシノプリル": ["ACEI", "RAAS"],
            "ペリンドプリル": ["ACEI", "RAAS"],
            "カプトプリル": ["ACEI", "RAAS"],
            "イミダプリル": ["ACEI", "RAAS"],
            "テモカプリル": ["ACEI", "RAAS"],
            "シラザプリル": ["ACEI", "RAAS"],
            
            # CCB系
            "アムロジピン": ["CCB"],
            "ニフェジピン": ["CCB"],
            "ジルチアゼム": ["CCB"],
            "ベラパミル": ["CCB"],
            
            # 刺激性下剤
            "センノシド": ["STIM_LAX"],
            "センナ": ["STIM_LAX"],
            "センナ実": ["STIM_LAX"],
            "ピコスルファートナトリウム": ["STIM_LAX"],
            "ラキソベロン": ["STIM_LAX"],
            
            # 抗血小板薬
            "アスピリン": ["ANTI_PLATELET"],
            "クロピドグレル": ["ANTI_PLATELET"],
            "プラスグレル": ["ANTI_PLATELET"],

            # リン吸着薬（同効チェック用）
            "沈降炭酸カルシウム": ["PHOS_BINDER"],
            "セベラマー": ["PHOS_BINDER"],
            "ビキサロマー": ["PHOS_BINDER"],
            "炭酸ランタン": ["PHOS_BINDER"],
            # ブランド名の一部（マッピングが未整備でもタグ付けのため）
            "キックリン": ["PHOS_BINDER"],  # セベラマー
            
            # 鎮痛薬
            "トラマドール": ["OPIOID"],
            "アセトアミノフェン": ["ANALGESIC"],
            
            # PDE5阻害薬
            "タダラフィル": ["PDE5"],
            "シルデナフィル": ["PDE5"],
            "バルデナフィル": ["PDE5"],
            "アバナフィル": ["PDE5"],
            
            # 硝酸薬相当
            "ニコランジル": ["NITRATE"],
            "ニトログリセリン": ["NITRATE"],
            "イソソルビド": ["NITRATE"],
            
            # P-CAB
            "ボノプラザン": ["P-CAB"],
            "タケキャブ": ["P-CAB"],
            
            # PPI
            "ランソプラゾール": ["PPI"],
            "オメプラゾール": ["PPI"],
            "エソメプラゾール": ["PPI"],
            "ラベプラゾール": ["PPI"],
        }

        # 動的タグを上書き反映（学習内容を優先）
        for g, tags in self._dynamic_tags.items():
            if isinstance(tags, list) and tags:
                self.interaction_tags[g] = list(dict.fromkeys(tags))

        # 国内向け薬剤辞書（一般名・商品名・別名・剤形・配合剤対応）
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

            # 電解質・ミネラル（カルシウム製剤）
            'L-アスパラギン酸カルシウム': {
                'normalized': 'L-アスパラギン酸カルシウム',
                'generic_name': 'L-アスパラギン酸カルシウム',
                'category': 'electrolyte_mineral_calcium',
                'aliases': [
                    'アスパラ-CA', 'アスパラーCA', 'アスパラCA',
                    'アスパラ-CA錠200', 'アスパラーCA錠200', 'アスパラCA錠200',
                    'L-アスパラギン酸カルシウム錠200'
                ],
                'confidence': 1.0,
                'class_jp': '電解質・ミネラル（カルシウム製剤）'
            },
            
            # PDE5阻害薬
            'タダラフィル': {
                'normalized': 'タダラフィル',
                'generic_name': 'タダラフィル',
                'category': 'pde5_inhibitor',
                'aliases': ['タダラフィル', 'シアリス', 'シアリス錠'],
                'confidence': 1.0,
                'class_jp': 'PDE5阻害薬（前立腺肥大症・ED・肺高血圧）'
            },
            
            # 硝酸薬
            'ニコランジル': {
                'normalized': 'ニコランジル',
                'generic_name': 'ニコランジル',
                'category': 'nitrate',
                'aliases': ['ニコランジル', 'シグマート', 'シグマート錠'],
                'confidence': 1.0,
                'class_jp': '冠血管拡張薬（狭心症治療薬）'
            },
            
            # ARNI
            'エンレスト': {
                'normalized': 'サクビトリル/バルサルタン',
                'generic_name': 'サクビトリル/バルサルタン',
                'category': 'arni',
                'aliases': ['サクビトリル/バルサルタン', 'エンレスト錠'],
                'confidence': 1.0,
                'class_jp': 'ARNI（心不全治療薬）'
            },
            
            # 配合剤
            'テラムロAP': {
                'normalized': 'テルミサルタン/アムロジピン',
                'generic_name': 'テルミサルタン/アムロジピン',
                'category': 'ca_antagonist_arb_combination',
                'aliases': ['テルミサルタン/アムロジピン', 'テラムロAP錠', 'テラムロ', 'テラムロジン', 'テラムロプリド', 'テラムロリウム'],  # OCR誤認識パターンを追加
                'confidence': 1.0,
                'components': ['テルミサルタン', 'アムロジピン'],
                'class_jp': '降圧薬（ARB＋Ca拮抗薬）'
            },
            
            # P-CAB
            'タケキャブ': {
                'normalized': 'ボノプラザン',
                'generic_name': 'ボノプラザン',
                'category': 'p_cab',
                'aliases': ['ボノプラザン', 'タケキャブ錠', 'タケキャブOD錠', 'ラベプラゾール', 'ラベプラゾールナトリウム'],  # OCR誤認識パターンを追加
                'confidence': 1.0,
                'class_jp': 'P-CAB（新型胃酸抑制薬）'
            },
            
            # PPI
            'ランソプラゾール': {
                'normalized': 'ランソプラゾール',
                'generic_name': 'ランソプラゾール',
                'category': 'ppi',
                'aliases': ['ランソプラゾール', 'タケプロン', 'タケプロン錠', 'タケプロンOD錠'],
                'confidence': 1.0,
                'class_jp': 'PPI（プロトンポンプ阻害薬）'
            },
            
            # ACE阻害薬
            'エナラプリル': {
                'normalized': 'エナラプリル',
                'generic_name': 'エナラプリル',
                'category': 'ace_inhibitor',
                'aliases': ['エナラプリル', 'レニベース', 'レニベース錠'],
                'confidence': 1.0,
                'class_jp': 'ACE阻害薬'
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
        
        # 製薬会社名マッピング（成分逆引き用）
        self.manufacturer_mapping = {
            "サンド": {
                "common_drugs": ["タダラフィル", "シアリス"],
                "categories": ["pde5_inhibitor"],
                "brand_mappings": {
                    "タダラフィル": "シアリス",
                    "シアリス": "タダラフィル"
                }
            },
            "トーワ": {
                "common_drugs": ["ニコランジル", "エナラプリル", "ランソプラゾール"],
                "categories": ["nitrate", "ace_inhibitor", "ppi"],
                "brand_mappings": {
                    "ニコランジル": "シグマート",
                    "エナラプリル": "レニベース",
                    "ランソプラゾール": "タケプロン"
                }
            },
            "サワイ": {
                "common_drugs": ["テラムロAP", "テルミサルタン/アムロジピン"],
                "categories": ["ca_antagonist_arb_combination"],
                "brand_mappings": {
                    "テラムロAP": "テルミサルタン/アムロジピン",
                    "テルミサルタン/アムロジピン": "テラムロAP"
                }
            },
            "武田": {
                "common_drugs": ["タケキャブ", "ボノプラザン"],
                "categories": ["p_cab"],
                "brand_mappings": {
                    "タケキャブ": "ボノプラザン",
                    "ボノプラザン": "タケキャブ"
                }
            },
            "ノバルティス": {
                "common_drugs": ["エンレスト", "サクビトリル/バルサルタン"],
                "categories": ["arni"],
                "brand_mappings": {
                    "エンレスト": "サクビトリル/バルサルタン",
                    "サクビトリル/バルサルタン": "エンレスト"
                }
            }
        }
        
        # 類似薬剤候補辞書（スコアリング用）
        self.similar_drug_candidates = {
            "テラムロジン": {
                "candidates": [
                    {"name": "テラムロAP", "score": 0.95, "reason": "配合剤の誤認識"},
                    {"name": "テルミサルタン", "score": 0.7, "reason": "成分の一部"},
                    {"name": "アムロジピン", "score": 0.6, "reason": "成分の一部"}
                ]
            },
            "テラムロプリド": {
                "candidates": [
                    {"name": "テラムロAP", "score": 0.95, "reason": "OCR誤読パターン"},
                    {"name": "テルミサルタン", "score": 0.7, "reason": "成分の一部"},
                    {"name": "アムロジピン", "score": 0.6, "reason": "成分の一部"}
                ]
            },
            "テラムロリウム": {
                "candidates": [
                    {"name": "テラムロAP", "score": 0.95, "reason": "OCR誤読パターン（AP→リウム）"},
                    {"name": "テルミサルタン", "score": 0.7, "reason": "成分の一部"},
                    {"name": "アムロジピン", "score": 0.6, "reason": "成分の一部"}
                ]
            },
            "ラベプラゾール": {
                "candidates": [
                    {"name": "ボノプラザン", "score": 0.9, "reason": "PPI→P-CABの誤認識"},
                    {"name": "ランソプラゾール", "score": 0.8, "reason": "類似PPI"},
                    {"name": "オメプラゾール", "score": 0.7, "reason": "類似PPI"}
                ]
            },
            "ラベプラゾールナトリウム": {
                "candidates": [
                    {"name": "ボノプラザン", "score": 0.95, "reason": "PPI→P-CABの誤認識"},
                    {"name": "ランソプラゾール", "score": 0.8, "reason": "類似PPI"}
                ]
            }
        }
        
        # 信頼度閾値（実用精度向上のため厳格化）
        self.confidence_thresholds = {
            'high': 0.85,    # 高信頼度（確定）
            'medium': 0.75,  # 中信頼度（要確認）
            'low': 0.5       # 低信頼度（不明）
        }
    
    def normalize_drug_name(self, drug_name: str) -> dict[str, Any]:
        """薬剤名の正規化（辞書＋あいまい一致＋信頼度＋類似候補スコアリング）"""
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
            
            # 類似候補スコアリング（新機能）
            similar_candidates = self._get_similar_candidates(cleaned_name)
            if similar_candidates:
                best_candidate = similar_candidates[0]
                if best_candidate['score'] >= 0.9:  # 高スコアの場合は採用
                    candidate_name = best_candidate['name']
                    if candidate_name in self.drug_dictionary:
                        result = self.drug_dictionary[candidate_name].copy()
                        result['original'] = drug_name
                        result['cleaned'] = cleaned_name
                        result['confidence'] = best_candidate['score']
                        result['match_type'] = 'similar_candidate'
                        result['correction_reason'] = best_candidate['reason']
                        result['candidates'] = [c['name'] for c in similar_candidates[:3]]
                        logger.info(f"類似候補マッチ: {drug_name} -> {result['normalized']} (スコア: {best_candidate['score']:.2f}, 理由: {best_candidate['reason']})")
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
        """OCR誤読の正規化（強化版）"""
        if not drug_name:
            return ""
        
        # まず学習済みの誤読修正を適用
        try:
            name = self._fix_misread(drug_name)
        except Exception:
            name = drug_name
        
        # 完全一致チェック（最優先）
        if name in self.ocr_aliases:
            corrected = self.ocr_aliases[name]
            logger.info(f"Direct OCR correction: {name} -> {corrected}")
            return corrected
        
        # 部分一致チェック
        cleaned = name
        for ocr_error, correct in self.ocr_aliases.items():
            if ocr_error in cleaned:
                cleaned = cleaned.replace(ocr_error, correct)
                logger.info(f"OCR alias correction: {ocr_error} -> {correct} in '{drug_name}'")
        
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
    
    def _fuzzy_match(self, drug_name: str) -> List[dict[str, Any]]:
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
    
    def validate_drug_list(self, drug_names: List[str]) -> dict[str, Any]:
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
    
    def _aliases(self, name: str) -> List[str]:
        """同義語展開してから配合名を分解"""
        # 同義語辞書から展開
        expanded_name = self._dynamic_syn.get(name, name)
        
        # 配合名を分解（・/＋\+\sで分割）
        parts = re.split(r"[・/＋\+\s]", expanded_name)
        parts = [p.strip() for p in parts if p.strip()]
        
        # 各パーツを同義語辞書で展開
        result = []
        for part in (parts if parts else [expanded_name]):
            result.append(self._dynamic_syn.get(part, part))
        
        return result
    
    def tags_for_drug(self, drug: dict) -> set[str]:
        """薬剤情報から相互作用タグを取得（配合名分解対応）"""
        names = [
            drug.get("generic") or "",
            drug.get("brand") or "",
            drug.get("raw") or ""
        ]
        
        tags = set()
        for name in names:
            if not name:
                continue
                
            # 同義語展開・配合名分解
            aliases = self._aliases(name)
            
            # 各エイリアスからタグを取得
            for alias in aliases:
                tags.update(self.get_interaction_tags(alias))
        
        return tags
    
    def fuzzy_match_drug(self, drug_name: str, threshold: float = 80.0) -> Optional[dict]:
        """ファジーマッチングによる薬剤名検索"""
        try:
            from rapidfuzz import process, fuzz
            
            # 辞書から候補を取得
            candidates = []
            for key, info in self.drug_dictionary.items():
                candidates.append(key)
                candidates.extend(info.get('aliases', []))
            
            # ファジーマッチング実行
            match, score = process.extractOne(drug_name, candidates, scorer=fuzz.ratio)
            
            if score >= threshold:
                # マッチした薬剤の詳細情報を取得
                for key, info in self.drug_dictionary.items():
                    if match == key or match in info.get('aliases', []):
                        return {
                            'matched_name': match,
                            'score': score,
                            'drug_info': info,
                            'confidence': min(score / 100.0, 1.0)
                        }
            
            return None
            
        except ImportError:
            logger.warning("rapidfuzz not available, falling back to exact matching")
            return None
        except Exception as e:
            logger.error(f"Fuzzy matching failed: {e}")
            return None
    
    def _get_similar_candidates(self, drug_name: str) -> List[dict]:
        """類似候補スコアリングによる薬剤名検索"""
        if not drug_name:
            return []
        
        # 類似候補辞書から検索
        if drug_name in self.similar_drug_candidates:
            candidates = self.similar_drug_candidates[drug_name]['candidates']
            # スコアでソート
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates
        
        return []
    
    def get_manufacturer_hints(self, drug_name: str, manufacturer: str = None) -> List[dict]:
        """製薬会社名を活用した成分逆引き（強化版）"""
        hints = []
        
        if not drug_name and not manufacturer:
            return hints
        
        # 製薬会社名から推測
        if manufacturer and manufacturer in self.manufacturer_mapping:
            manufacturer_info = self.manufacturer_mapping[manufacturer]
            
            # 1. 直接マッチング
            for drug in manufacturer_info['common_drugs']:
                if drug in self.drug_dictionary:
                    drug_info = self.drug_dictionary[drug]
                    hints.append({
                        'name': drug,
                        'score': 0.8,
                        'reason': f'{manufacturer}の主要薬剤',
                        'category': manufacturer_info['categories'][0] if manufacturer_info['categories'] else 'unknown'
                    })
            
            # 2. ブランド名マッピング
            if 'brand_mappings' in manufacturer_info:
                for brand, generic in manufacturer_info['brand_mappings'].items():
                    if brand in drug_name or drug_name in brand:
                        hints.append({
                            'name': generic,
                            'score': 0.9,
                            'reason': f'{manufacturer}のブランド名マッピング',
                            'brand_name': brand
                        })
        
        # 薬剤名から製薬会社を推測
        if drug_name:
            for manufacturer_name, info in self.manufacturer_mapping.items():
                if drug_name in info['common_drugs']:
                    hints.append({
                        'manufacturer': manufacturer_name,
                        'score': 0.9,
                        'reason': f'{drug_name}の主要製造会社'
                    })
                
                # ブランド名からも推測
                if 'brand_mappings' in info:
                    for brand, generic in info['brand_mappings'].items():
                        if brand in drug_name or drug_name in brand:
                            hints.append({
                                'manufacturer': manufacturer_name,
                                'score': 0.85,
                                'reason': f'{drug_name}のブランド名から推測',
                                'brand_name': brand
                            })
        
        return hints
    
    def llm_correct_drug_name(self, ocr_text: str) -> Optional[str]:
        """ChatGPTによる薬剤名補正"""
        try:
            import openai
            
            # OpenAI APIキーの確認
            if not openai.api_key:
                logger.warning("OpenAI API key not configured")
                return None
            
            # 薬剤辞書から候補を取得
            drug_candidates = []
            for key, info in self.drug_dictionary.items():
                drug_candidates.append(f"{key} ({info.get('generic_name', '')})")
                for alias in info.get('aliases', []):
                    drug_candidates.append(f"{alias} ({info.get('generic_name', '')})")
            
            candidates_text = "\n".join(drug_candidates[:50])  # 最初の50個に制限
            
            prompt = f"""
以下のOCRテキストから薬剤名を特定し、最も適切な候補を1つ選んでください。

OCRテキスト: "{ocr_text}"

候補薬剤:
{candidates_text}

注意事項:
- 配合剤の場合は成分名を「/」で区切って記載
- 商品名の場合は一般名も併記
- 確信が持てない場合は「不明」と回答

回答形式: 薬剤名のみ（例: テラムロAP または テルミサルタン/アムロジピン）
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            corrected_name = response.choices[0].message.content.strip()
            
            if corrected_name and corrected_name != "不明":
                logger.info(f"LLM correction: '{ocr_text}' -> '{corrected_name}'")
                return corrected_name
            
            return None
            
        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return None
    
    def two_stage_drug_matching(self, drug_name: str) -> dict:
        """二段階チェックによる薬剤名マッチング"""
        # 第1段階: 完全一致
        exact_match = self._exact_match(drug_name)
        if exact_match:
            return {
                'method': 'exact',
                'result': exact_match,
                'confidence': 1.0
            }
        
        # 第2段階: ファジーマッチ
        fuzzy_match = self.fuzzy_match_drug(drug_name)
        if fuzzy_match:
            return {
                'method': 'fuzzy',
                'result': fuzzy_match,
                'confidence': fuzzy_match['confidence']
            }
        
        # 第3段階: LLM補正
        corrected_name = self.llm_correct_drug_name(drug_name)
        if corrected_name:
            # 補正後の名前で再検索
            corrected_match = self._exact_match(corrected_name)
            if corrected_match:
                return {
                    'method': 'llm_corrected',
                    'result': corrected_match,
                    'confidence': 0.8
                }
        
        # 全て失敗
        return {
            'method': 'failed',
            'result': None,
            'confidence': 0.0
        }
    
    def _exact_match(self, drug_name: str) -> Optional[dict]:
        """完全一致による薬剤名検索"""
        # 辞書から直接検索
        if drug_name in self.drug_dictionary:
            return self.drug_dictionary[drug_name]
        
        # エイリアスから検索
        for key, info in self.drug_dictionary.items():
            if drug_name in info.get('aliases', []):
                return info
        
        return None
