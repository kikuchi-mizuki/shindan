"""
合議制抽出システム
Regex×LLM×辞書の3系統候補から2票以上で採用
"""
import re
import logging
from typing import List, Any, Tuple, Optional
from collections import Counter
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class ConsensusExtractor:
    """合議制抽出システム"""
    
    def __init__(self):
        # 薬剤名抽出の正規表現パターン
        self.drug_name_patterns = [
            # 基本的な薬剤名パターン
            r'([ぁ-んァ-ヶ一-龥A-Za-z0-9・ー]+?)(?:錠|カプセル|口腔内崩壊錠|顆粒|ゲル|散|液|テープ)',
            # 漢方薬パターン
            r'(ツムラ[ぁ-んァ-ヶ一-龥A-Za-z0-9・ー]+?)(?:エキス顆粒|エキス散)',
            # 配合剤パターン
            r'([ぁ-んァ-ヶ一-龥A-Za-z0-9・ー]+?[・][ぁ-んァ-ヶ一-龥A-Za-z0-9・ー]+?)(?:錠|カプセル)',
            # 外用薬パターン
            r'([ぁ-んァ-ヶ一-龥A-Za-z0-9・ー]+?)(?:ゲル|軟膏|クリーム|貼付剤)',
        ]
        
        # 薬剤辞書（商品名→一般名）
        self.drug_dictionary = {
            "オルケディア": "エボカルセト",
            "リンゼス": "リナクロチド",
            "ラキソベロン": "ピコスルファートナトリウム",
            "グーフィス": "エロビキシバット",
            "芍薬甘草湯": "芍薬甘草湯",
            "ツムラ芍薬甘草湯": "芍薬甘草湯",
            "ツムラ芍薬甘草湯エキス顆粒": "芍薬甘草湯",
            "ファモチジン": "ファモチジン",
            "ファモチジンロ腔内崩壊": "ファモチジン",
            "ファモチジン口腔内崩壊": "ファモチジン",
            "アスピリン腸溶": "アスピリン",
            "アスピリン": "アスピリン",
            "エンレスト": "サクビトリル/バルサルタン",
            "テルミサルタン・アムロジピン": "テルミサルタン・アムロジピン",
            "ニフェジピン": "ニフェジピン",
            "センノシド": "センノシド",
            "センナ": "センナ",
            "センナ実配合": "センナ",
            "沈降炭酸カルシウム": "沈降炭酸カルシウム",
            "フェブキソスタット": "フェブキソスタット",
            "ナルフラフィン塩酸塩": "ナルフラフィン塩酸塩",
            "ジクロフェナクナトリウム": "ジクロフェナクナトリウム",
            "モンテルカスト": "モンテルカスト",
        }
        
        # ノイズパターン
        self.noise_patterns = [
            r'^[0-9\s\-×]+$',
            r'^[年月日時分秒]+$',
            r'^[患者|医師|薬剤師|保険]+$',
            r'^[変更不可|患者希望|次頁に続く]+$',
            r'^[般|外|麻|向]+$',
            r'^[食前|食後|食間|就寝前|眠前|朝|昼|夕]+$',
        ]
    
    def extract_drugs_consensus(self, text: str, llm_drugs: List[str] = None) -> List[dict[str, Any]]:
        """合議制で薬剤を抽出"""
        try:
            # 3系統の候補を取得
            regex_candidates = self._extract_by_regex(text)
            dictionary_candidates = self._extract_by_dictionary(text)
            llm_candidates = llm_drugs or []
            
            logger.info(f"Extraction candidates - Regex: {len(regex_candidates)}ionary: {len(dictionary_candidates)}, LLM: {len(llm_candidates)}")
            
            # 合議制で最終候補を決定
            consensus_drugs = self._consensus_voting(
                regex_candidates, 
                dictionary_candidates, 
                llm_candidates
            )
            
            # KEGGで裏取りできるかチェック
            verified_drugs = self._verify_with_kegg(consensus_drugs)
            
            logger.info(f"Consensus extraction result: {len(verified_drugs)} drugs")
            return verified_drugs
            
        except Exception as e:
            logger.error(f"Consensus extraction error: {e}")
            return []
    
    def _extract_by_regex(self, text: str) -> List[str]:
        """正規表現で薬剤名を抽出"""
        candidates = set()
        
        for pattern in self.drug_name_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # グループ化された場合
                
                # ノイズチェック
                if not self._is_noise(match):
                    candidates.add(match.strip())
        
        return list(candidates)
    
    def _extract_by_dictionary(self, text: str) -> List[str]:
        """辞書マッチで薬剤名を抽出"""
        candidates = set()
        
        for brand_name, generic_name in self.drug_dictionary.items():
            if brand_name in text:
                candidates.add(generic_name)
        
        return list(candidates)
    
    def _consensus_voting(self, regex_candidates: List[str], 
                         dictionary_candidates: List[str], 
                         llm_candidates: List[str]) -> List[dict[str, Any]]:
        """合議制投票で最終候補を決定"""
        # 候補を正規化
        normalized_regex = [self._normalize_drug_name(d) for d in regex_candidates]
        normalized_dict = [self._normalize_drug_name(d) for d in dictionary_candidates]
        normalized_llm = [self._normalize_drug_name(d) for d in llm_candidates]
        
        # 投票カウント
        vote_counter = Counter()
        
        # 各系統の投票
        for candidate in normalized_regex:
            vote_counter[candidate] += 1
        
        for candidate in normalized_dict:
            vote_counter[candidate] += 1
        
        for candidate in normalized_llm:
            vote_counter[candidate] += 1
        
        # 2票以上で採用、1票は保留
        accepted_drugs = []
        pending_drugs = []
        
        for drug_name, votes in vote_counter.items():
            if votes >= 2:
                # 2票以上で採用
                accepted_drugs.append({
                    'generic': drug_name,
                    'votes': votes,
                    'sources': self._get_sources(drug_name, normalized_regex, normalized_dict, normalized_llm),
                    'confidence': min(0.95, 0.7 + (votes - 2) * 0.1)
                })
            elif votes == 1:
                # 1票は保留
                pending_drugs.append({
                    'generic': drug_name,
                    'votes': votes,
                    'sources': self._get_sources(drug_name, normalized_regex, normalized_dict, normalized_llm),
                    'confidence': 0.5
                })
        
        logger.info(f"Consensus voting: {len(accepted_drugs)} accepted, {len(pending_drugs)} pending")
        
        return accepted_drugs + pending_drugs
    
    def _normalize_drug_name(self, drug_name: str) -> str:
        """薬剤名を正規化"""
        if not drug_name:
            return ""
        
        # 剤形を除去
        normalized = drug_name
        form_patterns = ['錠', 'カプセル', '口腔内崩壊錠', '顆粒', 'ゲル', '散', '液', 'テープ', '軟膏', 'クリーム']
        for form in form_patterns:
            normalized = normalized.replace(form, "")
        
        # 用量表記を除去
        normalized = re.sub(r'\d+\.?\d*\s*(mg|g|ml|μg|mcg)', '', normalized)
        normalized = re.sub(r'\d+%', '', normalized)
        
        # 特殊文字を除去
        normalized = re.sub(r'[()（）【】]', '', normalized)
        
        # 辞書で正規化
        for brand, generic in self.drug_dictionary.items():
            if brand in normalized:
                normalized = generic
                break
        
        return normalized.strip()
    
    def _is_noise(self, text: str) -> bool:
        """ノイズかどうかを判定"""
        if not text or len(text) < 2:
            return True
        
        for pattern in self.noise_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _get_sources(self, drug_name: str, regex_candidates: List[str], 
                    dictionary_candidates: List[str], llm_candidates: List[str]) -> List[str]:
        """薬剤名の抽出元を取得"""
        sources = []
        
        if drug_name in regex_candidates:
            sources.append('regex')
        if drug_name in dictionary_candidates:
            sources.append('dictionary')
        if drug_name in llm_candidates:
            sources.append('llm')
        
        return sources
    
    def _verify_with_kegg(self, drugs: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """KEGGで裏取り"""
        try:
            from services.kegg_client import KeggClient
            kegg_client = KeggClient()
            
            verified_drugs = []
            
            for drug in drugs:
                generic_name = drug['generic']
                
                # KEGGで検索
                kegg_result = kegg_client.search_drug(generic_name)
                
                if kegg_result:
                    # KEGGで見つかった場合は信頼度を上げる
                    drug['kegg_verified'] = True
                    drug['kegg_id'] = kegg_result.get('kegg_id')
                    drug['confidence'] = min(0.98, drug['confidence'] + 0.2)
                else:
                    # KEGGで見つからない場合は保留
                    drug['kegg_verified'] = False
                    drug['confidence'] = max(0.3, drug['confidence'] - 0.1)
                
                verified_drugs.append(drug)
            
            return verified_drugs
            
        except Exception as e:
            logger.warning(f"KEGG verification failed: {e}")
            return drugs
    
    def get_extraction_stats(self, drugs: List[dict[str, Any]]) -> dict[str, Any]:
        """抽出統計を取得"""
        if not drugs:
            return {
                'total_drugs': 0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0,
                'kegg_verified': 0,
                'coverage': 0.0
            }
        
        high_conf = sum(1 for d in drugs if d.get('confidence', 0) >= 0.8)
        medium_conf = sum(1 for d in drugs if 0.5 <= d.get('confidence', 0) < 0.8)
        low_conf = sum(1 for d in drugs if d.get('confidence', 0) < 0.5)
        kegg_verified = sum(1 for d in drugs if d.get('kegg_verified', False))
        
        return {
            'total_drugs': len(drugs),
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf,
            'kegg_verified': kegg_verified,
            'coverage': min(1.0, len(drugs) / 10.0)  # 期待値10剤に対するカバレッジ
        }
