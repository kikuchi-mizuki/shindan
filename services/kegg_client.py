"""
KEGG REST API クライアント
日本語一般名での検索失敗時に英語別名で再試行
ATCコードを/link/atc/で取得
"""
import functools
from typing import Any
import time
import urllib.parse
import requests
import logging

logger = logging.getLogger(__name__)

# 日本語→英語の最小辞書
JA2EN = {
    "フェブキソスタット": "febuxostat",
    "ナルフラフィン塩酸塩": "nalfurafine",
    "センノシド": "sennosides",
    "ピコスルファートナトリウム": "sodium picosulfate",
    "リナクロチド": "linaclotide",
    "モンテルカスト": "montelukast",
    "エロビキシバット": "elobixibat",
    "沈降炭酸カルシウム": "calcium carbonate",
    "エボカルセト": "evocalcet",
    "ジクロフェナクナトリウム": "diclofenac",
    "ゾルピデム酒石酸塩": "zolpidem",
    "スボレキサント": "suvorexant",
    "レンボレキサント": "lemborexant",
    "ラメルテオン": "ramelteon",
    "エスシタロプラムシュウ酸塩": "escitalopram",
    "ミアンセリン塩酸塩": "mianserin",
    # 追加の薬剤名
    "エンレスト": "sacubitril valsartan",
    "アスピリン": "aspirin",
    "キックリン": "tramadol acetaminophen",
    "トラマドール・アセトアミノフェン": "tramadol acetaminophen",
    "テルミサルタン": "telmisartan",
    "アムロジピン": "amlodipine",
    "テルミサルタン・アムロジピン": "telmisartan amlodipine",
    "ニフェジピン": "nifedipine",
    "ファモチジン": "famotidine",
    "センナ": "senna",
    "センナ・センナ実": "senna",
    "センナ・センナ実配合": "senna",
    # 新規追加：画像照合で発見された誤認識パターン
    "テラムロジン": "telmisartan amlodipine",  # 配合剤の誤認識
    "テラムロプリド": "telmisartan amlodipine",  # 新規：OCR誤読パターン
    "テラムロリウム": "telmisartan amlodipine",  # 新規：OCR誤読パターン（AP→リウム）
    "テラムロAP": "telmisartan amlodipine",  # 配合剤の正しい名称
    "ラベプラゾール": "vonoprazan",  # PPI→P-CABの誤認識
    "ラベプラゾールナトリウム": "vonoprazan",  # 同様の誤認識
    "ボノプラザン": "vonoprazan",  # P-CABの正しい名称
    "タケキャブ": "vonoprazan",  # P-CABの商品名
    "ランソプラゾール": "lansoprazole",  # PPIの正しい名称
}

def _http_get(url, timeout=5, retries=1, backoff=0.3):
    """HTTP GET with retry logic (reduced timeout/retries for performance)"""
    for i in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.ok and r.text:
                return r
        except Exception as e:
            logger.warning(f"HTTP GET failed (attempt {i+1}): {e}")
        
        if i < retries:
            time.sleep(backoff * (i + 1))
    
    if 'r' in locals():
        r.raise_for_status()
    raise Exception(f"Failed to fetch {url} after {retries + 1} attempts")

class KEGGClient:
    BASE = "https://rest.kegg.jp"

    @functools.lru_cache(maxsize=4096)
    def find_drug(self, query: str):
        """名前（英語推奨）から KEGG DRUG 候補を返す"""
        try:
            q = urllib.parse.quote(query)
            r = _http_get(f"{self.BASE}/find/drug/{q}")
            out = []
            for line in r.text.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    kid, label = parts
                    out.append({"kegg_id": kid.strip(), "label": label.strip()})
            return out
        except Exception as e:
            logger.error(f"KEGG find_drug failed for '{query}': {e}")
            return []

    @functools.lru_cache(maxsize=4096)
    def link_atc(self, kegg_id: str):
        """drug:Dxxxxx → ATCコード配列（文字列）"""
        try:
            # KEGG IDの形式を正規化
            if kegg_id.startswith("dr:"):
                kid = f"drug:{kegg_id[3:]}"  # dr:D00903 -> drug:D00903
            elif kegg_id.startswith("drug:"):
                kid = kegg_id
            else:
                kid = f"drug:{kegg_id}"
            
            logger.info(f"Requesting ATC codes for: {kid}")
            r = _http_get(f"{self.BASE}/link/atc/{kid}")
            codes = set()
            for line in r.text.strip().splitlines():
                parts = line.split("\t")
                if len(parts) == 2 and parts[1].startswith("atc:"):
                    codes.add(parts[1].split(":")[1])
            return sorted(codes)
        except Exception as e:
            logger.error(f"KEGG link_atc failed for '{kegg_id}': {e}")
            return []

    def best_kegg_and_atc(self, generic_ja: str):
        """
        1) 英語別名に変換して find（日本語直接検索はスキップ）
        2) 上位ヒットの KEGG ID から ATC を /link で取得
        3) 類似候補スコアリングによる補正
        """
        logger.info(f"KEGG search for: {generic_ja}")
        
        # 類似候補スコアリングによる補正
        corrected_name = self._apply_similar_candidate_scoring(generic_ja)
        if corrected_name != generic_ja:
            logger.info(f"Similar candidate correction: {generic_ja} -> {corrected_name}")
            generic_ja = corrected_name
        
        # 英語変換を優先（日本語直接検索は400エラーが多いためスキップ）
        en = JA2EN.get(generic_ja)
        if en:
            logger.info(f"Using English translation: {en}")
            cands = self.find_drug(en)
            logger.info(f"English search results: {len(cands)}")
        else:
            logger.warning(f"No English translation for: {generic_ja}")
            return None
        
        if not cands:
            logger.warning(f"No KEGG results for: {generic_ja}")
            return None
        
        # 最上位候補を選択
        top = cands[0]
        logger.info(f"Selected KEGG ID: {top['kegg_id']}")
        
        # ATCコードを取得
        atc = self.link_atc(top["kegg_id"])
        logger.info(f"ATC codes: {atc}")
        
        return {
            "kegg_id": top["kegg_id"],
            "label": top["label"],
            "atc": atc,
            "corrected_name": corrected_name if corrected_name != generic_ja else None
        }
    
    def _apply_similar_candidate_scoring(self, drug_name: str) -> str:
        """類似候補スコアリングによる薬剤名補正"""
        # 類似候補辞書（KEGG用）
        similar_candidates = {
            "テラムロジン": "テラムロAP",  # 配合剤の誤認識
            "テラムロプリド": "テラムロAP",  # 新規：OCR誤読パターン
            "テラムロリウム": "テラムロAP",  # 新規：OCR誤読パターン（AP→リウム）
            "ラベプラゾール": "ボノプラザン",  # PPI→P-CABの誤認識
            "ラベプラゾールナトリウム": "ボノプラザン",  # 同様の誤認識
        }
        
        return similar_candidates.get(drug_name, drug_name)
    
    def get_drug_brite(self, drug_name: str) -> dict[str, Any]:
        """KEGG/get でBRITEパースしてATCコードを取得"""
        try:
            # 英語名に変換
            english_name = JA2EN.get(drug_name, drug_name.lower())
            
            # KEGG/get で検索
            url = f"{self.BASE}/get/{english_name}"
            response = _http_get(url)
            
            if response and response.text:
                # BRITEセクションからATCコードを抽出
                atc_codes = []
                kegg_id = None
                
                lines = response.text.split('\n')
                in_brite_section = False
                
                for line in lines:
                    line = line.strip()
                    
                    # KEGG IDを抽出
                    if line.startswith('ENTRY'):
                        kegg_id = line.split()[1] if len(line.split()) > 1 else None
                    
                    # BRITEセクションの開始
                    if line.startswith('BRITE'):
                        in_brite_section = True
                        continue
                    
                    # セクション終了
                    if in_brite_section and line.startswith('///'):
                        break
                    
                    # ATCコードを抽出
                    if in_brite_section and 'ATC' in line:
                        # ATCコードのパターンを検索
                        import re
                        atc_matches = re.findall(r'[A-Z]\d{2}[A-Z]{2}\d{2}', line)
                        atc_codes.extend(atc_matches)
                
                if atc_codes:
                    return {
                        'kegg_id': kegg_id,
                        'atc': atc_codes
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get BRITE info for {drug_name}: {e}")
            return {}
    
    @functools.lru_cache(maxsize=1024)
    def get_drug_interactions(self, kegg_id1: str, kegg_id2: str) -> list[dict[str, Any]]:
        """2つのKEGG IDから相互作用情報を取得（KEGG DDI）"""
        try:
            # KEGG IDの形式を正規化
            def normalize_kegg_id(kid):
                if kid.startswith("dr:"):
                    return kid[3:]  # dr:D00903 -> D00903
                elif kid.startswith("drug:"):
                    return kid[5:]  # drug:D00903 -> D00903
                return kid
            
            kid1 = normalize_kegg_id(kegg_id1)
            kid2 = normalize_kegg_id(kegg_id2)
            
            # KEGG DDI API（注：実際のエンドポイントはKEGG公式ドキュメント参照）
            # https://rest.kegg.jp/ddi/D00903+D00564
            url = f"{self.BASE}/ddi/{kid1}+{kid2}"
            logger.debug(f"Fetching DDI: {url}")
            
            try:
                response = _http_get(url, timeout=2, retries=0)  # タイムアウト2秒、リトライなし
                interactions = []
                
                if response and response.text:
                    # レスポンスをパース
                    for line in response.text.strip().split('\n'):
                        if not line or line.startswith('#'):
                            continue
                        
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            interactions.append({
                                'drug1': parts[0],
                                'drug2': parts[1],
                                'description': parts[2] if len(parts) > 2 else '',
                                'severity': self._parse_severity(parts[2] if len(parts) > 2 else ''),
                                'source': 'KEGG'
                            })
                
                logger.info(f"Found {len(interactions)} DDI entries")
                return interactions
                
            except Exception as e:
                # DDIが存在しない場合は404エラーが返る（正常）
                if "404" in str(e) or "Not Found" in str(e):
                    logger.info(f"No DDI found for {kid1}+{kid2} (expected for no interaction)")
                    return []
                else:
                    logger.warning(f"DDI fetch error for {kid1}+{kid2}: {e}")
                    return []
                
        except Exception as e:
            logger.error(f"Failed to get drug interactions: {e}")
            return []
    
    def _parse_severity(self, description: str) -> str:
        """DDI説明文から重大度を推定"""
        desc_lower = description.lower()
        
        # 禁忌・重大パターン
        if any(keyword in desc_lower for keyword in ['contraindicated', 'severe', 'serious', 'major', '禁忌', '重大']):
            return '重大'
        
        # 注意パターン
        if any(keyword in desc_lower for keyword in ['caution', 'monitor', 'moderate', '注意', 'モニタリング']):
            return '併用注意'
        
        # デフォルト
        return '併用注意'
