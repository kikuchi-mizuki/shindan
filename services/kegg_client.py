"""
KEGG REST API クライアント
日本語一般名での検索失敗時に英語別名で再試行
ATCコードを/link/atc/で取得
"""
import functools
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
        """
        logger.info(f"KEGG search for: {generic_ja}")
        
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
            "atc": atc
        }
