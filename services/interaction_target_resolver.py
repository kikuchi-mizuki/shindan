"""
相互作用対象薬特定サービス
最小パッチによる対象薬の正確な特定
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class InteractionTargetResolver:
    """相互作用対象薬特定サービス"""
    
    def __init__(self):
        # 1) ブランド→一般名の正規化
        self.BRAND2GENERIC = {
            "エンレスト": "サクビトリル/バルサルタン",
            "テラムロAP": "テルミサルタン/アムロジピン",
            "タケキャブ": "ボノプラザン",
            "ランソプラゾールOD": "ランソプラゾール",
        }
        
        # 2) 同義語と薬効クラス
        self.CLASS_MAP = {
            "タダラフィル": ["PDE5"],
            "ニコランジル": ["NITRATE_LIKE"],  # 硝酸薬相当
            "サクビトリル/バルサルタン": ["ARNI", "ARB"],  # ARB含有
            "テルミサルタン/アムロジピン": ["ARB", "CCB"],
            "エナラプリル": ["ACEI"],
            "ボノプラザン": ["PCAB"],
            "ランソプラゾール": ["PPI"],
        }
        
        self.ALIASES = {
            # 入力にブランド名が来ても拾えるように
            "エンレスト": "サクビトリル/バルサルタン",
            "テラムロAP": "テルミサルタン/アムロジピン",
            "タケキャブ": "ボノプラザン",
            "ランソプラゾールOD": "ランソプラゾール",
        }
    
    def canonicalize(self, name: str) -> str:
        """薬剤名を正規化"""
        name = name.strip()
        name = self.BRAND2GENERIC.get(name, name)
        name = self.ALIASES.get(name, name)
        return name
    
    def index_by_class(self, drugs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """薬剤を薬効クラス別にインデックス化"""
        buckets = {}
        for d in drugs:
            gname = self.canonicalize(d.get("name", d.get("generic", "")))
            for cls in self.CLASS_MAP.get(gname, []):
                buckets.setdefault(cls, []).append(gname)
        
        # 重複除去
        for k in buckets:
            buckets[k] = sorted(set(buckets[k]))
        return buckets
    
    def rule_raas_arni_plus_acei(self, bx: Dict[str, List[str]]) -> List[str]:
        """ARNIとACEIが同時に存在"""
        if bx.get("ARNI") and bx.get("ACEI"):
            return sorted(set(bx["ARNI"] + bx["ACEI"]))
        return []
    
    def rule_raas_overlap(self, bx: Dict[str, List[str]]) -> List[str]:
        """RAAS系(ACEI/ARB/ARNI)の複数併用"""
        raas = bx.get("ACEI", []) + bx.get("ARB", []) + bx.get("ARNI", [])
        raas = sorted(set(raas))
        return raas if len(raas) >= 2 else []
    
    def rule_pde5_nitrate(self, bx: Dict[str, List[str]]) -> List[str]:
        """PDE5阻害薬と硝酸薬相当の併用"""
        if bx.get("PDE5") and bx.get("NITRATE_LIKE"):
            return sorted(set(bx["PDE5"] + bx["NITRATE_LIKE"]))
        return []
    
    def rule_acid_dup(self, bx: Dict[str, List[str]]) -> List[str]:
        """PPI + P-CABの重複"""
        if bx.get("PPI") and bx.get("PCAB"):
            return sorted(set(bx["PPI"] + bx["PCAB"]))
        return []
    
    def rule_poly_antihypertensive(self, bx: Dict[str, List[str]]) -> List[str]:
        """降圧薬の多剤併用"""
        pool = []
        for k in ["ACEI", "ARB", "ARNI", "CCB", "NITRATE_LIKE"]:
            pool += bx.get(k, [])
        pool = sorted(set(pool))
        return pool if len(pool) >= 3 else []
    
    def format_targets(self, names: List[str]) -> str:
        """対象薬名をフォーマット"""
        return "、".join(names) if names else "（該当なし）"
    
    def resolve_targets(self, drugs: List[Dict[str, Any]]) -> Dict[str, str]:
        """薬剤リストから相互作用対象薬を特定"""
        try:
            bx = self.index_by_class(drugs)
            targets = {}
            
            # RAAS禁忌（ARNI＋ACEI）
            raas_contraindicated = self.rule_raas_arni_plus_acei(bx)
            if raas_contraindicated:
                targets["raas_contraindicated"] = self.format_targets(raas_contraindicated)
            
            # RAAS重複
            raas_overlap = self.rule_raas_overlap(bx)
            if raas_overlap:
                targets["raas_overlap"] = self.format_targets(raas_overlap)
            
            # PDE5＋硝酸薬相当
            pde5_nitrate = self.rule_pde5_nitrate(bx)
            if pde5_nitrate:
                targets["pde5_nitrate"] = self.format_targets(pde5_nitrate)
            
            # 胃酸抑制の重複
            acid_dup = self.rule_acid_dup(bx)
            if acid_dup:
                targets["acid_dup"] = self.format_targets(acid_dup)
            
            # 降圧薬の多剤併用
            poly_antihypertensive = self.rule_poly_antihypertensive(bx)
            if poly_antihypertensive:
                targets["poly_antihypertensive"] = self.format_targets(poly_antihypertensive)
            
            logger.info(f"Resolved targets: {targets}")
            return targets
            
        except Exception as e:
            logger.error(f"Target resolution failed: {e}")
            return {}
