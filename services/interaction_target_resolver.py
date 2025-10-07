"""
相互作用対象薬特定サービス
最小パッチによる対象薬の正確な特定
"""
import re
import unicodedata
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
        
        # 2) 同義語辞書
        self.ALIASES = {
            # テルミサルタン/アムロジピンの表記ゆれ補正
            "テルミサルタン＋アムロジピン": "テルミサルタン/アムロジピン",
            "テルミサルタン・アムロジピン": "テルミサルタン/アムロジピン",
            "テルミサルタン／アムロジピン": "テルミサルタン/アムロジピン",
            "アムロジピン/テルミサルタン": "テルミサルタン/アムロジピン",
            "テラムロAP": "テルミサルタン/アムロジピン",
            # 誤読の補正（ログで頻出）
            "テラムロリウム": "テルミサルタン/アムロジピン",
            "テラムロプリド": "テルミサルタン/アムロジピン",
            "ラベプラゾールナトリウム": "ラベプラゾール",  # ←最終的にボノプラザンへはブランド正規化で吸収
        }
        
        # 3) 薬効クラス付与
        self.CLASS_MAP = {
            "タダラフィル": ["PDE5"],
            "ニコランジル": ["NITRATE_LIKE"],   # 硝酸薬相当
            "サクビトリル/バルサルタン": ["ARNI", "ARB", "RAAS"],  # ARNIはARB性分含む
            "テルミサルタン/アムロジピン": ["ARB", "CCB", "RAAS"],
            "エナラプリル": ["ACEI", "RAAS"],
            "ボノプラザン": ["PCAB"],
            "ランソプラゾール": ["PPI"],
            
            # 新しく追加した薬剤クラス
            "クラリスロマイシン": ["ABX", "CYP3A_STRONG_INHIB"],
            "スボレキサント": ["OREXIN_ANTAGONIST", "SEDATIVE"],
            "レンボレキサント": ["OREXIN_ANTAGONIST", "SEDATIVE"],
            "ラメルテオン": ["MELATONIN_AGONIST", "SEDATIVE", "RAMELTEON"],
            "フルボキサミン": ["SSRI", "RAMELTEON_CONTRA_PAIR"],
            "アムロジピン": ["DHP_CCB"],
            "エソメプラゾール": ["PPI"],
            
            # 今回の11剤に対応するクラス
            "センノシド": ["STIM_LAX", "LAXATIVE"],
            "ピコスルファートナトリウム": ["STIM_LAX", "LAXATIVE"],
            "リナクロチド": ["GCC_AGONIST", "LAXATIVE"],
            "エロビキシバット": ["IBAT_INHIB", "LAXATIVE"],
            "芍薬甘草湯": ["GLYCYRRHIZA", "KAMPO"],
            "沈降炭酸カルシウム": ["CALCIUM_BINDER", "PHOSPHATE_BINDER"],
            "エボカルセト": ["CALCIMIMETIC"],
        }
    
    def sanitize_name(self, s: str) -> str:
        """名前の正規化（全角→半角、記号統一、メーカー表記や用量の除去）"""
        s = unicodedata.normalize("NFKC", s or "")
        # メーカー「～」や( )内を削除
        s = re.sub(r'「.*?」|\(.*?\)', '', s)
        # 用量・剤形（例: 5mg, 10mg, OD）を末尾から削除
        s = re.sub(r'\s*(\d+(\.\d+)?)\s*mg\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\bOD\b', '', s, flags=re.IGNORECASE)
        # 区切り記号の統一（＋ + ・ ・ / ／ などをスラッシュに）
        s = re.sub(r'[＋+･・／/，,]', '/', s)
        # 余分な空白の整理
        s = re.sub(r'\s+', '', s).strip()
        return s

    def canonicalize(self, name: str) -> str:
        """薬剤名を正規化"""
        name = self.sanitize_name(name)
        name = self.BRAND2GENERIC.get(name, name)
        name = self.ALIASES.get(name, name)
        # ブランドを一般名にもう一段変換（例：テラムロAP → テルミサルタン/アムロジピン）
        name = self.BRAND2GENERIC.get(name, name)
        return name
    
    def index_by_class(self, drugs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """薬剤を薬効クラス別にインデックス化"""
        buckets = {}
        for d in drugs:
            # 薬剤名の取得（generic > name > brand > raw の順で優先）
            drug_name = d.get("generic") or d.get("name") or d.get("brand") or d.get("raw", "")
            gname = self.canonicalize(drug_name)
            for cls in self.CLASS_MAP.get(gname, []):
                buckets.setdefault(cls, []).append(gname)
        
        # 重複除去
        for k in buckets:
            buckets[k] = sorted(set(buckets[k]))
        return buckets
    
    def rule_raas_arni_plus_acei(self, bx: Dict[str, List[str]]) -> List[str]:
        """ARNI + ACEI 同時"""
        if bx.get("ARNI") and bx.get("ACEI"):
            return sorted(set(bx["ARNI"] + bx["ACEI"]))
        return []
    
    def rule_raas_overlap(self, bx: Dict[str, List[str]]) -> List[str]:
        """ACEI/ARB/ARNI 重複"""
        acei = set(bx.get("ACEI", []))
        arb  = set(bx.get("ARB", []))
        arni = set(bx.get("ARNI", []))
        raas = sorted(acei | arb | arni)   # 和集合で3系統すべてを対象化
        return raas if len(raas) >= 2 else []
    
    def rule_pde5_nitrate(self, bx: Dict[str, List[str]]) -> List[str]:
        """PDE5 + 硝酸薬相当"""
        if bx.get("PDE5") and bx.get("NITRATE_LIKE"):
            return sorted(set(bx["PDE5"] + bx["NITRATE_LIKE"]))
        return []
    
    def rule_acid_dup(self, bx: Dict[str, List[str]]) -> List[str]:
        """PPI + P-CAB"""
        if bx.get("PPI") and bx.get("PCAB"):
            return sorted(set(bx["PPI"] + bx["PCAB"]))
        return []
    
    def rule_poly_antihypertensive(self, bx: Dict[str, List[str]]) -> List[str]:
        """降圧薬多剤（3系統以上）"""
        pool = []
        for k in ["ACEI", "ARB", "ARNI", "CCB", "NITRATE_LIKE"]:
            pool += bx.get(k, [])
        pool = sorted(set(pool))
        return pool if len(pool) >= 3 else []
    
    def rule_contra_orexin_strong_cyp3a(self, bx: Dict[str, List[str]]) -> List[str]:
        """オレキシン受容体拮抗薬 + CYP3A強阻害薬（禁忌）"""
        if bx.get("OREXIN_ANTAGONIST") and bx.get("CYP3A_STRONG_INHIB"):
            return sorted(set(bx["OREXIN_ANTAGONIST"] + bx["CYP3A_STRONG_INHIB"]))
        return []
    
    def rule_contra_fluvoxamine_ramelteon(self, bx: Dict[str, List[str]]) -> List[str]:
        """フルボキサミン + ラメルテオン（禁忌）"""
        if bx.get("RAMELTEON") and bx.get("RAMELTEON_CONTRA_PAIR"):
            return ["ラメルテオン", "フルボキサミン"]
        return []
    
    def rule_caution_cyp3a_dhpccb(self, bx: Dict[str, List[str]]) -> List[str]:
        """CYP3A阻害薬 + DHP-CCB（注意）"""
        cyp3a_drugs = bx.get("CYP3A_STRONG_INHIB", []) + bx.get("CYP3A_MOD_INHIB", [])
        if cyp3a_drugs and bx.get("DHP_CCB"):
            return sorted(set(cyp3a_drugs + bx["DHP_CCB"]))
        return []
    
    def rule_poly_sedatives(self, bx: Dict[str, List[str]]) -> List[str]:
        """鎮静薬多剤（2剤以上）"""
        pool = bx.get("SEDATIVE", [])
        return sorted(set(pool)) if len(set(pool)) >= 2 else []
    
    def rule_poly_laxatives(self, bx: Dict[str, List[str]]) -> List[str]:
        """下剤の多剤併用（2剤以上）"""
        pool = bx.get("LAXATIVE", [])
        return sorted(set(pool)) if len(set(pool)) >= 2 else []
    
    def rule_glycyrrhiza_stim_lax(self, bx: Dict[str, List[str]]) -> List[str]:
        """甘草 + 刺激性下剤（低K血症リスク）"""
        if bx.get("GLYCYRRHIZA") and bx.get("STIM_LAX"):
            return sorted(set(bx["GLYCYRRHIZA"] + bx["STIM_LAX"]))
        return []
    
    def rule_calcium_binder_caution(self, bx: Dict[str, List[str]]) -> List[str]:
        """Ca製剤による吸収低下（一般的注意）"""
        pool = bx.get("CALCIUM_BINDER", [])
        return sorted(set(pool)) if pool else []
    
    def rule_calcimimetic_electrolyte(self, bx: Dict[str, List[str]]) -> List[str]:
        """カルシミメティクス + 下剤多剤（低Ca/QT延長リスク）"""
        if bx.get("CALCIMIMETIC") and bx.get("LAXATIVE") and len(set(bx.get("LAXATIVE", []))) >= 2:
            return sorted(set(bx["CALCIMIMETIC"] + bx["LAXATIVE"]))
        return []
    
    def join_targets(self, names: List[str]) -> str:
        """対象薬名をフォーマット（一般名で統一）"""
        if not names:
            return "（該当なし）"
        
        # 表示用の正規化（一般名で統一）
        PREFER_GENERIC = True
        GENERIC_DISPLAY = {
            "エンレスト": "サクビトリル/バルサルタン",
            "テラムロAP": "テルミサルタン/アムロジピン",
            "タケキャブ": "ボノプラザン",
            "ランソプラゾールOD": "ランソプラゾール",
        }
        
        def display_name(name: str) -> str:
            # 用量情報を除去（5mg、100mg等）
            import re
            name = re.sub(r'\s*\d+(\.\d+)?\s*mg\b', '', name, flags=re.IGNORECASE)
            # メーカー情報を除去（「トーワ」、「サンド」等）
            name = re.sub(r'「.*?」', '', name)
            # 余分な空白を整理
            name = re.sub(r'\s+', '', name).strip()
            
            # 一般名に変換
            return GENERIC_DISPLAY.get(name, name) if PREFER_GENERIC else name
        
        # 一般名で統一して結合
        normalized_names = [display_name(name) for name in names]
        # 重複除去
        unique_names = []
        for name in normalized_names:
            if name not in unique_names:
                unique_names.append(name)
        
        # RAAS重複の場合は特定の順序で表示
        if len(unique_names) == 3 and all(x in unique_names for x in ["エナラプリル", "サクビトリル/バルサルタン", "テルミサルタン/アムロジピン"]):
            ordered_names = ["エナラプリル", "サクビトリル/バルサルタン", "テルミサルタン/アムロジピン"]
            return "、".join(ordered_names)
        
        return "、".join(unique_names)
    
    def explain_raas_overlap(self, targets: List[str]) -> str:
        """RAAS重複の理由を一行補足"""
        # ARNIが含まれていればARB含有の旨を補足
        if any("サクビトリル/バルサルタン" in x for x in targets):
            return "（注：ARNIはARB成分〈バルサルタン〉を含むため）"
        return ""
    
    def build_report(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """薬剤リストから相互作用レポートを生成"""
        try:
            bx = self.index_by_class(drugs)
            findings = []
            
            # RAAS禁忌（ARNI＋ACEI）
            t = self.rule_raas_arni_plus_acei(bx)
            raas_contraindicated = t
            if t:
                findings.append({
                    "severity": "重大",
                    "title": "RAAS禁忌（ARNI＋ACEIの同時併用）",
                    "targets": self.join_targets(t),
                    "action": "禁忌（36時間ルール）。ACEI中止後36時間空けてARNIへ切替。開始/変更後1–2週でCr/eGFR/K、血圧を再評価。",
                    "priority": 0  # 禁忌（最優先）
                })
            
            # RAAS重複（RAAS禁忌がヒットした場合は除外）
            t = self.rule_raas_overlap(bx)
            if t and not raas_contraindicated:
                action = "低血圧・高K・腎機能悪化リスク。原則併用回避を検討。" + self.explain_raas_overlap(t)
                action += "開始/変更後1–2週でCr/eGFR/K、血圧を再評価。"
                findings.append({
                    "severity": "重大",
                    "title": "RAAS重複（原則併用回避：ACEI/ARB/ARNI）",
                    "targets": self.join_targets(t),
                    "action": action,
                    "priority": 0  # 重大リスク
                })
            
            # PDE5＋硝酸薬相当
            t = self.rule_pde5_nitrate(bx)
            if t:
                findings.append({
                    "severity": "重大",
                    "title": "PDE5阻害薬＋硝酸薬相当（禁忌）",
                    "targets": self.join_targets(t),
                    "action": "禁忌。重度低血圧・失神リスクのため併用回避。",
                    "priority": 0  # 禁忌（最優先）
                })
            
            # 胃酸抑制の重複
            t = self.rule_acid_dup(bx)
            if t:
                findings.append({
                    "severity": "併用注意",
                    "title": "胃酸抑制薬の重複",
                    "targets": self.join_targets(t),
                    "action": "同効薬の重複投与に注意。適応・目的が同じなら簡素化を検討。",
                    "priority": 5  # 軽微
                })
            
            # 降圧薬の多剤併用
            t = self.rule_poly_antihypertensive(bx)
            if t:
                findings.append({
                    "severity": "併用注意",
                    "title": "降圧薬の多剤併用",
                    "targets": self.join_targets(t),
                    "action": "過度な降圧リスク。血圧・腎機能・K値のモニタリングを強化。",
                    "priority": 2  # 血圧異常リスク
                })
            
            # オレキシン受容体拮抗薬 + CYP3A強阻害薬（禁忌）
            t = self.rule_contra_orexin_strong_cyp3a(bx)
            if t:
                findings.append({
                    "severity": "重大",
                    "title": "オレキシン受容体拮抗薬＋CYP3A強阻害薬（禁忌）",
                    "targets": self.join_targets(t),
                    "action": "禁忌。CYP3A強阻害によりオレキシン薬の曝露が大幅上昇。併用回避。",
                    "priority": 0  # 禁忌（最優先）
                })
            
            # フルボキサミン + ラメルテオン（禁忌）
            t = self.rule_contra_fluvoxamine_ramelteon(bx)
            if t:
                findings.append({
                    "severity": "重大",
                    "title": "フルボキサミン＋ラメルテオン（禁忌）",
                    "targets": self.join_targets(t),
                    "action": "禁忌。フルボキサミンによりラメルテオンの曝露が大幅上昇。併用回避。",
                    "priority": 0  # 禁忌（最優先）
                })
            
            # CYP3A阻害薬 + DHP-CCB（注意）
            t = self.rule_caution_cyp3a_dhpccb(bx)
            if t:
                findings.append({
                    "severity": "併用注意",
                    "title": "CYP3A阻害薬＋DHP-CCB（注意）",
                    "targets": self.join_targets(t),
                    "action": "CYP3A阻害によりDHP-CCBの曝露上昇。過度の降圧・浮腫に注意。血圧・腎機能モニタリング強化。",
                    "priority": 2  # 血圧異常リスク
                })
            
            # 鎮静薬多剤（注意）
            t = self.rule_poly_sedatives(bx)
            if t:
                findings.append({
                    "severity": "併用注意",
                    "title": "鎮静薬の多剤併用",
                    "targets": self.join_targets(t),
                    "action": "過鎮静・転倒リスク。必要性を再評価し、簡素化を検討。",
                    "priority": 1  # 転倒リスク（高齢者で特に重要）
                })
            
            # 下剤の多剤併用（注意）
            t = self.rule_poly_laxatives(bx)
            if t:
                findings.append({
                    "severity": "併用注意",
                    "title": "下剤の多剤併用",
                    "targets": self.join_targets(t),
                    "action": "下痢・脱水・電解質異常（特にK）に注意。便回数・体重・血圧のモニタリング。必要性を再評価し簡素化を検討。",
                    "priority": 2  # 電解質異常リスク
                })
            
            # 甘草 + 刺激性下剤（注意）
            t = self.rule_glycyrrhiza_stim_lax(bx)
            if t:
                findings.append({
                    "severity": "併用注意",
                    "title": "甘草（芍薬甘草湯）＋刺激性下剤",
                    "targets": self.join_targets(t),
                    "action": "低K血症（偽性アルドステロン症）や浮腫・高血圧のリスク。K値・血圧・浮腫を定期確認。",
                    "priority": 1  # 患者安全への影響が大きい
                })
            
            # Ca製剤による吸収低下（注意）
            t = self.rule_calcium_binder_caution(bx)
            if t:
                # キレート対象薬が含まれているかチェック
                chelate_sensitive = ["ニューキノロン", "テトラサイクリン", "甲状腺", "ビスホスホネート", "ミコフェノール", "鉄剤", "レボチロキシン", "シプロフロキサシン", "レボフロキサシン"]
                has_chelate_target = any(any(keyword in d.get('generic', d.get('name', d.get('raw', ''))).lower() for keyword in chelate_sensitive) for d in drugs)
                
                note = "" if has_chelate_target else "（該当薬なし：一般注意）"
                findings.append({
                    "severity": "併用注意",
                    "title": f"Ca製剤による他剤の吸収低下{note}",
                    "targets": self.join_targets(t),
                    "action": "キレート・吸着による他剤の吸収低下に注意。該当薬（ニューキノロン、テトラサイクリン、甲状腺薬、ビスホスホネート、ミコフェノール酸、鉄剤など）は2時間以上間隔をあける。",
                    "priority": 4  # 一般注意（該当薬なしの場合は影響小）
                })
            
            # カルシミメティクス + 下剤多剤（状況依存の注意）
            t = self.rule_calcimimetic_electrolyte(bx)
            if t:
                findings.append({
                    "severity": "併用注意",
                    "title": "エボカルセト＋下剤多剤（低Ca/QT延長リスク）",
                    "targets": self.join_targets(t),
                    "action": "Ca低下によりQT延長素因が増す可能性。Ca・Mg・Kを定期確認（特に開始・増量時）。",
                    "priority": 3  # 状況依存だが重要
                })
            
            # 優先度でソート（小さい数字=高優先度）
            findings.sort(key=lambda x: x.get('priority', 99))
            
            logger.info(f"Build report: {len(findings)} findings")
            return findings
            
        except Exception as e:
            logger.error(f"Report building failed: {e}")
            return []
    
    def resolve_targets(self, drugs: List[Dict[str, Any]]) -> Dict[str, str]:
        """薬剤リストから相互作用対象薬を特定（後方互換性のため）"""
        try:
            bx = self.index_by_class(drugs)
            targets = {}
            
            # RAAS禁忌（ARNI＋ACEI）
            raas_contraindicated = self.rule_raas_arni_plus_acei(bx)
            if raas_contraindicated:
                targets["raas_contraindicated"] = self.join_targets(raas_contraindicated)
            
            # RAAS重複（RAAS禁忌が出たら同一ペアを非表示）
            raas_overlap = self.rule_raas_overlap(bx)
            if raas_overlap and not raas_contraindicated:
                # RAAS重複の場合は特定の順序で表示
                if len(raas_overlap) == 3 and all(x in raas_overlap for x in ["エナラプリル", "サクビトリル/バルサルタン", "テルミサルタン/アムロジピン"]):
                    targets["raas_overlap"] = "エナラプリル、サクビトリル/バルサルタン、テルミサルタン/アムロジピン"
                else:
                    targets["raas_overlap"] = self.join_targets(raas_overlap)
            
            # PDE5＋硝酸薬相当
            pde5_nitrate = self.rule_pde5_nitrate(bx)
            if pde5_nitrate:
                targets["pde5_nitrate"] = self.join_targets(pde5_nitrate)
            
            # 胃酸抑制の重複
            acid_dup = self.rule_acid_dup(bx)
            if acid_dup:
                targets["acid_dup"] = self.join_targets(acid_dup)
            
            # 降圧薬の多剤併用
            poly_antihypertensive = self.rule_poly_antihypertensive(bx)
            if poly_antihypertensive:
                targets["poly_antihypertensive"] = self.join_targets(poly_antihypertensive)
            
            logger.info(f"Resolved targets: {targets}")
            return targets
            
        except Exception as e:
            logger.error(f"Target resolution failed: {e}")
            return {}
