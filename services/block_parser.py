"""
処方箋ブロック分割パーサー
行ブロック分割の標準化と剤形語分割
"""
import re
import logging
from typing import List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class BlockParser:
    """処方箋ブロック分割パーサー"""
    
    def __init__(self):
        # 処方ブロックの正規表現パターン
        self.prescription_block_pattern = re.compile(
            r'^\s*(\d+)\s*【(?:般|外|麻|向)】\s*(.*?)(?=^\s*\d+\s*【(?:般|外|麻|向)】|\Z)',
            re.MULTILINE | re.DOTALL
        )
        
        # 剤形語のパターン
        self.form_patterns = [
            r'(錠|カプセル|口腔内崩壊錠|OD錠|腸溶錠|徐放錠|CR錠)',
            r'(顆粒|細粒|散|粉末)',
            r'(ゲル|軟膏|クリーム|ローション)',
            r'(貼付剤|テープ|パッチ)',
            r'(液|シロップ|エキス)',
            r'(注射|点滴|静注)',
            r'(吸入|スプレー|ネブライザー)',
            r'(坐剤|浣腸|点眼|点鼻)'
        ]
        
        # 剤形語の正規化マッピング
        self.form_normalization = {
            "口腔内崩壊錠": "口腔内崩壊",
            "OD錠": "口腔内崩壊",
            "腸溶錠": "腸溶",
            "徐放錠": "徐放",
            "CR錠": "徐放",
            "細粒": "顆粒",
            "粉末": "散",
            "ローション": "液",
            "貼付剤": "貼付",
            "パッチ": "貼付",
            "エキス": "液",
            "静注": "注射",
            "点滴": "注射",
            "スプレー": "吸入",
            "ネブライザー": "吸入",
            "浣腸": "坐剤",
            "点眼": "点眼",
            "点鼻": "点鼻"
        }
    
    def parse_prescription_blocks(self, text: str) -> List[dict[str, Any]]:
        """処方箋テキストをブロックに分割"""
        try:
            blocks = []
            matches = self.prescription_block_pattern.findall(text)
            
            for block_num, block_content in matches:
                # ブロック内の薬剤を抽出
                drugs = self._extract_drugs_from_block(block_content)
                
                if drugs:
                    blocks.append({
                        'block_number': int(block_num),
                        'content': block_content.strip(),
                        'drugs': drugs,
                        'drug_count': len(drugs)
                    })
            
            logger.info(f"Parsed {len(blocks)} prescription blocks")
            return blocks
            
        except Exception as e:
            logger.error(f"Block parsing error: {e}")
            return []
    
    def _extract_drugs_from_block(self, block_content: str) -> List[dict[str, Any]]:
        """ブロック内から薬剤を抽出（剤形語で分割）"""
        try:
            drugs = []
            lines = block_content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 剤形語で分割
                drug_candidates = self._split_by_form_patterns(line)
                
                for candidate in drug_candidates:
                    if self._is_valid_drug_candidate(candidate):
                        drug_info = self._parse_drug_info(candidate)
                        if drug_info:
                            drugs.append(drug_info)
            
            return drugs
            
        except Exception as e:
            logger.error(f"Drug extraction error: {e}")
            return []
    
    def _split_by_form_patterns(self, text: str) -> List[str]:
        """剤形語パターンでテキストを分割"""
        candidates = [text]  # 初期候補
        
        for pattern in self.form_patterns:
            new_candidates = []
            for candidate in candidates:
                # 剤形語の前後で分割
                parts = re.split(pattern, candidate)
                if len(parts) > 1:
                    # 剤形語を含む部分を再構築
                    for i in range(0, len(parts) - 1, 2):
                        if i + 1 < len(parts):
                            drug_part = parts[i] + parts[i + 1]
                            if drug_part.strip():
                                new_candidates.append(drug_part.strip())
                else:
                    new_candidates.append(candidate)
            candidates = new_candidates
        
        return candidates
    
    def _is_valid_drug_candidate(self, candidate: str) -> bool:
        """薬剤候補の妥当性チェック"""
        if not candidate or len(candidate) < 2:
            return False
        
        # ノイズパターンを除外
        noise_patterns = [
            r'^[0-9\s\-×]+$',  # 数字と記号のみ
            r'^[年月日時分秒]+$',  # 日時のみ
            r'^[患者|医師|薬剤師|保険]+$',  # 役職名のみ
            r'^[変更不可|患者希望|次頁に続く]+$',  # 固定文言
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, candidate):
                return False
        
        # 剤形語を含むかチェック
        has_form = any(re.search(pattern, candidate) for pattern in self.form_patterns)
        
        # 薬剤名らしい文字を含むかチェック
        has_drug_chars = bool(re.search(r'[ぁ-んァ-ヶ一-龥A-Za-z]', candidate))
        
        return has_form and has_drug_chars
    
    def _parse_drug_info(self, drug_text: str) -> dict[str, Any]:
        """薬剤テキストから情報を抽出"""
        try:
            # 剤形を抽出
            form = self._extract_form(drug_text)
            
            # 薬剤名を抽出（剤形を除く）
            drug_name = self._extract_drug_name(drug_text, form)
            
            # 用量を抽出
            strength = self._extract_strength(drug_text)
            
            # 投与量を抽出
            dose = self._extract_dose(drug_text)
            
            # 投与頻度を抽出
            frequency = self._extract_frequency(drug_text)
            
            return {
                'raw': drug_text,
                'drug_name': drug_name,
                'form': form,
                'strength': strength,
                'dose': dose,
                'frequency': frequency,
                'confidence': 0.9  # ブロック分割による高信頼度
            }
            
        except Exception as e:
            logger.error(f"Drug info parsing error: {e}")
            return None
    
    def _extract_form(self, text: str) -> str:
        """剤形を抽出"""
        for pattern in self.form_patterns:
            match = re.search(pattern, text)
            if match:
                form = match.group(1)
                return self.form_normalization.get(form, form)
        return "錠"  # デフォルト
    
    def _extract_drug_name(self, text: str, form: str) -> str:
        """薬剤名を抽出（剤形を除く）"""
        # 剤形を除去
        name = text
        for form_variant in [form] + list(self.form_normalization.keys()):
            name = name.replace(form_variant, "")
        
        # 用量表記を除去
        name = re.sub(r'\d+\.?\d*\s*(mg|g|ml|μg|mcg)', '', name)
        name = re.sub(r'\d+%', '', name)
        
        # 特殊文字を除去
        name = re.sub(r'[()（）【】]', '', name)
        
        return name.strip()
    
    def _extract_strength(self, text: str) -> str:
        """用量を抽出"""
        strength_patterns = [
            r'(\d+\.?\d*)\s*(mg|g|ml|μg|mcg)',
            r'(\d+\.?\d*)%'
        ]
        
        for pattern in strength_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return ""
    
    def _extract_dose(self, text: str) -> str:
        """投与量を抽出"""
        dose_patterns = [
            r'(\d+)\s*(錠|包|ml|g)',
            r'(\d+)\s*×\s*(\d+)',  # 例: 3×2
        ]
        
        for pattern in dose_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return ""
    
    def _extract_frequency(self, text: str) -> str:
        """投与頻度を抽出"""
        frequency_patterns = [
            r'(食前|食後|食間|就寝前|眠前|朝|昼|夕)',
            r'(\d+)\s*回',
            r'(毎日|隔日|週\d+回)'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return ""
    
    def get_parsing_stats(self, blocks: List[dict[str, Any]]) -> dict[str, Any]:
        """パース統計を取得"""
        if not blocks:
            return {
                'total_blocks': 0,
                'total_drugs': 0,
                'avg_drugs_per_block': 0,
                'coverage': 0.0
            }
        
        total_drugs = sum(block['drug_count'] for block in blocks)
        avg_drugs_per_block = total_drugs / len(blocks)
        
        return {
            'total_blocks': len(blocks),
            'total_drugs': total_drugs,
            'avg_drugs_per_block': avg_drugs_per_block,
            'coverage': min(1.0, total_drugs / 10.0)  # 期待値10剤に対するカバレッジ
        }
