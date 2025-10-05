"""
OCR結果の番号ブロック分割と薬剤名抽出ユーティリティ
"""
import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def split_numbered_blocks(text: str) -> List[Tuple[str, str]]:
    """
    テキストを番号ブロックで分割
    
    Args:
        text: OCR結果のテキスト
        
    Returns:
        List[Tuple[str, str]]: [(番号, ブロックテキスト), ...]
    """
    try:
        # 例) 「7 ...\n...\n8 ...\n...」を [("7",block7),("8",block8),...] に
        # より柔軟なパターンで番号ブロックを検出
        pat = re.compile(r"(?m)^\s*(\d{1,2})\s*[【】]?\s*(.*?)(?=^\s*\d{1,2}\s*[【】]?\s*|\Z)", re.S)
        blocks = pat.findall(text)
        
        logger.info(f"Split text into {len(blocks)} numbered blocks")
        for i, (no, blk) in enumerate(blocks):
            logger.debug(f"Block {i+1}: No.{no} - {blk[:100]}...")
        
        return blocks
        
    except Exception as e:
        logger.error(f"Error splitting numbered blocks: {e}")
        return []

# 薬剤名抽出パターン（改良版）
NAME_PAT = re.compile(
    r"(?:ツムラ)?([ぁ-んァ-ヶ一-龥A-Za-z0-9・ー\-]+?)(?:錠|カプセル|口腔内崩壊錠|顆粒|ゲル|散|液|エキス顆粒|テープ|点眼液|点鼻液|吸入液|注射剤|注射液|錠剤|カプセル剤|顆粒剤|ゲル剤|散剤|液剤)"
)

def extract_names_from_block(block_text: str) -> List[str]:
    """
    ブロックから薬剤名候補を抽出（1ブロックに複数剤OK）
    
    Args:
        block_text: 番号ブロックのテキスト
        
    Returns:
        List[str]: 抽出された薬剤名のリスト
    """
    try:
        # ノイズ除去
        t = block_text.replace("（医療用）", "").replace("次頁に続く", "").replace("医療用", "")
        
        # 薬剤名パターンで抽出
        matches = NAME_PAT.finditer(t)
        names = []
        
        for match in matches:
            name = match.group(1).strip()
            if name and len(name) >= 2:  # 最低2文字以上
                # 数字で始まる場合は除去（例：7オルケディア → オルケディア）
                if name[0].isdigit():
                    name = name[1:]
                if name and len(name) >= 2:
                    names.append(name)
        
        # 追加のパターンマッチング（剤形が明示されていない場合）
        additional_patterns = [
            r"([ぁ-んァ-ヶ一-龥A-Za-z0-9・ー]+?)(?=\s*\d+mg|\s*\d+μg|\s*\d+g|\s*\d+%)",
            r"([ぁ-んァ-ヶ一-龥A-Za-z0-9・ー]+?)(?=\s*×|\s*食後|\s*食前|\s*眠前)"
        ]
        
        for pattern in additional_patterns:
            additional_matches = re.finditer(pattern, t)
            for match in additional_matches:
                name = match.group(1).strip()
                if name and len(name) >= 3 and name not in names:  # 重複を避ける
                    # 数字で始まる場合は除去
                    if name[0].isdigit():
                        name = name[1:]
                    if name and len(name) >= 3:
                        names.append(name)

        # 外用剤のサイズ/枚数を抽出（情報付加用途）
        size_match = re.search(r"(\d+)\s*cm\s*×\s*(\d+)\s*cm", t)
        qty_match = re.search(r"(\d+)\s*枚", t)
        days_match = re.search(r"(\d+)\s*日分", t)
        if size_match:
            logger.info(f"Detected external size: {size_match.group(0)}")
        if qty_match:
            logger.info(f"Detected external quantity: {qty_match.group(0)}")
        if days_match:
            logger.info(f"Detected days: {days_match.group(0)}")
        
        # 重複除去（順序保持）
        unique_names = list(dict.fromkeys(names))
        
        logger.debug(f"Extracted {len(unique_names)} names from block: {unique_names}")
        return unique_names
        
    except Exception as e:
        logger.error(f"Error extracting names from block: {e}")
        return []

def extract_drug_names_from_text(text: str) -> List[str]:
    """
    テキスト全体から薬剤名を抽出（番号ブロック分割方式）
    
    Args:
        text: OCR結果のテキスト
        
    Returns:
        List[str]: 抽出された薬剤名のリスト
    """
    try:
        # 番号ブロックに分割
        blocks = split_numbered_blocks(text)
        
        all_names = []
        for no, block_text in blocks:
            names = extract_names_from_block(block_text)
            all_names.extend(names)
            logger.info(f"Block {no}: extracted {len(names)} names")
        
        # 重複除去（順序保持）
        unique_names = list(dict.fromkeys(all_names))
        
        logger.info(f"Total extracted {len(unique_names)} unique drug names: {unique_names}")
        return unique_names
        
    except Exception as e:
        logger.error(f"Error extracting drug names from text: {e}")
        return []

def post_filter_drugs(names: List[str], text: str) -> List[str]:
    """
    薬剤名の後処理フィルター
    
    Args:
        names: 薬剤名のリスト
        text: 元のテキスト
        
    Returns:
        List[str]: フィルタリング後の薬剤名リスト
    """
    try:
        keep = []
        for name in names:
            # ジクロフェナクは外用のみ採用
            if "ジクロフェナク" in name and not any(k in text for k in ["ゲル", "外用", "貼付"]):
                logger.info(f"Filtered out diclofenac (no dosage form): {name}")
                continue
            keep.append(name)
        
        logger.info(f"Post-filter: {len(names)} -> {len(keep)} drugs")
        return keep
        
    except Exception as e:
        logger.error(f"Error in post-filter: {e}")
        return names
