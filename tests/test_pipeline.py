"""
回帰テストシステム
ゴールデンセットによる自動テスト
"""
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# テスト対象のモジュールをインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from services.block_parser import BlockParser
from services.consensus_extractor import ConsensusExtractor
from services.deduper import dedupe_with_form_agnostic, collapse_combos
from services.classifier_kegg import KeggClassifier
from services.interaction_engine import InteractionEngine
from services.quality_gate import QualityGate

logger = logging.getLogger(__name__)

class PipelineTester:
    """パイプラインテスター"""
    
    def __init__(self):
        self.block_parser = BlockParser()
        self.consensus_extractor = ConsensusExtractor()
        self.kegg_classifier = KeggClassifier()
        self.interaction_engine = InteractionEngine()
        self.quality_gate = QualityGate()
    
    def run_pipeline(self, text: str) -> Dict[str, Any]:
        """画像→本文→抽出→分類→判定 までを1関数化"""
        try:
            # 1. ブロック分割
            blocks = self.block_parser.parse_prescription_blocks(text)
            block_stats = self.block_parser.get_parsing_stats(blocks)
            
            # 2. 合議制抽出
            all_drugs = []
            for block in blocks:
                block_drugs = self.consensus_extractor.extract_drugs_consensus(
                    block['content']
                )
                all_drugs.extend(block_drugs)
            
            extraction_stats = self.consensus_extractor.get_extraction_stats(all_drugs)
            
            # 3. 重複統合
            unique_drugs = collapse_combos(all_drugs)
            unique_drugs, removed_count = dedupe_with_form_agnostic(unique_drugs)
            
            # 4. KEGG分類
            classified_drugs = self.kegg_classifier.classify_many(unique_drugs)
            classification_stats = self.kegg_classifier.get_classification_stats(classified_drugs)
            
            # 5. 相互作用チェック
            interaction_result = self.interaction_engine.check_drug_interactions(classified_drugs)
            
            # 6. 品質チェック
            quality_stats = {
                'coverage': block_stats.get('coverage', 0.0),
                'kegg_cover_rate': classification_stats.get('classification_rate', 0.0),
                'low_conf': len([d for d in classified_drugs if d.get('confidence', 0) < 0.8])
            }
            quality_result = self.quality_gate.check_quality(quality_stats, classified_drugs)
            
            return {
                'drugs': classified_drugs,
                'interactions': interaction_result,
                'quality': quality_result,
                'stats': {
                    'block_stats': block_stats,
                    'extraction_stats': extraction_stats,
                    'classification_stats': classification_stats,
                    'quality_stats': quality_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                'error': str(e),
                'drugs': [],
                'interactions': {'has_interactions': False},
                'quality': {'passed': False},
                'stats': {}
            }

def load_text(file_path: str) -> str:
    """テキストファイルを読み込み"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {e}")
        return ""

def test_golden_cases():
    """ゴールデンセットのテストを実行"""
    try:
        # テストケースを読み込み
        cases_file = Path(__file__).parent / "golden" / "cases.yaml"
        if not cases_file.exists():
            logger.error(f"Test cases file not found: {cases_file}")
            return False
        
        with open(cases_file, 'r', encoding='utf-8') as f:
            test_cases = yaml.safe_load(f)
        
        tester = PipelineTester()
        passed_tests = 0
        total_tests = len(test_cases)
        
        for case in test_cases:
            case_name = case.get('name', 'Unknown')
            logger.info(f"Running test case: {case_name}")
            
            try:
                # 入力ファイルを読み込み
                input_file = case['inputs']['ocr_text_file']
                text_file_path = Path(__file__).parent / "golden" / input_file
                
                if not text_file_path.exists():
                    logger.warning(f"Input file not found: {text_file_path}")
                    continue
                
                text = load_text(str(text_file_path))
                if not text:
                    logger.warning(f"Empty input text for case: {case_name}")
                    continue
                
                # パイプラインを実行
                result = tester.run_pipeline(text)
                
                if 'error' in result:
                    logger.error(f"Pipeline error for case {case_name}: {result['error']}")
                    continue
                
                # 期待値と比較
                expectations = case.get('expect', {})
                test_passed = True
                
                # 薬剤数チェック
                expected_count = expectations.get('unique_count')
                if expected_count is not None:
                    actual_count = len(result['drugs'])
                    if actual_count != expected_count:
                        logger.error(f"Drug count mismatch for {case_name}: expected {expected_count}, got {actual_count}")
                        test_passed = False
                
                # 必須薬剤チェック
                must_include = expectations.get('must_include', [])
                drug_names = {d.get('generic', d.get('raw', '')) for d in result['drugs']}
                
                for required_drug in must_include:
                    if not any(required_drug in name for name in drug_names):
                        logger.error(f"Missing required drug for {case_name}: {required_drug}")
                        test_passed = False
                
                # 除外薬剤チェック
                must_not_include = expectations.get('must_not_include', [])
                for excluded_drug in must_not_include:
                    if any(excluded_drug in name for name in drug_names):
                        logger.error(f"Unexpected drug found for {case_name}: {excluded_drug}")
                        test_passed = False
                
                # ルールヒットチェック
                expected_rules = expectations.get('rule_hits', [])
                actual_rule_ids = set()
                
                for rule in (result['interactions'].get('major_interactions', []) + 
                           result['interactions'].get('moderate_interactions', [])):
                    actual_rule_ids.add(rule.get('id', ''))
                
                for expected_rule in expected_rules:
                    if expected_rule not in actual_rule_ids:
                        logger.error(f"Missing expected rule for {case_name}: {expected_rule}")
                        test_passed = False
                
                # 品質要件チェック
                quality_reqs = expectations.get('quality_requirements', {})
                quality_stats = result['stats'].get('quality_stats', {})
                
                for req_key, req_value in quality_reqs.items():
                    actual_value = quality_stats.get(req_key, 0)
                    if actual_value < req_value:
                        logger.error(f"Quality requirement not met for {case_name}: {req_key} = {actual_value} < {req_value}")
                        test_passed = False
                
                if test_passed:
                    logger.info(f"✅ Test case passed: {case_name}")
                    passed_tests += 1
                else:
                    logger.error(f"❌ Test case failed: {case_name}")
                
            except Exception as e:
                logger.error(f"Test case execution failed for {case_name}: {e}")
        
        # 結果サマリー
        logger.info(f"Test Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed!")
            return True
        else:
            logger.error(f"❌ {total_tests - passed_tests} tests failed")
            return False
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # テスト実行
    success = test_golden_cases()
    exit(0 if success else 1)
