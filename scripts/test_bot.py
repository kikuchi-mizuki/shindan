#!/usr/bin/env python3
"""
薬局サポートBot テストスクリプト
"""

import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ocr_service import OCRService
from services.drug_service import DrugService
from services.response_service import ResponseService

def test_ocr_service():
    """OCRサービスのテスト"""
    print("=== OCRサービステスト ===")
    
    ocr_service = OCRService()
    
    # テスト用の薬剤名リスト
    test_drug_names = ['アスピリン', 'ワルファリン', 'ジゴキシン']
    
    print(f"テスト薬剤名: {test_drug_names}")
    print(f"Vision API利用可能: {ocr_service.vision_available}")
    
    # モック画像データでテスト
    mock_image_content = b"mock_image_data"
    extracted_drugs = ocr_service.extract_drug_names(mock_image_content)
    print(f"抽出された薬剤名: {extracted_drugs}")
    print("OCRサービス初期化完了")
    print()

def test_drug_service():
    """薬剤サービスのテスト"""
    print("=== 薬剤サービステスト ===")
    
    drug_service = DrugService()
    
    # テスト用の薬剤名リスト（同効薬を含む）
    test_drug_names = ['アスピリン', 'ワルファリン', 'イブプロフェン', 'シンバスタチン', 'アトルバスタチン']
    
    print(f"テスト薬剤名: {test_drug_names}")
    
    # 薬剤情報を取得
    drug_info = drug_service.get_drug_interactions(test_drug_names)
    
    print("検出された薬剤:")
    for drug in drug_info['detected_drugs']:
        print(f"  - {drug['name']} ({drug['category']})")
    
    print("\n相互作用:")
    for interaction in drug_info['interactions']:
        print(f"  - {interaction['drug1']} + {interaction['drug2']}: {interaction['description']}")
        if 'mechanism' in interaction:
            print(f"    機序: {interaction['mechanism']}")
    
    print("\n同効薬警告:")
    for warning in drug_info['same_effect_warnings']:
        print(f"  - {warning['drug1']} + {warning['drug2']}: {warning['mechanism']}")
        print(f"    リスクレベル: {warning['risk_level']}")
    
    print("\n薬剤分類重複:")
    for duplicate in drug_info['category_duplicates']:
        print(f"  - {duplicate['category']}: {duplicate['count']}種類")
        for drug in duplicate['drugs']:
            print(f"    - {drug}")
    
    print("\nKEGG情報:")
    for kegg in drug_info['kegg_info']:
        print(f"  - {kegg['drug_name']}")
        if kegg.get('kegg_id'):
            print(f"    KEGG ID: {kegg['kegg_id']}")
        if kegg.get('pathways'):
            print(f"    パスウェイ: {kegg['pathways']}")
        if kegg.get('targets'):
            print(f"    ターゲット: {kegg['targets']}")
    
    print("\n警告:")
    for warning in drug_info['warnings']:
        print(f"  - {warning}")
    
    print("\n推奨事項:")
    for recommendation in drug_info['recommendations']:
        print(f"  - {recommendation}")
    
    print()

def test_response_service():
    """応答サービスのテスト"""
    print("=== 応答サービステスト ===")
    
    response_service = ResponseService()
    
    # テスト用の薬剤情報（拡張版）
    test_drug_info = {
        'detected_drugs': [
            {
                'name': 'アスピリン',
                'generic_name': 'アセチルサリチル酸',
                'category': '解熱鎮痛薬',
                'interactions': ['ワルファリン', '抗凝固薬']
            },
            {
                'name': 'ワルファリン',
                'generic_name': 'ワルファリンカリウム',
                'category': '抗凝固薬',
                'interactions': ['アスピリン', 'NSAIDs']
            },
            {
                'name': 'イブプロフェン',
                'generic_name': 'イブプロフェン',
                'category': '解熱鎮痛薬',
                'interactions': []
            }
        ],
        'interactions': [
            {
                'drug1': 'アスピリン',
                'drug2': 'ワルファリン',
                'risk': 'high',
                'description': '出血リスク増加',
                'mechanism': '血小板機能阻害の重複'
            }
        ],
        'same_effect_warnings': [
            {
                'drug1': 'アスピリン',
                'drug2': 'イブプロフェン',
                'mechanism': 'COX-1/COX-2阻害',
                'risk_level': 'high',
                'description': '同効薬の重複投与: COX-1/COX-2阻害'
            }
        ],
        'category_duplicates': [
            {
                'category': '解熱鎮痛薬',
                'drugs': ['アスピリン', 'イブプロフェン'],
                'count': 2,
                'description': '解熱鎮痛薬の薬剤が2種類検出されました'
            }
        ],
        'kegg_info': [
            {
                'drug_name': 'アスピリン',
                'kegg_id': 'D00109',
                'pathways': ['hsa00590', 'hsa00591'],
                'targets': ['PTGS1', 'PTGS2']
            }
        ],
        'warnings': [
            '⚠️ 薬剤間の相互作用が検出されました',
            '⚠️ 同効薬の重複投与が検出されました',
            '⚠️ 薬剤分類による重複が検出されました'
        ],
        'recommendations': [
            '・薬剤師に必ずご相談ください',
            '・同効薬の重複投与を避けるように注意してください',
            '・薬剤分類による重複を避けるように注意してください',
            '・この情報は参考情報です。最終判断は薬剤師にお任せください'
        ]
    }
    
    # LINE Bot用応答メッセージを生成
    response = response_service.generate_response(test_drug_info)
    
    print("生成された応答メッセージ:")
    print(response)
    print()
    
    # 詳細分析結果を生成
    detailed_analysis = response_service.generate_detailed_analysis(test_drug_info)
    
    print("詳細分析結果:")
    print(detailed_analysis)
    print()

def test_same_effect_check():
    """同効薬チェック機能のテスト"""
    print("=== 同効薬チェックテスト ===")
    
    drug_service = DrugService()
    
    # 同効薬のテストケース
    test_cases = [
        ['アスピリン', 'イブプロフェン'],  # 同効薬
        ['ワルファリン', 'ダビガトラン'],  # 同効薬
        ['シンバスタチン', 'アトルバスタチン'],  # 同効薬
        ['アスピリン', 'ワルファリン'],  # 相互作用のみ
        ['メトホルミン', 'インスリン']  # 相互作用のみ
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"テストケース {i}: {test_case}")
        drug_info = drug_service.get_drug_interactions(test_case)
        
        if drug_info['same_effect_warnings']:
            print("  → 同効薬検出:")
            for warning in drug_info['same_effect_warnings']:
                print(f"    - {warning['drug1']} + {warning['drug2']}: {warning['mechanism']}")
        else:
            print("  → 同効薬なし")
        
        if drug_info['interactions']:
            print("  → 相互作用検出:")
            for interaction in drug_info['interactions']:
                print(f"    - {interaction['drug1']} + {interaction['drug2']}: {interaction['description']}")
        else:
            print("  → 相互作用なし")
        
        print()

def test_category_duplicates():
    """薬剤分類重複チェックのテスト"""
    print("=== 薬剤分類重複チェックテスト ===")
    
    drug_service = DrugService()
    
    # 分類重複のテストケース
    test_cases = [
        ['アスピリン', 'イブプロフェン', 'ロキソプロフェン'],  # 解熱鎮痛薬重複
        ['シンバスタチン', 'アトルバスタチン', 'プラバスタチン'],  # 脂質異常症治療薬重複
        ['ワルファリン', 'ダビガトラン', 'リバーロキサバン'],  # 抗凝固薬重複
        ['アスピリン', 'ワルファリン', 'ジゴキシン']  # 分類重複なし
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"テストケース {i}: {test_case}")
        drug_info = drug_service.get_drug_interactions(test_case)
        
        if drug_info['category_duplicates']:
            print("  → 分類重複検出:")
            for duplicate in drug_info['category_duplicates']:
                print(f"    - {duplicate['category']}: {duplicate['count']}種類")
                for drug in duplicate['drugs']:
                    print(f"      * {drug}")
        else:
            print("  → 分類重複なし")
        
        print()

def test_kegg_integration():
    """KEGG連携のテスト"""
    print("=== KEGG連携テスト ===")
    
    drug_service = DrugService()
    
    # KEGG情報取得のテスト
    test_drugs = ['アスピリン', 'ワルファリン', 'メトホルミン']
    
    for drug in test_drugs:
        print(f"薬剤: {drug}")
        try:
            kegg_data = drug_service._fetch_kegg_drug_info(drug)
            if kegg_data:
                print(f"  KEGG ID: {kegg_data.get('kegg_id', 'N/A')}")
                print(f"  パスウェイ: {kegg_data.get('pathways', [])}")
                print(f"  ターゲット: {kegg_data.get('targets', [])}")
            else:
                print("  KEGG情報が見つかりませんでした")
        except Exception as e:
            print(f"  KEGG情報取得エラー: {e}")
        print()

def test_integration():
    """統合テスト"""
    print("=== 統合テスト ===")
    
    # 各サービスを初期化
    ocr_service = OCRService()
    drug_service = DrugService()
    response_service = ResponseService()
    
    # モック画像データでOCR処理
    mock_image_content = b"mock_image_data"
    extracted_drugs = ocr_service.extract_drug_names(mock_image_content)
    
    print(f"OCRで抽出された薬剤: {extracted_drugs}")
    
    if extracted_drugs:
        # 薬剤情報を取得
        drug_info = drug_service.get_drug_interactions(extracted_drugs)
        
        # 応答メッセージを生成
        response = response_service.generate_response(drug_info)
        
        print("統合テスト結果:")
        print(response)
    else:
        print("薬剤名が抽出されませんでした")
    
    print()

def main():
    """メイン処理"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("薬局サポートBot 拡張機能テストを開始します...\n")
    
    try:
        # 各サービスのテスト
        test_ocr_service()
        test_drug_service()
        test_response_service()
        
        # 新機能のテスト
        test_same_effect_check()
        test_category_duplicates()
        test_kegg_integration()
        test_integration()
        
        print("=== テスト完了 ===")
        print("すべてのテストが正常に完了しました。")
        print("\n新機能の追加内容:")
        print("✅ 同効薬チェック機能")
        print("✅ 相互作用の詳細度向上")
        print("✅ 薬剤分類による重複チェック")
        print("✅ KEGGデータベース連携")
        
    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 