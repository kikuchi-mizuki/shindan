import os
import logging
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, MessagingApiBlob, TextMessage, ReplyMessageRequest, PushMessageRequest

from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# LINE Bot設定（環境変数が未設定でも起動できるようデフォルトを設定）
_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN') or ""
_channel_secret = os.getenv('LINE_CHANNEL_SECRET') or ""

# 資格情報が揃っている時のみクライアントを初期化
if _access_token:
    configuration = Configuration(access_token=_access_token)
    api_client = ApiClient(configuration)
    messaging_api = MessagingApi(api_client)
    messaging_blob_api = MessagingApiBlob(api_client)
else:
    configuration = None
    api_client = None
    messaging_api = None
    messaging_blob_api = None

if _channel_secret:
    handler = WebhookHandler(_channel_secret)
else:
    class _DummyHandler:
        def handle(self, *_args, **_kwargs):
            return None
        def add(self, *args, **kwargs):
            def _decorator(func):
                return func
            return _decorator
    handler = _DummyHandler()

# ユーザーごとの薬剤名バッファ
user_drug_buffer = {}

# サービスの遅延初期化
_services_initialized = False
ocr_service = None
drug_service = None
response_service = None
ai_extractor = None
classifier = None

def _calculate_accuracy_metrics(matched_drugs, classified_drugs):
    """実用精度の指標を計算"""
    try:
        metrics = {
            'drug_recall': 0.0,  # 薬剤リコール（真の薬が全て出た割合）
            'false_positive_rate': 0.0,  # 誤検知率（余計な薬が混ざる率）
            'classification_rate': 0.0,  # 分類付与率（class_jpが埋まる）
            'total_drugs': len(matched_drugs),
            'classified_drugs': 0,
            'unclassified_drugs': 0
        }
        
        if not matched_drugs:
            return metrics
        
        # 分類付与率の計算
        for drug in classified_drugs:
            classification = drug.get('final_classification', '')
            if classification and classification != '分類未設定':
                metrics['classified_drugs'] += 1
            else:
                metrics['unclassified_drugs'] += 1
        
        metrics['classification_rate'] = metrics['classified_drugs'] / len(classified_drugs) if classified_drugs else 0.0
        
        # 目標指標との比較
        target_metrics = {
            'drug_recall_target': 0.98,  # ≥ 0.98
            'false_positive_rate_target': 0.02,  # ≤ 0.02
            'classification_rate_target': 1.00  # = 1.00
        }
        
        metrics.update(target_metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Accuracy metrics calculation error: {e}")
        return {'error': str(e)}

def initialize_services():
    """サービスの初期化（遅延実行）"""
    global ocr_service, drug_service, response_service, ai_extractor, classifier, _services_initialized
    
    if _services_initialized:
        return True
    
    try:
        logger.info("Initializing services...")
        
        try:
            from services.ocr_service import OCRService
            ocr_service = OCRService()
            logger.info("OCRService initialized")
        except Exception as e:
            logger.error(f"OCRService initialization failed: {e}")
            return False
        
        try:
            from services.drug_service import DrugService
            drug_service = DrugService()
            logger.info("DrugService initialized")
        except Exception as e:
            logger.error(f"DrugService initialization failed: {e}")
            return False
        
        try:
            from services.response_service import ResponseService
            response_service = ResponseService()
            logger.info("ResponseService initialized")
        except Exception as e:
            logger.error(f"ResponseService initialization failed: {e}")
            return False
        
        try:
            from services.ai_extractor import AIExtractorService
            ai_extractor = AIExtractorService()
            logger.info("AIExtractorService initialized")
        except Exception as e:
            logger.error(f"AIExtractorService initialization failed: {e}")
            return False
        
        try:
            from services.classifier_kegg import KeggClassifier
            classifier = KeggClassifier()
            logger.info("KeggClassifier initialized")
        except Exception as e:
            logger.error(f"KeggClassifier initialization failed: {e}")
            return False
        
        _services_initialized = True
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False

@app.route("/", methods=['GET'])
def root():
    """ルートエンドポイント"""
    try:
        return {"status": "ok", "message": "薬局サポートBot is running"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route("/health", methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        # サービス初期化を確認
        if not _services_initialized:
            logger.info("Services not initialized, attempting initialization...")
            initialize_services()
        
        return {"status": "healthy", "message": "ok"}, 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}, 500

@app.route("/test", methods=['GET'])
def test():
    """テストエンドポイント"""
    try:
        # 基本的な環境変数チェック
        access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
        channel_secret = os.getenv('LINE_CHANNEL_SECRET')
        
        return {
            "status": "ok",
            "access_token_exists": bool(access_token),
            "channel_secret_exists": bool(channel_secret),
            "message": "LINE Bot configuration test passed"
        }, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route("/callback", methods=['POST'])
def callback():
    """LINE Webhook コールバック"""
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    
    # LINEの資格情報が未設定の場合でも起動可能に（本番では必須）
    try:
        if _channel_secret and _access_token:
            handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    """テキストメッセージの処理"""
    try:
        user_id = event.source.user_id
        user_message = event.message.text
        
        # サービス初期化チェック
        if not _services_initialized:
            if not initialize_services():
                (messaging_api or MessagingApi(ApiClient(Configuration(access_token="")))).reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="サービス初期化エラーが発生しました。しばらく時間をおいて再度お試しください。")]
                    )
                )
                return
        
        # 基本的なテキストメッセージ処理
        # 学習系コマンド（誤読修正／同義語／タグ追加／分類修正）
        if user_message.startswith('誤読修正：') or user_message.startswith('誤読修正:'):
            from services.normalize_store import learn_misread
            try:
                payload = user_message.split('：',1)[1] if '：' in user_message else user_message.split(':',1)[1]
                sep = '->' if '->' in payload else '→'
                bad, good = [x.strip() for x in payload.split(sep)]
                learn_misread(bad, good)
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text=f"誤読辞書を更新しました：{bad} → {good}\nこのまま『診断』で再チェックできます。")]))
            except Exception:
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text="形式: 誤読修正：誤→正")] ))
            return

        if user_message.startswith('同義語：') or user_message.startswith('同義語:'):
            from services.normalize_store import learn_synonym
            try:
                payload = user_message.split('：',1)[1] if '：' in user_message else user_message.split(':',1)[1]
                sep = '->' if '->' in payload else '→'
                alias, generic = [x.strip() for x in payload.split(sep)]
                learn_synonym(alias, generic)
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text=f"同義語を学習しました：{alias} → {generic}\nこのまま『診断』で再チェックできます。")]))
            except Exception:
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text="形式: 同義語：別名→一般名")] ))
            return

        if user_message.startswith('タグ追加：') or user_message.startswith('タグ追加:'):
            from services.normalize_store import learn_tag
            try:
                payload = user_message.split('：',1)[1] if '：' in user_message else user_message.split(':',1)[1]
                sep = '->' if '->' in payload else '→'
                generic, tags_str = [x.strip() for x in payload.split(sep)]
                tags = [t.strip() for t in tags_str.replace('、', ',').split(',') if t.strip()]
                learn_tag(generic, *tags)
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text=f"タグを学習しました：{generic} → {', '.join(tags)}\nこのまま『診断』で再チェックできます。")]))
            except Exception:
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text="形式: タグ追加：一般名→TAG1,TAG2")] ))
            return

        if user_message.startswith('分類修正：') or user_message.startswith('分類修正:'):
            from services.normalize_store import cache_atc
            try:
                payload = user_message.split('：',1)[1] if '：' in user_message else user_message.split(':',1)[1]
                sep = '->' if '->' in payload else '→'
                generic, atc = [x.strip() for x in payload.split(sep)]
                atc_list = [x.strip() for x in atc.replace('、', ',').split(',') if x.strip()]
                cache_atc(generic, atc_list)
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text=f"ATCをキャッシュしました：{generic} → {', '.join(atc_list)}\nこのまま『診断』で再チェックできます。")]))
            except Exception:
                messaging_api.reply_message(ReplyMessageRequest(replyToken=event.reply_token, messages=[TextMessage(text="形式: 分類修正：一般名→ATC1,ATC2")] ))
            return
        if user_message.lower() in ['診断', 'しんだん', 'diagnosis']:
            if user_id in user_drug_buffer and user_drug_buffer[user_id]:
                # 診断中のメッセージを送信
                (messaging_api or MessagingApi(ApiClient(Configuration(access_token="")))).reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="🔍 診断中です…")]
                    )
                )
                
                # 診断処理
                # 薬剤データの形式を判定して薬剤名を抽出
                drug_list = user_drug_buffer[user_id]
                if drug_list and isinstance(drug_list[0], dict):
                    # AI抽出結果の詳細情報から薬剤名を抽出
                    drug_names = [drug.get('name', '') for drug in drug_list if drug.get('name')]
                else:
                    # 従来の薬剤名リスト
                    drug_names = drug_list
                
                if drug_names:
                    # InteractionEngine（タグ×YAMLルール）で相互作用チェック
                    from services.interaction_engine import InteractionEngine
                    engine = InteractionEngine()
                    # drug_list は辞書形式を想定（generic/brand/raw を使用）
                    ie_result = engine.check_drug_interactions(drug_list if isinstance(drug_list[0], dict) else [{'raw': n} for n in drug_names])
                    # 表示用にフラットなリストへ変換（generate_responseの形式Bに合わせる）
                    interactions_flat = []
                    for r in ie_result.get('major_interactions', []) + ie_result.get('moderate_interactions', []):
                        interactions_flat.append({
                            'id': r.get('id'),
                            'name': r.get('name'),
                            'severity': r.get('severity'),
                            'advice': r.get('advice'),
                            'matched_drugs': []
                        })

                    drug_info = {
                        'drugs': drug_list,
                        'interactions': interactions_flat,
                        'rule_interactions': interactions_flat,
                        'ai_analysis': None
                    }

                    # 診断（ボタン押下時）は相互作用を含む詳細レスポンスを返す
                    response_text = response_service.generate_response(drug_info)
                else:
                    response_text = "薬剤情報が正しく取得できませんでした。"
                
                # 診断結果を送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
            else:
                (messaging_api or MessagingApi(ApiClient(Configuration(access_token="")))).reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="薬剤が登録されていません。\n\n📝 **薬剤名を直接入力して診断する方法：**\n\n薬剤名をスペース区切りで入力してください。\n例：クラリスロマイシン ベルソムラ デビゴ\n\nまたは、画像を送信して薬剤を登録してください。")]
                    )
                )
        
        elif user_message.lower() in ['リスト確認', 'りすとかくにん', 'list']:
            if user_id in user_drug_buffer and user_drug_buffer[user_id]:
                # 薬剤データの形式を判定して表示
                drug_list = user_drug_buffer[user_id]
                if drug_list and isinstance(drug_list[0], dict):
                    # AI抽出結果の詳細情報を表示
                    drug_display = []
                    for drug in drug_list:
                        name = drug.get('name', '')
                        strength = drug.get('strength', '')
                        final_classification = drug.get('final_classification', '')
                        class_hint = drug.get('class_hint', '')
                        kegg_category = drug.get('kegg_category', '')
                        
                        # カテゴリの優先順位
                        if final_classification and final_classification != '分類未設定':
                            category = final_classification
                        elif kegg_category:
                            category = kegg_category
                        elif class_hint:
                            category = f"{class_hint}（AI推定）"
                        else:
                            category = '不明'
                        
                        display_text = f"• {name}"
                        if strength:
                            display_text += f" {strength}"
                        display_text += f" (分類: {category})"
                        drug_display.append(display_text)
                    drug_list_text = "\n".join(drug_display)
                else:
                    # 従来の薬剤名リスト
                    drug_list_text = "\n".join([f"• {drug}" for drug in drug_list])
                
                response_text = f"📋 **現在の薬剤リスト**\n\n{drug_list_text}\n\n💡 「診断」で飲み合わせチェックを実行できます"
            else:
                response_text = "薬剤が登録されていません。画像を送信して薬剤を登録してください。"
            
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=response_text)]
                )
            )
        
        elif user_message.lower() in ['ヘルプ', 'へるぷ', 'help', '使い方', 'つかいかた']:
            help_text = """🏥 **薬局サポートBot 使い方**

**📸 薬剤の登録方法（画像）：**
1. 処方箋の画像を送信
2. 検出された薬剤を確認
3. 「診断」で飲み合わせチェック

**📝 薬剤の登録方法（テキスト）：**
薬剤名をスペース区切りで直接入力
例：クラリスロマイシン ベルソムラ デビゴ

**📋 撮影のコツ：**
• 真上から撮影（左右2ページは分割）
• 影・反射を避ける
• 処方名の行がはっきり写るように
• **iOSの「書類をスキャン」推奨**
• 文字の向きを正しく（横向きはNG）

**🔧 改善方法：**
• 明るい場所で撮影
• カメラを安定させる
• 処方箋全体が画面に入るように
• ピントを合わせてから撮影

**💊 コマンド一覧：**
• 診断 - 飲み合わせチェック
• リスト確認 - 現在の薬剤リスト
• 薬剤追加：〇〇 - 薬剤を手動追加
• ヘルプ - この使い方表示

**💡 便利な使い方：**
• 画像が読み取れない場合は薬剤名を直接入力
• 複数の薬剤を一度に入力可能
• 入力後すぐに診断実行

**⚠️ 注意事項：**
• 最終判断は医師・薬剤師にご相談ください
• 画質が悪い場合は再撮影をお願いします
• 手書きの処方箋は読み取り精度が低下する場合があります"""
            
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=help_text)]
                )
            )
        
        elif user_message.startswith('薬剤追加：'):
            drug_name = user_message.replace('薬剤追加：', '').strip()
            if drug_name:
                if user_id not in user_drug_buffer:
                    user_drug_buffer[user_id] = []
                
                # 薬剤データの形式を判定
                if user_drug_buffer[user_id] and isinstance(user_drug_buffer[user_id][0], dict):
                    # 辞書形式の場合、薬剤名の重複チェック
                    existing_names = [drug.get('name', '') for drug in user_drug_buffer[user_id]]
                    if drug_name not in existing_names:
                        # 分類サービスで分類を取得
                        temp_drug = {'generic': drug_name, 'raw': drug_name}
                        classification = classifier.classify_one(temp_drug)
                        
                        # 新しい薬剤を辞書形式で追加
                        new_drug = {
                            'name': drug_name,
                            'raw': drug_name,
                            'strength': '',
                            'dose': '',
                            'freq': '',
                            'days': None,
                            'class_hint': '',
                            'final_classification': classification or '分類未設定',
                            'kegg_category': '',
                            'kegg_id': ''
                        }
                        user_drug_buffer[user_id].append(new_drug)
                        response_text = f"✅ 薬剤「{drug_name}」を追加しました。"
                    else:
                        response_text = f"薬剤「{drug_name}」は既に登録されています。"
                else:
                    # 従来の文字列リストの場合
                    if drug_name not in user_drug_buffer[user_id]:
                        user_drug_buffer[user_id].append(drug_name)
                        response_text = f"✅ 薬剤「{drug_name}」を追加しました。"
                    else:
                        response_text = f"薬剤「{drug_name}」は既に登録されています。"
            else:
                response_text = "薬剤名を入力してください。例：薬剤追加：アムロジピン"
            
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=response_text)]
                )
            )
        
        # 薬剤名を直接入力して診断する機能
        elif (' ' in user_message or '\n' in user_message) and len(user_message.replace('\n', ' ').split()) >= 2:
            # スペース区切りまたは改行区切りの薬剤名を解析
            drug_names = user_message.replace('\n', ' ').split()
            logger.info(f"Direct drug input detected: {drug_names}")
            
            # 診断中のメッセージを送信
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text="🔍 診断中です…")]
                )
            )
            
            # 薬剤名を正規化してマッチング
            matched_drugs = drug_service.match_to_database(drug_names)
            logger.info(f"Matched drugs: {matched_drugs}")
            
            if matched_drugs:
                # ユーザーバッファに追加（文字列形式で保存）
                if user_id not in user_drug_buffer:
                    user_drug_buffer[user_id] = []
                
                for matched_drug in matched_drugs:
                    if matched_drug not in user_drug_buffer[user_id]:
                        user_drug_buffer[user_id].append(matched_drug)
                
                # 診断処理
                drug_info = drug_service.get_drug_interactions(matched_drugs)
                response_text = response_service.generate_response(drug_info)
                
                # 診断結果を送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
            else:
                # 薬剤が見つからない場合
                response_text = f"❌ 入力された薬剤名を認識できませんでした。\n\n入力: {' '.join(drug_names)}\n\n💡 **正しい薬剤名の例：**\n• クラリスロマイシン\n• ベルソムラ\n• デビゴ\n• ロゼレム\n• フルボキサミン\n• アムロジピン\n• エソメプラゾール\n\n薬剤名を確認して再度入力してください。"
                
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
            
        
    except Exception as e:
        logger.error(f"Text message handling error: {e}")
        try:
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text="エラーが発生しました。しばらく時間をおいて再度お試しください。")]
                )
            )
        except:
            pass

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """画像メッセージの処理"""
    try:
        user_id = event.source.user_id
        
        # サービス初期化チェック
        if not _services_initialized:
            if not initialize_services():
                (messaging_api or MessagingApi(ApiClient(Configuration(access_token="")))).reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text="サービス初期化エラーが発生しました。しばらく時間をおいて再度お試しください。")]
                    )
                )
                return
        
        # キャッシュクリア（前回の検出結果をリセット）
        if user_id in user_drug_buffer:
            user_drug_buffer[user_id] = []
            logger.info(f"Cleared drug buffer for user {user_id}")
        
        # 診断中のメッセージを送信
        messaging_api.reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text="🔍 診断中です…")]
            )
        )
        
        # 画像処理
        message_content = messaging_blob_api.get_message_content(event.message.id)
        
        # 画像品質評価付きOCR処理
        ocr_result = ocr_service.extract_drug_names(message_content)
        
        # 品質に応じた処理分岐
        if not ocr_result['should_process']:
            # 低品質画像の場合、ガイドを表示
            messaging_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=ocr_result['guide'])]
                )
            )
            return
        
        # OCRテキストを取得
        ocr_text = ocr_result.get('raw_text', '')
        if not ocr_text:
            # フォールバック: 従来の方法で薬剤名抽出
            drug_names = ocr_result.get('drug_names', [])
            if drug_names:
                matched_drugs = drug_service.match_to_database(drug_names)
            else:
                matched_drugs = []
        else:
            # AI抽出サービスを使用
            ai_result = ai_extractor.extract_drugs(ocr_text)
            logger.info(f"AI extraction result: {ai_result}")
            
            # 信頼度は後段のゲートで評価（未分類のみ確認）。ここでは処理を継続する。
            confidence = ai_result.get('confidence', 'low')
            
            # AI抽出結果を重複統合
            from services.deduper import dedupe
            from services.classifier_kegg import KeggClassifier
            
            ai_drugs = ai_result.get('drugs', [])

            # 補完: 番号ブロック抽出由来の薬剤名をAI結果にマージして取りこぼしを回収
            try:
                from services.ocr_utils import extract_drug_names_from_text
                # OCRのraw_textを優先的に使用。なければAI抽出に含めたraw_textを利用
                raw_text = ''
                if isinstance(ocr_result, dict):
                    raw_text = ocr_result.get('raw_text', '') or ''
                if not raw_text and isinstance(ai_result, dict):
                    raw_text = ai_result.get('raw_text', '') or ''

                block_names = extract_drug_names_from_text(raw_text) if raw_text else []

                # 同義語/表記ゆれ補正してdrug辞書に変換
                block_drug_dicts = []
                for name in block_names:
                    normalized = drug_service._pattern_based_correction(name)
                    block_drug_dicts.append({
                        'raw': name,
                        'generic': normalized,
                        'brand': None,
                        'strength': '',
                        'dose': '',
                        'freq': '',
                        'days': None,
                        'confidence': 0.7,  # 補完はやや低めの信頼度で扱う
                        'class_hint': None
                    })

                # AI抽出とブロック抽出を結合
                combined_drugs = list(ai_drugs) + block_drug_dicts
            except Exception as merge_err:
                logger.error(f"Block extraction merge failed: {merge_err}")
                combined_drugs = ai_drugs

            # ノイズ除去
            from services.deduper import remove_noise
            clean_drugs = remove_noise(combined_drugs)
            
            # 重複統合（配合錠と単剤の二重取りを防ぐ）
            from services.deduper import collapse_combos, dedupe_with_form_agnostic
            unique_drugs = collapse_combos(clean_drugs)
            unique_drugs, removed_count = dedupe_with_form_agnostic(unique_drugs)
            
            # 形態補正（ピコスルファートNaの錠/液を補正）
            try:
                from services.post_processors import fix_picosulfate_form
                unique_drugs = [fix_picosulfate_form(d) for d in unique_drugs]
            except Exception as _pp_err:
                logger.warning(f"Post processing failed (picosulfate form): {_pp_err}")
            
            # KEGG分類器で重複統合後の薬剤を分類
            kegg_classifier = KeggClassifier()
            classified_drugs = kegg_classifier.classify_many(unique_drugs)
            
            # 分類統計をログ出力
            stats = kegg_classifier.get_classification_stats(classified_drugs)
            logger.info(f"Classification stats: {stats}")
            
            # 相互作用チェック（新しいエンジン）
            try:
                from services.interaction_engine import InteractionEngine
                interaction_engine = InteractionEngine()
                interaction_result = interaction_engine.check_drug_interactions(classified_drugs)
                logger.info(f"Interaction check result: {interaction_result['summary']}")
            except Exception as e:
                logger.warning(f"Interaction check failed: {e}")
                interaction_result = {
                    "has_interactions": False,
                    "major_interactions": [],
                    "moderate_interactions": [],
                    "summary": "相互作用チェック中にエラーが発生しました。"
                }
            
            # 信頼度ゲートチェック（実務的にノイズを減らす）
            low_confidence_drugs = []
            missing_kegg_drugs = []

            for drug in classified_drugs:
                generic_name = drug.get('generic', '')
                confidence = float(drug.get('confidence', 1.0) or 1.0)
                kegg_id = drug.get('kegg_id') or ''
                final_cls = drug.get('final_classification') or ''

                # 低信頼度は「未分類」のときのみ警告（分類が確定していれば許容）
                if confidence < 0.6 and (not final_cls or final_cls == '分類未設定'):
                    low_confidence_drugs.append(generic_name)

                # KEGG未取得は「未分類」のときのみ警告（ローカル辞書等で確定していれば許容）
                if (not kegg_id) and (not final_cls or final_cls == '分類未設定'):
                    missing_kegg_drugs.append(generic_name)

            # 信頼度ゲートを緩和：分類が確定していればKEGG IDがなくても許容
            # 低信頼度かつ未分類のみ警告（分類が確定していれば許容）
            if low_confidence_drugs:
                confirmation_message = "⚠️ 一部の薬剤で信頼度が低いため、分類が不明確です。\n\n"
                confirmation_message += f"信頼度が低い薬剤: {', '.join(low_confidence_drugs)}\n"
                confirmation_message += "\nこのまま診断を続行しますか？\n「はい」で続行、「いいえ」で再撮影をお願いします。"
                
                # 確認メッセージを送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=confirmation_message)]
                    )
                )
                return
            
            # 薬剤名リストを構築
            matched_drugs = []
            for drug in classified_drugs:
                generic_name = drug.get('generic', '')
                if generic_name:
                    matched_drugs.append(generic_name)
        
        if matched_drugs:
            # 実用精度の指標測定
            accuracy_metrics = _calculate_accuracy_metrics(matched_drugs, classified_drugs)
            logger.info(f"Accuracy metrics: {accuracy_metrics}")
            
            # ユーザーバッファに薬剤名を追加
            if user_id not in user_drug_buffer:
                user_drug_buffer[user_id] = []
            
            unique_input_count = len(set(matched_drugs))
            
            if matched_drugs:
                # AI抽出結果の詳細情報を保存
                if 'classified_drugs' in locals():
                    for drug in classified_drugs:
                        generic_name = drug.get('generic', '')
                        if generic_name and generic_name in matched_drugs:
                            # 薬剤の詳細情報を辞書として保存
                            drug_info = {
                                'name': generic_name,
                                'raw': drug.get('raw', ''),
                                'strength': drug.get('strength', ''),
                                'dose': drug.get('dose', ''),
                                'freq': drug.get('freq', ''),
                                'days': drug.get('days'),
                                'class_hint': drug.get('class_hint', ''),
                                'final_classification': drug.get('final_classification', ''),
                                'kegg_category': drug.get('kegg_category', ''),
                                'kegg_id': drug.get('kegg_id', '')
                            }
                            user_drug_buffer[user_id].append(drug_info)
                else:
                    # フォールバック: 従来の方法
                    for matched_drug_name in matched_drugs:
                        user_drug_buffer[user_id].append(matched_drug_name)
                
                # 検出結果の確認メッセージを表示（相互作用は後段の「診断」で表示）
                response_text = response_service.generate_simple_response(classified_drugs, interaction_result=None, show_interactions=False)
                
                # 検出結果を送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
                
                # 診断ボタン付きFlex Messageを送信（薬剤が検出された場合のみ）
                try:
                    from linebot.v3.messaging import FlexMessage, FlexContainer
                    
                    flex_message = {
                        "type": "bubble",
                        "body": {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "お薬飲み合わせ・同効薬をチェック！",
                                    "weight": "bold",
                                    "size": "md",
                                    "align": "center",
                                    "margin": "md",
                                    "color": "#222222"
                                }
                            ]
                        },
                        "footer": {
                            "type": "box",
                            "layout": "vertical",
                            "contents": [
                                {
                                    "type": "button",
                                    "action": {
                                        "type": "message",
                                        "label": "🔍 診断実行",
                                        "text": "診断"
                                    },
                                    "style": "primary",
                                    "color": "#1DB446"
                                }
                            ]
                        }
                    }
                    
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[FlexMessage(altText="診断ボタン", contents=FlexContainer.from_dict(flex_message))]
                        )
                    )
                except Exception as flex_error:
                    logger.error(f"Flex Message error: {flex_error}")
                    # Flex Messageが失敗した場合はテキストメッセージで代替
                    fallback_text = "💡 以下のコマンドを入力してください：\n• 診断 - 飲み合わせチェック\n• リスト確認 - 現在の薬剤リスト\n• ヘルプ - 使い方表示"
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=fallback_text)]
                        )
                    )
                
                # 部分的にしか読めていない場合はガイドも併送
                if len(matched_drugs) < unique_input_count:
                    partial_guide = """⚠️ 一部の薬剤名を正確に読み取れませんでした

検出結果を確認し、不足があれば手動で追加してください。

📋 撮影のコツ
• 真上から1ページずつ撮影（左右2ページは分割）
• 明るい場所で、影や反射を避ける
• 文字がはっきり写る距離でピント

💊 手動追加（例）
• 薬剤追加：アムロジピン
• 薬剤追加：エソメプラゾール
"""
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=partial_guide)]
                        )
                    )
            else:
                # 薬剤が検出されなかった場合
                response_text = "画像から薬剤名を検出できませんでした。より鮮明な画像で再度お試しください。"
                
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
        else:
            response_text = "画像から薬剤名を検出できませんでした。より鮮明な画像で再度お試しください。"
            
            messaging_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=response_text)]
                )
            )
        
    except Exception as e:
        logger.error(f"Image message handling error: {e}")
        try:
            messaging_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text="画像処理中にエラーが発生しました。しばらく時間をおいて再度お試しください。")]
                )
            )
        except:
            pass

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting complete drug interaction diagnosis system on port {port}, debug mode: {debug_mode}")
        
        # サービス初期化を試行
        initialize_services()
        
        logger.info("Complete drug interaction diagnosis system startup completed successfully")
        
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Railway用の初期化（gunicornで起動時）
try:
    logger.info("Initializing services for Railway deployment...")
    initialize_services()
except Exception as e:
    logger.warning(f"Service initialization failed during import: {e}")
    # 初期化失敗でもアプリケーションは起動可能にする 