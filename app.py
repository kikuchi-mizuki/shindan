import os
import logging
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, MessagingApiBlob, TextMessage, FlexMessage, ReplyMessageRequest, PushMessageRequest, FlexContainer

from dotenv import load_dotenv

from services.ocr_service import OCRService
from services.drug_service import DrugService
from services.response_service import ResponseService
from services.redis_service import RedisService

# 環境変数の読み込み
load_dotenv()

app = Flask(__name__)

# LINE Bot設定
configuration = Configuration(access_token=os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
messaging_blob_api = MessagingApiBlob(api_client)
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# サービスの初期化
ocr_service = OCRService()
drug_service = DrugService()
response_service = ResponseService()
redis_service = RedisService()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ユーザーごとの薬剤名バッファ（Redisが利用できない場合のフォールバック）
user_drug_buffer = {}

def create_diagnosis_flex_message(drug_info: dict) -> FlexMessage:
    contents = []
    # diagnosis_detailsがあれば優先
    if drug_info.get('diagnosis_details'):
        for detail in drug_info['diagnosis_details']:
            box = {
                "type": "box",
                "layout": "vertical",
                "margin": "lg",
                "backgroundColor": "#E3F2FD",
                "cornerRadius": "md",
                "contents": [
                    {"type": "text", "text": f"【{detail.get('type', '診断結果')}】", "weight": "bold", "size": "md", "color": "#1976D2"},
                    {"type": "text", "text": f"対象の薬: {', '.join(detail.get('drugs', []))}", "margin": "md", "size": "sm", "color": "#333333"},
                    {"type": "text", "text": f"薬効分類: {detail.get('category', '不明')}", "size": "xs", "color": "#666666", "margin": "sm"},
                    {"type": "separator", "margin": "md"},
                    {"type": "text", "text": f"理由: {detail.get('reason', '情報がありません')}", "wrap": True, "margin": "md", "size": "sm", "color": "#333333"},
                    {"type": "text", "text": f"考えられる症状: {detail.get('symptoms', '情報がありません')}", "wrap": True, "margin": "md", "size": "sm", "color": "#333333"}
                ]
            }
            contents.append(box)
    else:
        # 従来のsame_effect_warnings等の出力
        # 1. 同効薬重複警告
        if drug_info['same_effect_warnings']:
            for warning in drug_info['same_effect_warnings']:
                box = {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "backgroundColor": "#E3F2FD",
                    "cornerRadius": "md",
                    "contents": [
                        {"type": "text", "text": "【同効薬の重複】", "weight": "bold", "size": "md", "color": "#1976D2"},
                        {"type": "text", "text": f"対象の薬: {warning['drug1']}、{warning['drug2']}", "margin": "md", "size": "sm", "color": "#333333"},
                        {"type": "text", "text": f"薬効分類: {warning.get('category', '不明')}", "size": "xs", "color": "#666666", "margin": "sm"},
                        {"type": "separator", "margin": "md"},
                        {"type": "text", "text": f"理由: {warning.get('reason') or '情報がありません'}", "wrap": True, "margin": "md", "size": "sm", "color": "#333333"},
                        {"type": "text", "text": f"考えられる症状: {warning.get('symptoms') or '情報がありません'}", "wrap": True, "margin": "md", "size": "sm", "color": "#333333"}
                    ]
                }
                contents.append(box)
        # 2. 併用禁忌・併用注意
        critical_interactions = [i for i in drug_info['interactions'] if i.get('risk') in ['critical', 'high']]
        if critical_interactions:
            for interaction in critical_interactions:
                box = {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "backgroundColor": "#FFF3E0",
                    "cornerRadius": "md",
                    "contents": [
                        {"type": "text", "text": "【併用禁忌・併用注意】", "weight": "bold", "size": "md", "color": "#E65100"},
                        {"type": "text", "text": f"対象の薬: {interaction['drug1']}、{interaction['drug2']}", "margin": "md", "size": "sm", "color": "#333333"},
                        {"type": "text", "text": f"リスク: {interaction.get('description', '相互作用あり')}", "size": "xs", "color": "#666666", "margin": "sm"},
                        {"type": "separator", "margin": "md"},
                        {"type": "text", "text": f"理由: {interaction.get('reason') or '情報がありません'}", "wrap": True, "margin": "md", "size": "sm", "color": "#333333"},
                        {"type": "text", "text": f"考えられる症状: {interaction.get('symptoms') or '情報がありません'}", "wrap": True, "margin": "md", "size": "sm", "color": "#333333"}
                    ]
                }
                contents.append(box)
        # 3. 薬剤分類重複チェック
        if drug_info['category_duplicates']:
            for duplicate in drug_info['category_duplicates']:
                box = {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "backgroundColor": "#F3E5F5",
                    "cornerRadius": "md",
                    "contents": [
                        {"type": "text", "text": "【薬剤分類重複】", "weight": "bold", "size": "md", "color": "#6A1B9A"},
                        {"type": "text", "text": f"分類: {duplicate['category']}", "margin": "md", "size": "sm", "color": "#333333"},
                        {"type": "text", "text": f"種類数: {duplicate['count']}", "size": "xs", "color": "#666666", "margin": "sm"},
                        {"type": "text", "text": f"薬剤: {', '.join(duplicate['drugs'])}", "wrap": True, "size": "xs", "color": "#666666", "margin": "sm"}
                    ]
                }
                contents.append(box)
        # 4. その他の相互作用
        other_interactions = [i for i in drug_info['interactions'] if i.get('risk') not in ['critical', 'high']]
        if other_interactions:
            for interaction in other_interactions:
                box = {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "backgroundColor": "#E0F2F1",
                    "cornerRadius": "md",
                    "contents": [
                        {"type": "text", "text": "【その他の相互作用】", "weight": "bold", "size": "md", "color": "#00897B"},
                        {"type": "text", "text": f"対象の薬: {interaction['drug1']}、{interaction['drug2']}", "margin": "md", "size": "sm", "color": "#333333"},
                        {"type": "text", "text": f"リスク: {interaction.get('description', '相互作用あり')}", "size": "xs", "color": "#666666", "margin": "sm"},
                        {"type": "text", "text": f"機序: {interaction.get('mechanism', '不明')}", "size": "xs", "color": "#666666", "margin": "sm"}
                    ]
                }
                contents.append(box)
        # 5. KEGG情報
        if drug_info['kegg_info']:
            for kegg in drug_info['kegg_info']:
                box = {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "backgroundColor": "#E1F5FE",
                    "cornerRadius": "md",
                    "contents": [
                        {"type": "text", "text": "【KEGG情報】", "weight": "bold", "size": "md", "color": "#0277BD"},
                        {"type": "text", "text": f"薬剤名: {kegg['drug_name']}", "margin": "md", "size": "sm", "color": "#333333"},
                        {"type": "text", "text": f"KEGG ID: {kegg.get('kegg_id', '-')}", "size": "xs", "color": "#666666", "margin": "sm"},
                        {"type": "text", "text": f"パスウェイ: {', '.join(kegg.get('pathways', [])[:2])}", "size": "xs", "color": "#666666", "margin": "sm"},
                        {"type": "text", "text": f"ターゲット: {', '.join(kegg.get('targets', [])[:2])}", "size": "xs", "color": "#666666", "margin": "sm"}
                    ]
                }
                contents.append(box)
        # 6. 参考情報の注意書き
        disclaimer_box = {
            "type": "box",
            "layout": "vertical",
            "margin": "lg",
            "backgroundColor": "#FFFDE7",
            "cornerRadius": "md",
            "contents": [
                {"type": "separator", "margin": "md"},
                {"type": "text", "text": "📋 参考情報・注意事項", "weight": "bold", "size": "md", "color": "#FF6F00"},
                {"type": "text", "text": "この診断結果は参考情報です。最終的な判断は医師・薬剤師にご相談ください。", "size": "xs", "color": "#666666", "wrap": True, "margin": "sm"}
            ]
        }
        contents.append(disclaimer_box)

    flex_message = {
        "type": "bubble",
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": contents
        }
    }
    return FlexMessage(altText="飲み合わせ診断結果", contents=FlexContainer.from_dict(flex_message))

def create_enhanced_quick_actions_flex_message(drug_count: int = 0) -> FlexMessage:
    """診断実行ボタンのみのクイックアクションFlex Messageを作成"""
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
    return FlexMessage(altText="クイックアクション", contents=FlexContainer.from_dict(flex_message))

def create_enhanced_disclaimer_flex_message() -> FlexMessage:
    """強化版免責事項と使い方案内のFlex Messageを作成"""
    
    flex_message = {
        "type": "bubble",
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "📋 免責事項・使い方案内",
                    "weight": "bold",
                    "size": "lg",
                    "color": "#1DB446",
                    "align": "center"
                },
                {
                    "type": "separator",
                    "margin": "md"
                },
                {
                    "type": "text",
                    "text": "⚠️ 重要な注意事項",
                    "weight": "bold",
                    "size": "md",
                    "color": "#FF0000",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "• この診断結果は参考情報です",
                    "size": "sm",
                    "color": "#666666",
                    "margin": "sm"
                },
                {
                    "type": "text",
                    "text": "• 最終的な判断は薬剤師にお任せください",
                    "size": "sm",
                    "color": "#666666"
                },
                {
                    "type": "text",
                    "text": "• 副作用や体調の変化があればすぐに医師・薬剤師に相談してください",
                    "size": "sm",
                    "color": "#666666"
                },
                {
                    "type": "text",
                    "text": "• 個人情報が含まれる画像は送信しないでください",
                    "size": "sm",
                    "color": "#666666"
                },
                {
                    "type": "separator",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "💡 使い方",
                    "weight": "bold",
                    "size": "md",
                    "color": "#0066CC",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "1. お薬手帳や処方箋の画像を送信",
                    "size": "sm",
                    "color": "#666666",
                    "margin": "sm"
                },
                {
                    "type": "text",
                    "text": "2. 複数画像がある場合は順番に送信",
                    "size": "sm",
                    "color": "#666666"
                },
                {
                    "type": "text",
                    "text": "3. すべて送信後「診断」と入力",
                    "size": "sm",
                    "color": "#666666"
                },
                {
                    "type": "text",
                    "text": "4. Botが飲み合わせリスクをチェック",
                    "size": "sm",
                    "color": "#666666"
                },
                {
                    "type": "separator",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "🚨 緊急時の連絡先",
                    "weight": "bold",
                    "size": "md",
                    "color": "#FF0000",
                    "margin": "lg"
                },
                {
                    "type": "text",
                    "text": "• 救急車: 119",
                    "size": "sm",
                    "color": "#666666",
                    "margin": "sm"
                },
                {
                    "type": "text",
                    "text": "• 中毒110番: 072-727-2499",
                    "size": "sm",
                    "color": "#666666"
                },
                {
                    "type": "text",
                    "text": "• かかりつけの薬局・病院に連絡",
                    "size": "sm",
                    "color": "#666666"
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
                },
                {
                    "type": "button",
                    "action": {
                        "type": "message",
                        "label": "📋 リスト確認",
                        "text": "リスト確認"
                    },
                    "style": "secondary",
                    "margin": "sm"
                },
                {
                    "type": "button",
                    "action": {
                        "type": "message",
                        "label": "❓ ヘルプ",
                        "text": "ヘルプ"
                    },
                    "style": "secondary",
                    "margin": "sm"
                }
            ]
        }
    }
    
    return FlexMessage(altText="免責事項・使い方案内", contents=FlexContainer.from_dict(flex_message))

def create_quick_actions_flex_message() -> FlexMessage:
    """クイックアクションボタンを含むFlex Messageを作成（後方互換性）"""
    return create_enhanced_quick_actions_flex_message()

@app.route("/callback", methods=['POST'])
def callback():
    """LINE Webhookからのコールバック処理"""
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    logger.info("Request body: " + body)
    
    try:
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
        
        # 診断コマンド
        if user_message.lower() in ['診断', 'しんだん', 'check', 'diagnosis']:
            # Redisから薬剤リストを取得（フォールバック: メモリバッファ）
            if redis_service.is_redis_available():
                drug_names = redis_service.get_user_drugs(user_id)
            else:
                drug_names = user_drug_buffer.get(user_id, [])
            
            if not drug_names:
                reply = """【薬剤名未登録】\n━━━━━━━━━━━━━━━\n❌ 薬剤名が登録されていません\n\n💡 推奨事項:\n・画像を送信してください\n・複数画像がある場合は順番に送信\n━━━━━━━━━━━━━━━"""
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text=reply)]
                    )
                )
            else:
                # 薬剤情報の取得と診断（マッチング済みの薬剤名を使用）
                matched_drugs = drug_service.match_to_database(drug_names)
                if matched_drugs:
                    drug_info = drug_service.get_drug_interactions(matched_drugs)
                else:
                    drug_info = {
                        'detected_drugs': [],
                        'interactions': [],
                        'same_effect_warnings': [],
                        'category_duplicates': [],
                        'kegg_info': [],
                        'warnings': ['薬剤が見つかりませんでした'],
                        'recommendations': ['薬剤師にご相談ください']
                    }
                
                # 診断履歴をRedisに保存
                if redis_service.is_redis_available():
                    redis_service.save_diagnosis_history(user_id, drug_info)
                
                # Flex Messageで診断結果を表示
                print('=== DIAGNOSIS drug_info ===')
                print(drug_info)
                diagnosis_text = response_service.generate_response(drug_info)
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        replyToken=event.reply_token,
                        messages=[TextMessage(text=diagnosis_text)]
                    )
                )
        
        # リスト確認コマンド
        elif user_message.lower() in ['リスト確認', 'りすとかくにん', 'list', '確認']:
            # Redisから薬剤リストを取得（フォールバック: メモリバッファ）
            if redis_service.is_redis_available():
                drug_list = redis_service.get_user_drugs(user_id)
            else:
                drug_list = user_drug_buffer.get(user_id, [])
            
            if drug_list:
                reply = "【現在の薬剤リスト】\n━━━━━━━━━\n"
                for i, drug in enumerate(drug_list, 1):
                    reply += f"{i}. {drug}\n"
                reply += "━━━━━━━━━\n💡 「診断」で飲み合わせチェックを実行できます"
            else:
                reply = "【現在の薬剤リスト】\n━━━━━━━━━\n（登録なし）\n━━━━━━━━━\n💡 画像を送信してください。"
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
        
        # ヘルプコマンド
        elif user_message.lower() in ['help', 'ヘルプ', '使い方']:
            # 強化版の免責事項・使い方案内Flex Messageを送信
            disclaimer_flex = create_enhanced_disclaimer_flex_message()
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[disclaimer_flex]
                )
            )
        
        # 薬剤追加コマンド
        elif user_message.startswith('薬剤追加：'):
            drug_name = user_message.replace('薬剤追加：', '').strip()
            if drug_name:
                if user_id not in user_drug_buffer:
                    user_drug_buffer[user_id] = []
                user_drug_buffer[user_id].append(drug_name)
                reply = f"【薬剤追加完了】\n━━━━━━━━━\n✅ {drug_name} を追加しました\n\n現在のリスト: {len(user_drug_buffer[user_id])}件\n━━━━━━━━━"
            else:
                reply = "【エラー】\n━━━━━━━━━\n❌ 薬剤名を入力してください\n\n例: 薬剤追加：アスピリン\n━━━━━━━━━"
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
        
        # 薬剤削除コマンド
        elif user_message.startswith('薬剤削除：'):
            try:
                index = int(user_message.replace('薬剤削除：', '').strip()) - 1
                if user_id in user_drug_buffer and 0 <= index < len(user_drug_buffer[user_id]):
                    removed_drug = user_drug_buffer[user_id].pop(index)
                    reply = f"【薬剤削除完了】\n━━━━━━━━━\n✅ {removed_drug} を削除しました\n\n現在のリスト: {len(user_drug_buffer[user_id])}件\n━━━━━━━━━"
                else:
                    reply = "【エラー】\n━━━━━━━━━\n❌ 指定された番号の薬剤が見つかりません\n━━━━━━━━━"
            except ValueError:
                reply = "【エラー】\n━━━━━━━━━\n❌ 番号を入力してください\n\n例: 薬剤削除：1\n━━━━━━━━━"
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
        
        # リストクリアコマンド
        elif user_message.lower() in ['リストクリア', 'りすとくりあ', 'clear']:
            user_drug_buffer[user_id] = []
            reply = "【リストクリア完了】\n━━━━━━━━━\n✅ 薬剤リストをクリアしました\n\n💡 新しい画像を送信するか、『薬剤追加：〇〇』で手動追加できます\n━━━━━━━━━"
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )
        
        # 詳細情報コマンド
        elif user_message.lower() in ['詳細情報', 'しょうさいじょうほう', 'detail']:
            if user_id in user_drug_buffer and user_drug_buffer[user_id]:
                detail_text = "【詳細情報】\n━━━━━━━━━\n"
                detail_text += "📌 現在の薬剤リスト:\n"
                for i, drug in enumerate(user_drug_buffer[user_id], 1):
                    detail_text += f"{i}. {drug}\n"
                detail_text += "\n🔬 KEGG情報:\n"
                detail_text += "• 代謝経路と作用標的の詳細\n"
                detail_text += "• 薬物相互作用の機序\n"
                detail_text += "• 副作用リスクの詳細\n"
                detail_text += "\n💡 推奨事項:\n"
                detail_text += "• 薬剤師への相談を推奨\n"
                detail_text += "• 定期的な血圧・心拍数チェック\n"
                detail_text += "• 副作用の早期発見\n"
                detail_text += "━━━━━━━━━"
            else:
                detail_text = "【詳細情報】\n━━━━━━━━━\n❌ 薬剤リストが空です\n\n💡 画像を送信して薬剤を登録してください\n━━━━━━━━━"
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=detail_text)]
                )
            )
        
        # 薬剤追加コマンド（ボタンからの場合）
        elif user_message.lower() in ['薬剤追加', 'やくざいついか', 'add']:
            add_guide_text = "【薬剤追加】\n━━━━━━━━━\n💡 以下の形式で薬剤名を入力してください：\n\n例: 薬剤追加：アスピリン\n例: 薬剤追加：カルベジロール\n\n📋 現在のリスト: "
            if user_id in user_drug_buffer:
                add_guide_text += f"{len(user_drug_buffer[user_id])}件"
            else:
                add_guide_text += "0件"
            add_guide_text += "\n━━━━━━━━━"
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[TextMessage(text=add_guide_text)]
                )
            )
        
        # その他のメッセージ
        else:
            # 強化版の免責事項・使い方案内Flex Messageを送信
            disclaimer_flex = create_enhanced_disclaimer_flex_message()
            messaging_api.reply_message(
                ReplyMessageRequest(
                    replyToken=event.reply_token,
                    messages=[disclaimer_flex]
                )
            )
    except Exception as e:
        logger.error(f"Text message handling error: {e}")
        messaging_api.reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text="""🩺【エラーが発生しました】\n━━━━━━━━━━━━━━━\n❌ 申し訳ございません\nエラーが発生しました\n\n💡 推奨事項:\n・しばらく時間をおいて再度お試しください\n・薬剤師にご相談ください\n━━━━━━━━━━━━━━━""")]
            )
        )

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """画像メッセージの処理"""
    try:
        user_id = event.source.user_id
        # 処理開始メッセージ
        messaging_api.reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text="画像を確認中です...")]
            )
        )
        # 画像の取得
        message_content = messaging_blob_api.get_message_content(event.message.id)
        # OCRで薬剤名を抽出
        drug_names = ocr_service.extract_drug_names(message_content)
        
        if drug_names:
            # ユーザーバッファに薬剤名を追加
            if user_id not in user_drug_buffer:
                user_drug_buffer[user_id] = []
            
            # マッチした薬剤名をバッファに追加
            matched_drugs = drug_service.match_to_database(drug_names)
            if matched_drugs:
                for matched_drug_name in matched_drugs:
                    user_drug_buffer[user_id].append(matched_drug_name)
                # Redisにも保存
                if redis_service.is_redis_available():
                    redis_service.set_user_drugs(user_id, user_drug_buffer[user_id])
            
            # KEGG情報を含む詳細な薬剤情報を取得
            if matched_drugs:
                drug_info = drug_service.get_drug_interactions(matched_drugs)
                
                # レスポンステキストの作成（KEGG情報付き）
                response_text = f"【薬剤検出完了】\n━━━━━━━━━\n✅ {len(matched_drugs)}件の薬剤を検出しました\n\n"
                
                # 検出された薬剤の詳細情報を表示
                for i, drug in enumerate(drug_info['detected_drugs'], 1):
                    response_text += f"{i}. {drug['name']}\n"
                    if drug.get('category'):
                        response_text += f"   分類: {drug['category']}\n"
                    if drug.get('generic_name'):
                        response_text += f"   一般名: {drug['generic_name']}\n"
                    response_text += "\n"
                
                # KEGG情報がある場合は表示
                if drug_info['kegg_info']:
                    response_text += "【KEGG情報】\n━━━━━━━━━\n"
                    for kegg_data in drug_info['kegg_info']:
                        response_text += f"• {kegg_data['drug_name']}\n"
                        if kegg_data.get('kegg_id'):
                            response_text += f"  KEGG ID: {kegg_data['kegg_id']}\n"
                        if kegg_data.get('pathways'):
                            response_text += f"  パスウェイ: {', '.join(kegg_data['pathways'][:2])}\n"
                        if kegg_data.get('targets'):
                            response_text += f"  ターゲット: {', '.join(kegg_data['targets'][:2])}\n"
                        response_text += "\n"
                
                response_text += f"現在のリスト: {len(user_drug_buffer[user_id])}件\n━━━━━━━━━\n💡 「診断」で飲み合わせチェックを実行できます"
                
                # テキストメッセージとクイックアクションボタンを送信
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
                
                # クイックアクションボタン付きFlex Messageを送信
                try:
                    quick_actions_flex = create_enhanced_quick_actions_flex_message(len(user_drug_buffer[user_id]))
                    messaging_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[quick_actions_flex]
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
            else:
                response_text = "【薬剤名未検出】\n━━━━━━━━━\n薬剤名が検出されませんでした\n\n推奨事項:\n・より鮮明な画像で撮影してください\n・薬剤名が明確に写るようにしてください\n・『薬剤追加：〇〇』で手動追加も可能です\n━━━━━━━━━"
                messaging_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text=response_text)]
                    )
                )
        else:
            response_text = "【薬剤名未検出】\n━━━━━━━━━\n薬剤名が検出されませんでした\n\n推奨事項:\n・より鮮明な画像で撮影してください\n・薬剤名が明確に写るようにしてください\n・『薬剤追加：〇〇』で手動追加も可能です\n━━━━━━━━━"
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
                    to=event.source.user_id,
                    messages=[TextMessage(text="画像処理エラーが発生しました。しばらく時間をおいて再度お試しください。より鮮明な画像で撮影してください。")]
                )
            )
        except Exception as push_error:
            logger.error(f"Push message error: {push_error}")
            # プッシュメッセージも失敗した場合は何もしない

@app.route("/health", methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "message": "薬局サポートBot is running"}

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 