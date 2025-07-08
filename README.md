# 💊 薬局サポートBot

AIを活用した薬剤飲み合わせチェックシステム。処方箋やお薬手帳の画像をLINEで送信すると、薬剤名を自動抽出し、飲み合わせリスクをチェックします。

## 🎯 目的

- **薬局の教育コスト削減**: 新人薬剤師の学習を効率化
- **リスク低減**: 飲み合わせ確認のミスを防止
- **業務効率化**: スマホ完結で直感的な操作

## 🧩 解決する課題

- 薬局は常に忙しく、新人教育が後回しになりがち
- 飲み合わせチェックは知識と経験頼りでミスが許されない
- 新人が判断に迷う場面が多い

## 🆕 新機能（v2.0）

### 同効薬チェック機能
- 同じ作用機序を持つ薬剤の重複投与を検出
- COX-1/COX-2阻害薬、抗凝固薬、スタチン系薬剤などの同効薬グループを定義
- リスクレベル（critical/high/medium/low）による分類

### 相互作用の詳細度向上
- 相互作用の機序（CYP3A4阻害、タンパク結合競合など）を詳細表示
- リスクレベルによる優先度付け
- より具体的な副作用情報の提供

### 薬剤分類による重複チェック
- 治療分類（解熱鎮痛薬、抗凝固薬、脂質異常症治療薬など）による重複検出
- 同一分類内での複数薬剤投与の警告
- 薬剤師の処方チェックを支援

### KEGGデータベース連携
- 京都大学ゲノムネットワークのKEGGデータベースから薬剤情報を取得
- 薬剤の作用パスウェイ情報の提供
- 作用ターゲット（酵素、受容体など）の特定

## ✅ 機能

### 現在の機能（PoC段階）
- 📸 画像から薬剤名の自動抽出（OCR）
- 🔍 PMDAデータベースとの照合
- ⚠️ 飲み合わせリスクの自動チェック
- 💬 分かりやすい応答メッセージ

### 将来の機能拡張予定
- 🤖 ChatGPTによる自然文生成
- 📊 よくある薬品パターンの記憶
- 📈 使用履歴の分析

## 🖥 技術仕様

### 使用技術
- **言語**: Python 3.11
- **フレームワーク**: Flask
- **LINE API**: LINE Messaging API
- **OCR**: Google Cloud Vision API
- **データベース**: PMDA公開データ（CSV）
- **デプロイ**: Railway

### セキュリティ
- 個人情報は保存せず、処理後即時破棄
- Bot判定回避のため、コピペで実行・手動送信
- 医療行為と誤認されないよう、補助ツールとして明示

## 👤 想定ユーザー

- 20〜60代の新人薬剤師（業務経験1〜3年）
- 街の調剤薬局で教育の負担を抱える管理薬剤師
- 人材育成やリスク軽減に関心のある薬局経営者

## 🚀 セットアップ

### 前提条件
- Python 3.11以上
- LINE Developers アカウント
- Google Cloud Platform アカウント（Vision API有効化）

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd medicine-support
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定
```bash
cp env.example .env
```

`.env`ファイルを編集して以下の値を設定：
```env
# LINE Bot設定
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token_here
LINE_CHANNEL_SECRET=your_line_channel_secret_here

# Google Cloud Vision API設定
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json

# OpenAI API設定（将来の拡張用）
OPENAI_API_KEY=your_openai_api_key_here

# アプリケーション設定
PORT=5000
FLASK_ENV=development
```

### 4. データベースの初期化
```bash
python scripts/init_database.py
```

### 5. テストの実行
```bash
python scripts/test_bot.py
```

### 6. アプリケーションの起動
```bash
python app.py
```

## 📱 LINE Bot設定

### 1. LINE Developers でチャネル作成
1. [LINE Developers Console](https://developers.line.biz/) にアクセス
2. 新しいプロバイダーを作成
3. Messaging APIチャネルを作成

### 2. Webhook URL設定
- Webhook URL: `https://your-domain.com/callback`
- Webhook送信: 有効
- 応答メッセージ: 無効（推奨）

### 3. チャネルアクセストークン取得
- チャネルアクセストークン（長期）を取得
- チャネルシークレットを取得

## 🔧 開発

### プロジェクト構造
```
medicine-support/
├── app.py                 # メインアプリケーション
├── requirements.txt       # 依存関係
├── services/             # サービス層
│   ├── ocr_service.py    # OCR処理
│   ├── drug_service.py   # 薬剤情報処理
│   └── response_service.py # 応答生成
├── utils/                # ユーティリティ
│   └── pmda_downloader.py # PMDAデータ管理
├── scripts/              # スクリプト
│   ├── init_database.py  # DB初期化
│   └── test_bot.py       # テスト
├── data/                 # データファイル
└── README.md            # このファイル
```

### テスト
```bash
# 全機能テスト
python scripts/test_bot.py

# 個別テスト
python -c "from services.drug_service import DrugService; print(DrugService().get_drug_interactions(['アスピリン']))"
```

## 📊 PMDAデータ連携

### データ取得
- PMDAが公開する医薬品の添付文書CSVを活用
- 含まれる情報：販売名・一般名・薬効分類・製造業者名など
- 月1〜2回の自動更新に対応予定

### データ処理
```python
from utils.pmda_downloader import PMDADownloader

downloader = PMDADownloader()
downloader.update_database()  # データベース更新
```

## 🔄 開発ロードマップ

### フェーズ1: PoC（現在）
- ✅ LINEで画像送信 → 薬剤名抽出
- ✅ 飲み合わせリスクチェック
- ✅ 小規模薬局でのテスト運用

### フェーズ2: 機能拡張
- 🤖 ChatGPTによる自然文生成
- 📊 よくある薬品パターンの記憶
- 🔍 応答候補の強化

### フェーズ3: 正式リリース
- 📱 LINE公式アカウントとして提供
- 💰 月額課金制（サブスクモデル）

### フェーズ4: 横展開
- 🏥 全国の薬局チェーン・個人薬局への導入
- 📚 教育 × 確認 × 顧客説明の業務支援

## ⚠️ 注意事項

### 医療行為との区別
- このBotは薬剤師の確認を補助するツールです
- 最終的な判断は薬剤師にお任せください
- 医療行為と誤認されないよう設計されています

### 個人情報の取り扱い
- 処方箋画像は一時処理のみで保存しません
- 利用者には「個人情報を写さないように」注意喚起
- 処理後は即座にデータを破棄

### 利用制限
- LINE APIの無料枠（月1000通）内で運用
- 将来的には有料プランへの移行を検討

## 🤝 貢献

### 開発への参加
1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

### フィードバック
- バグ報告や機能要望は [Issues](../../issues) でお知らせください
- 薬剤師の皆様からの実務的なフィードバックを歓迎します

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 📞 サポート

- **技術的な質問**: [Issues](../../issues) でお知らせください
- **薬剤師向けサポート**: 薬剤師会や関連団体を通じてご連絡ください
- **商用利用**: 別途ライセンス契約が必要です

## 🙏 謝辞

- PMDA（独立行政法人医薬品医療機器総合機構）のデータ提供
- LINE株式会社のMessaging API
- Google Cloud PlatformのVision API
- 薬剤師の皆様からの貴重なフィードバック

---

**⚠️ 免責事項**: このBotは薬剤師の確認を補助するツールです。医療行為ではありません。最終的な判断は薬剤師にお任せください。 