# Railway 本番環境設定ガイド

## 🚀 Railwayでのデプロイ手順

### 1. Railwayプロジェクトの作成

1. [Railway](https://railway.app/)にアクセス
2. GitHubアカウントでログイン
3. 「New Project」→「Deploy from GitHub repo」
4. このリポジトリを選択

### 2. 環境変数の設定

Railwayのダッシュボードで以下の環境変数を設定：

#### 必須設定
```
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
LINE_CHANNEL_SECRET=your_line_channel_secret
GOOGLE_APPLICATION_CREDENTIALS=medicine-support:credentials:vision-key.json
```

#### Redis設定（Redisアドオン追加後）
```
REDIS_URL=your_railway_redis_url
REDIS_HOST=your_railway_redis_host
REDIS_PORT=your_railway_redis_port
REDIS_DB=0
```

#### その他の設定
```
PORT=8080
FLASK_ENV=production
```

### 3. Redisアドオンの追加

1. Railwayプロジェクトで「New」→「Database」→「Redis」
2. 作成されたRedisの「Connect」タブから接続情報を取得
3. 環境変数に設定

### 4. Google Cloud認証情報の設定

1. Google Cloud Consoleでサービスアカウントキーをダウンロード
2. Railwayの「Variables」タブで「New Variable」
3. キー名: `GOOGLE_APPLICATION_CREDENTIALS`
4. 値: ダウンロードしたJSONファイルの内容をコピー&ペースト

### 5. LINE Bot Webhook URLの更新

1. Railwayでデプロイ完了後、提供されるURLをコピー
2. LINE Developers ConsoleでWebhook URLを更新:
   ```
   https://your-app-name.railway.app/callback
   ```

### 6. デプロイ確認

1. Railwayダッシュボードでデプロイ状況を確認
2. ヘルスチェック: `https://your-app-name.railway.app/health`
3. LINE Botでテストメッセージを送信

## 🔧 トラブルシューティング

### Redis接続エラー
- Redisアドオンが正しく追加されているか確認
- 環境変数`REDIS_URL`が正しく設定されているか確認

### Google Cloud認証エラー
- サービスアカウントキーが正しく設定されているか確認
- Vision APIが有効になっているか確認

### LINE Bot接続エラー
- Webhook URLが正しく設定されているか確認
- LINE Channel Access Tokenが正しいか確認

## 📊 監視とログ

- Railwayダッシュボードでログを確認
- アプリケーションのヘルスチェックを定期的に実行
- Redis接続状況を監視

## 🔄 更新手順

1. GitHubにコードをプッシュ
2. Railwayで自動的にデプロイが開始
3. デプロイ完了後、動作確認 