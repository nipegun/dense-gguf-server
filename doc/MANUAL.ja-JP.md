# DGS マニュアル

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · **ja-JP** · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server は、**NiPeGun** が開発・保守する独立したプロジェクトです。

DGS は次の密なモデルと、それに対応する Web チャットアプリケーションを提供します。

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

GLM と Kimi の全バージョンを含め、その他の言語モデルはサポートしません。MoE モデルも対象外です。サーバーは元の GGUF 識別情報と層の次元を検証し、投機的デコード用のドラフトモデルにも同じ検証を行います。許可されたモデルの量子化版は引き続き利用できます。

<a id="local-build"></a>

## Debian でのビルドと実行

Debian 13、または C++17、CMake 3.19+、Python 3.13+、Node.js 20.19+ もしくは 22.12+ が使える互換システムを使用します。root としてパッケージをインストールします。

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

`http://127.0.0.1:8080` を開きます。重みは別途用意してください。画像や動画には `--mmproj` で対応する Qwen プロジェクターを指定します。動画処理には FFmpeg も必要です。音声モデルと音声生成はサポートしません。

Release モード、ネイティブ CPU カーネル、利用可能な場合のリンク時最適化は既定で有効です。CUDA と Vulkan のビルドには、対応 SDK と `-DGGML_CUDA=ON` または `-DGGML_VULKAN=ON` を使います。配備先の CPU 向けにビルドするか、`-DGGML_NATIVE=OFF` を使用してください。スループット向上は測定していません。

## 現在の形式のみ

GGUF v3 のみ受け付けます。`general.name`、または `general.basename` と `general.size_label` の組み合わせが許可モデルを識別する必要があり、ファイル名だけでは不十分です。これはメタデータと次元の検証であり、重みの真正性を認証するものではありません。次元が同じでも Qwen3.5-27B のファイルを Qwen3.8-27B として受け付けることはありません。

チャットにはモデルの完全な Jinja テンプレートか `--chat-template-file` が必要です。旧固定テンプレートエンジン、`--jinja`/`--no-jinja` スイッチ、廃止済み C API、古い長形式 CLI オプションの別名、旧推論モードは削除されています。現在の `/api/v1/*` 契約は維持されますが、旧別名 `/completion` と `/embedding` はありません。

Web UI は現在の保存名前空間 `DgsUi` と構造化メッセージを使います。旧名前空間、テーマ、メッセージマーカー、旧 JSON 会話エクスポートの読込や移行は行いません。現在の JSONL セッションエクスポートと ZIP アーカイブをサポートします。ブラウザー内の古いデータは削除しません。MCP は StreamableHTTP または WebSocket を使い、旧 SSE トランスポートと自動フォールバックは削除されています。クリップボード操作には安全なコンテキストでの現在の Clipboard API が必要です。

## 構成とドキュメント

- `backend/`：サーバー、Qwen 推論、共通テンソルカーネル、Qwen 視覚処理。
- `frontend/`：現在の Svelte アプリとバージョン固定された依存関係。
- `deploy/`：ビルド補助とローカル Web UI の埋め込み。
- [技術マップ](CODE.ja-JP.md)と[マニュアル](MANUAL.ja-JP.md)。

通常のビルドではローカル Web UI をビルドして埋め込みます。`cd frontend && npm ci && npm run build` で生成済みの UI を再利用する場合は `-DDGS_USE_LOCAL_UI=ON` を指定します。ソース変更後はリソースを再生成してください。`-DDGS_BUILD_UI=OFF` は埋め込みリソースを省きます。このモードでは `--path` で現在のビルド済み UI ディレクトリを指定します。

プロジェクトの規則ではテストは禁止されています。検証はコンパイルと静的型チェックに限られ、モデル推論やベンチマークは実行していません。このプロジェクトでは Git を使用せず、保護された `.git` ディレクトリの削除は所有者が手動で行います。

## 日常の操作

Web UI で会話を開始し、読込済みの許可モデルを選び、テキストや対応する画像・動画を送信します。設定ではサンプリング、コンテキスト、ツールを変更できます。`-c` はコンテキスト長、`-t` は CPU スレッド数、`-ngl auto` は GPU への自動配置を指定します。認証には `--api-key` を使います。互換クライアントの接続先は `http://127.0.0.1:8080/api/v1` と `/api/v1/chat/completions` です。モデルディレクトリに別のファイルがあっても、未対応のチェックポイントは拒否されます。

読み込みに失敗したら、モデルの識別情報、アーキテクチャ、次元、GGUF バージョンを確認します。検証を回避するために別モデルの識別情報を書き換えないでください。メモリ不足ではコンテキスト、バッチサイズ、GPU オフロード量を減らします。UI がない場合は再ビルドするか `--path` を指定します。

## MCP と Web 機能

内蔵シェル MCP サーバーは `--local-shell-mcp-server` または `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1` で有効にします。UI は自動検出し、ツール呼び出しの許可を求めます。インストール済みの Bash、Zsh、sh、PowerShell を公開し、コマンド、stdout、stderr を表示して、各呼び出しを JSONL に記録します。

Web UI には CPU/GPU・メモリモニター、ステータスバー、サーバー側共有設定、15 言語の選択、選択した地域の言語で回答させる任意の指示、Web と他の API クライアントを区別する待ち行列通知があります。モデル閲覧、ダウンロード、会話タブ、密なモデルの制限は維持されます。

UI は `/`、API は既定で `/api` の下にあります。OpenAI 互換クライアントのベースパスは `/api/v1` です。Swagger は `/api/doc/` で利用できます。新 API にルート直下の旧別名はありません。`--api-prefix` で `/api` 以下のパスを指定でき、`/api/config.js` がそれを UI に伝えます。

## ローカルシェル MCP

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

UI を開いて API キーを入力します。サーバーで有効なら `local shell mcp server` が自動で表示され、全体または会話ごとに無効化できます。`bash.run`、`zsh.run`、`sh.run`、`psh.run` はインストール済みのインタープリターについてのみ利用できます。エンドポイントは `POST /api/mcp/shell`、通信は MCP Streamable HTTP、応答は JSON、プロトコルは `2025-11-25` です。

各ツールは `command` と、任意の `cwd`、`timeout` を受け取ります。タイムアウトは秒単位で既定 60、最大 3600 です。呼び出しは対話端末のない独立プロセスで実行され、作業ディレクトリや環境の変更は次の呼び出しへ継承されません。キャンセルや期限超過でプロセスグループを終了します。コマンドはサーバーアカウントの権限で動くため、API キーでアクセスを保護してください。

チャットには stdin（コマンド）、stdout、stderr と終了コードが別々に表示されます。表示設定で自動展開できます。`~/.dgs/local-shell-mcp-server/` のログは、呼び出しごとの時刻付き JSONL に完全なコマンドと出力を保存します。チャットでは長い出力を省略することがあります。ファイルはサーバーアカウント専用で、自動ローテーションはありません。

sudo が既にある Linux では、パスワード要求が入力欄にフォーカスした許可・キャンセルのダイアログに表示されます。パスワードは補助プロセスにだけ送られ、モデル、会話、ログには渡りません。ユーザーの応答待ちの間はコマンドの制限時間を停止し、未回答の要求は 5 分で失効します。本プロジェクトは sudo をインストールせず、本番動作にも要求しません。

## モニターと共有設定

サイドバーの**モニター**または下部ステータスバーを開きます。ページが表示されている間、CPU/GPU 使用率は 2 秒ごとに更新されます。詳細画面には、ドライバーが提供する RAM、VRAM、温度、消費電力も表示されます。メモリは密な重み、MTP/ドラフト重み、KV/再帰状態キャッシュ、計算バッファー、保存済みプロンプト/チェックポイント、その他、空き容量に分かれます。NVIDIA は NVML を動的に、AMD は sysfs を使用し、取得できない値は不明のままです。ルーターモードは読込済み子プロセスを集計し、モデルを読み込んだり起こしたりしません。

設定はサーバーアカウントの `~/.dgs/config/ui-settings.json` にモード 0600 で保存されます。同じサーバーのブラウザーは設定、テーマ、言語、お気に入り、ツールの設定を共有します。保存は文書全体を置換するため、同時編集では最後の成功した保存が優先されます。API キー、会話、一時的なナビゲーションやストリーム状態はブラウザー内に残ります。古いブラウザー設定は移行しません。

「一般」で言語を選択します。変更すると選択を保存して UI を再読み込みするため、他の未保存の変更を先に保存してください。対応ロケールは `de-DE`、`en-GB`、`en-US`、`es-AR`、`es-ES`、`fr-FR`、`he-IL`、`hi-IN`、`it-IT`、`ja-JP`、`ko-KR`、`pt-BR`、`pt-PT`、`ru-RU`、`zh-CN` です。既定と未翻訳文字列の代替は米国英語です。**インターフェースの言語で回答する** (Respond in the interface language)を有効にすると、新しいリクエスト、ツールのターン、生成タイトル、KV 事前入力に地域言語の指示を追加します。既存のメッセージは翻訳せず、指示への従い方はモデルに依存します。

リクエストが空きスロットを待つ間、前にあるリクエスト数と、直接 API から来たものが含まれるかを表示します。通常の API クライアントの応答形式は変わりません。`"dgs_client": "webui"` を付けたストリーミングリクエストだけが `dgs_queue` イベントを受信します。

フロントエンド開発では `npm run dev` が `/api` を `http://localhost:9931` にプロキシします。別サーバーを使う場合は `VITE_PUBLIC_SERVER_ORIGIN` を設定します。本番ビルドは Web UI とローカル Swagger を埋め込むため、外部の文書 CDN は不要です。

## ソースとビルドのディレクトリ

`build/` には CMake の入口と `build-local.sh` があります。スクリプトは自身の場所を基準にプロジェクトを特定し、Release を設定して追加の CMake 引数を受け取ります。依存関係は事前にインストールしてください。`DGS_BUILD_JOBS` の既定値は 4 です。手動設定には `cmake -S build -B _/build` を使用し、エンジンのプロジェクトは `backend/CMakeLists.txt` が定義します。生成ファイルと `bin/dgs-server` は `_/build/` に入り、ビルドスクリプトとは別です。

コンパイル後、本番環境には root として `cmake --install _/build --prefix /usr/local` でインストールします。Web 準備の CMake 規則は `deploy/scripts/ui-assets.cmake` にあり、入力を `frontend/sources.cmake` が列挙し、`frontend/embed.py` が埋め込み C++ リソースを生成します。メイン README はルート、地域版は `doc/` にあります。


<a id="license"></a>

## ライセンスと第三者の表示

NiPeGun による貢献には [PolyForm Noncommercial 1.0.0](../LICENSE) が適用されます。組み込まれた第三者のコードには元のライセンスと著作権表示が引き続き適用され、[build/licenses/](../build/licenses/README.md) にまとめられています。

メイン LICENSE は原文のまま維持します。特に ggml 作者の MIT 表示は `build/licenses/LICENSE-llama.cpp` に残し、Swagger UI の Apache ライセンスと NOTICE も保持します。一覧には、組み込まれた HTTP、JSON、ハッシュ、メディア、subprocess、base64、CPU カーネル、数学ライブラリーの表示も含みます。依存ソースや Web バンドル内の表示も引き続き適用されます。モデルの重みとシステムライブラリーには独自の条件があります。

既定のデータ配置では、`cmake --install _/build --prefix PREFIX` がプロジェクト LICENSE と `build/licenses/` を `PREFIX/share/doc/dgs-server/licenses/` にコピーします。既定のプレフィックス `/usr/local` では `/usr/local/share/doc/dgs-server/licenses/` に配置され、ユーザー別には `~/.local` も指定できます。`Licenses` コンポーネントは単独でもインストールできます。実行ファイルを手動配布する場合もこれらの文書を同梱してください。
