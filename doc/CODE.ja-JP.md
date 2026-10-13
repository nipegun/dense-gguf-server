# DGS — コードマップ

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · **ja-JP** · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server は、**NiPeGun** が開発・保守する独立したプロジェクトです。

- [アーキテクチャ](#architecture)
- [モジュール](#modules)
- [主要シンボル](#symbols)
- [呼び出しフロー](#flows)
- [入口](#entrypoints)
- [影響](#impact)
- [拡張](#extension)
- [MCP、モニター、共有 UI](#integrated-features)
- [情報源と検証](#validation)
- [リポジトリ構成](#repository-organization)

- [ライセンスと第三者の表示](#license)

<a id="architecture"></a>

## アーキテクチャ

サーバーと Web アプリを各 1 つ提供します。ファクトリーは `dgs_model_qwen35` のみを生成し、Qwen3.8-27B もこの構造を共有します。DeltaNet カーネル、Qwen プロジェクターと共通視覚ヘルパーは維持します。MoE、旧 GGUF、他のプロファイルは拒否します。許可リストは HTTP 別名と独立して元のメタデータを読み、層、埋め込み、FFN、ヘッドを検証します。次元だけでは Qwen3.5-27B と Qwen3.8-27B を区別できません。GGUF 内の識別情報は申告値であり、暗号学的に認証しません。GGML と外部依存の現在のプリミティブとインターフェースは維持します。

<a id="modules"></a>

## モジュール

| モジュール | パス | 役割 | 依存先 | 利用元 |
|---|---|---|---|---|
| ポリシー | `backend/src/dgs-model-policy.*` | Qwen 許可リスト | GGUF, hparams | モデルローダー |
| Qwen | `backend/src/models/` | 密な計算グラフ | DeltaNet, GGML | サーバー |
| 視覚 | `backend/tools/mtmd/` | Qwen 画像・動画 | GGML、コーデック | サーバー |
| チャット | `backend/common/` | Jinja、XML ツール、サンプリング | モデル、vendor | サーバー |
| HTTP | `backend/tools/server/` | ルート、スロット、ツール | common, mtmd, UI | 実行ファイル |
| Web | `frontend/` | 現在の UI と保存形式 | 固定バージョンの npm パッケージ | ブラウザー |
| ビルド | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | ビルドと埋め込み | コンパイラー、npm、Python | デプロイ |
| シェル MCP | backend/tools/server/server-shell-mcp.* | MCP コマンドとパスワード要求 | HTTP, POSIX, common_json | サーバー、MCP Web UI |
| モニター | backend/tools/server/server-monitor.* | CPU/GPU テレメトリーとメモリ集計 | procfs、sysfs、任意の NVML | server-context、ルーター、Web |
| MTP の重み | backend/src/dgs-ext.h; dgs-model.cpp | 密な重みと MTP の割り当て分類 | GGML バッファーメタデータ | モニターのスナップショット |
| 共有設定 | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | 原子的に保存する設定文書 | ファイルシステム、API、SettingsService | 全 Web 状態ストア |
| API 文書 | backend/tools/server/server-api-doc.*; frontend/api.html | 現在の OpenAPI と Swagger | ルート登録簿、スキーマ、vendor/swagger-ui | /api/doc/ |
| 言語 | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 言語のカタログと回答指示 | JSON、共有設定 | Web コンポーネント、チャットサービス |
| 待ち行列通知 | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | 明示要求時のキューイベント | タスクと応答のキュー | ストリーミングチャット UI |
| Web リソース | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | UI のビルドと埋め込み | npm, Python | `dgs-ui` |
| ライセンスの配布 | `backend/CMakeLists.txt`, `build/licenses/` | プロジェクトと第三者の条件を分離 | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## 主要シンボル

| シンボル | ファイル:行 | 機能 |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | GGUF 識別情報の検証 |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | 密なモデルの次元検証 |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | MoE テンソルの拒否 |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | 許可モデルの生成 |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Qwen 次元の読み込み |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Jinja のレンダリング |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Qwen 視覚グラフの生成 |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | サーバー起動とルート登録 |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | MCP JSON-RPC の振り分け |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | シェル呼び出しの検証と実行 |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | sudo パスワード要求の転送 |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | ハードウェアとモデルメモリの集計 |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | 密なメモリ区分の公開 |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | 追加 MTP 重みの集計 |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | 共有設定の原子的な保存 |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | OpenAPI と Swagger の提供 |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | キュー内位置の通知 |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | 地域言語の回答指示を追加 |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | 使用済みパスワード文字列の消去 |

<a id="flows"></a>

## 呼び出しフロー

読込：`fDgsModelCreate` → MoE 拒否 → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → テンソル/グラフ。チャット：HTTP → キュー/スロット → Jinja → トークン化 → 推論 → ストリーミング応答。Web：移行なしで `DgsUi` スキーマを直接初期化。ビルド：npm → `frontend/dist` → `frontend/embed.py` → 埋め込みライブラリー。ドラフトモデルにも同じポリシーが適用され、MTP は許可モデルの予測層を使います。

<a id="entrypoints"></a>

## 入口

| ルート | ハンドラー | ファイル |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | 現在の補完ルート | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | 現在の API ルート | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | サーバー/ルーター | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | 埋め込み Web UI | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | スナップショットとルーター集計 | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | プレフィックスと保存済み言語 | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## 影響

ポリシー変更はドラフトや語彙のみの読込を含めすべてに影響します。GGUF メタデータの改名は識別を壊す可能性があり、ファイル名で代用してはいけません。現在の GGUF キーや共通プリミティブに残る歴史的な名前は旧モデルの有効化を意味しません。削除済み API、移行、別名を戻さないでください。スキーマ変更には前後端の協調更新が必要です。

<a id="extension"></a>

## 拡張

モデル追加にはアーキテクチャ、メタデータ、次元を含む許可リストの明示的な決定が必要です。名前だけで対象を広げたり MoE を再導入したりしないでください。新しいエンドポイントを登録し、Web 型も更新します。15 言語すべての README、MANUAL、CODE を同期して保守してください。

<a id="validation"></a>

## 情報源と検証

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

上記の公式設定がプロファイルを定義します。検証はコンパイルと静的解析のみで、規則によりテストは禁止されています。推論やスループット測定は行っていません。保護された `.git` は所有者が手動で削除し、エージェントは Git を使いません。



<a id="integrated-features"></a>

## MCP、モニター、共有 UI

シェル：CLI フラグ → 検出 → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → 権限確認 → `tools/call` → 独立プロセスグループ → JSON 結果と呼び出し別 JSONL → stdin/stdout/stderr 表示。askpass の別経路は呼び出し ID と私有 Unix ソケットを使用し、UI はパスワードをモデルのメッセージへ入れません。`common_json::mWipeString` は使用後にヘルパー内の機密 JSON 文字列を上書きします。

モニター：読込 → MTP 重みの分類 → 推論中は毎秒最大 1 回の主ループスナップショット → mutex 保護のコピー → `/api/monitor*` → 表示中のページが 2 秒ごとに照会。主モデルと別ドラフトの重みは一度だけ数えます。エキスパートのバッファー、キャッシュ、プリフェッチ区分は含みません。照会は推論タスクを登録せず、モデルも起こしません。ハードウェア使用率は推論スナップショットと独立しています。

設定：ブラウザー API キー → `/api/ui/settings` → `UiSettingsStore.mLoad` → ストア初期化 → 最後の変更から短く待って文書全体を保存。文書には `config`、`userOverrides`、`language`、`preferences` があり、各値は担当ストアのシリアライズ方法を保持します。共有は再読み込み時に反映され、リアルタイム共同編集ではありません。最後の文書全体の保存が優先されます。会話、タブ、再開カーソルはローカルです。

言語：`/api/config.js` → JSON カタログ選択 → UI。`fApplyResponseLanguage` は送信メッセージのコピーだけを変更し、チャット、ツール、タイトル、事前入力が使います。追加 DGS 文字列は en-US/en-GB/es-ES/es-AR を含み、他のロケールでは導入済みカタログに項目がなければ en-US に戻ります。

キュー：`dgs_client=webui` → 補完スキーマ → 送信元別カウンター → `server_task_result_queue` → `dgs_queue` SSE → 会話別 `QueueStore` → 応答待ちラベル。OpenAI Responses と Anthropic のイベント形式には追加イベントを送らず、未指定の API クライアントには影響しません。

API ベースは既定で `/api`、設定可能なのは `/api` 以下だけです。UI は `/` のままです。ハンドラーは相対パスを受け、ルーター子プロセスは常に `/api` を使い、Web が公開プレフィックスを解決します。API は service worker キャッシュではなくネットワークにアクセスします。パス変更はルーター転送、クライアント URL、OpenAPI を同時に更新する必要があります。

拡張：シェル候補とフロントエンドのツール名を同時に追加し、メモリ区分は密なスナップショット・API 型・色グループを合わせて更新します。文言は JSON カタログへ追加します。許可リストと現在形式を維持し、変更後は両側をコンパイルしてください。規則に従いテストを追加・実行しないでください。


統合機能の検証：CPU Release、LTO、70 個の埋め込み Web リソースのビルドに成功。Svelte/TypeScript はエラー・警告とも 0。Web バンドルは推奨サイズを超え、GCC/CMake は既存の警告とファイルシステムの時刻ずれも報告しました。テスト、サーバー起動、推論、GPU ビルドは行っていません。

<a id="repository-organization"></a>

## リポジトリ構成

```text
backend/
  CMakeLists.txt
  cmake/  common/  ggml/  include/  src/  vendor/
  tools/
    CMakeLists.txt
    mtmd/
    server/
build/
  CMakeLists.txt
  build-local.sh
  licenses/
deploy/
  scripts/ui-assets.cmake
doc/
  CODE*.md  MANUAL*.md  README.*.md
frontend/
  CMakeLists.txt  sources.cmake  embed.py
  src/  static/  scripts/
_/build/bin/dgs-server
```

ビルド：`build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd、フロントエンド、サーバー。UI：`frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → 埋め込みライブラリー。ラッパーとバックエンドはどちらも CMake ソースルートになれます。移動時は include、CMake パス、依存一覧、文書を同時に更新します。


構成の検証：build/ と backend/ の両方から CMake 設定に成功。build/ から CPU Release、LTO、埋め込み UI をビルドできました。Svelte/TypeScript はエラー・警告 0。Bash 構文と文書のローカルリンクを確認しました。テストや推論は実行していません。


<a id="license"></a>

## ライセンスと第三者の表示

NiPeGun による貢献には [PolyForm Noncommercial 1.0.0](../LICENSE) が適用されます。組み込まれた第三者のコードには元のライセンスと著作権表示が引き続き適用され、[build/licenses/](../build/licenses/README.md) にまとめられています。

メイン LICENSE は原文のまま維持します。特に ggml 作者の MIT 表示は `build/licenses/LICENSE-llama.cpp` に残し、Swagger UI の Apache ライセンスと NOTICE も保持します。一覧には、組み込まれた HTTP、JSON、ハッシュ、メディア、subprocess、base64、CPU カーネル、数学ライブラリーの表示も含みます。依存ソースや Web バンドル内の表示も引き続き適用されます。モデルの重みとシステムライブラリーには独自の条件があります。

既定のデータ配置では、`cmake --install _/build --prefix PREFIX` がプロジェクト LICENSE と `build/licenses/` を `PREFIX/share/doc/dgs-server/licenses/` にコピーします。既定のプレフィックス `/usr/local` では `/usr/local/share/doc/dgs-server/licenses/` に配置され、ユーザー別には `~/.local` も指定できます。`Licenses` コンポーネントは単独でもインストールできます。実行ファイルを手動配布する場合もこれらの文書を同梱してください。

配布：プロジェクト LICENSE と第三者表示 → CMake `Licenses` コンポーネント → `${prefix}/share/doc/dgs-server/licenses/`。依存を変更するときは該当表示を確認します。製品名と API 型には DGS を使い、第三者の帰属表示や GGUF 識別子は改名しません。未使用の外部カタログアダプターと未対応モデルのプリセットは削除され、モデル検索には既存の Hugging Face サービスを使います。

識別とライセンスの検証：CPU Release、LTO、埋め込み UI のビルドに成功し、Svelte/TypeScript のエラー・警告は 0。生成された CMake インストール規則には主ライセンスと収集済み表示が含まれます。ルート LICENSE は参照とバイト単位で一致します。テスト、推論、本番インストールは行っていません。
