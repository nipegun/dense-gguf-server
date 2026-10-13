# dense-gguf-server

この README は以下の言語でもご覧いただけます：\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), **ja-JP**, [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## 概要

Web チャットと HTTP 推論 API を通じて、密な GGUF モデルをローカルで実行します。

llama.cpp のように、ただし可能な限り最速の推論を実現するため、NiPeGun 流のメスさばきの精密さで作り上げています。

## スクリーンショット

スクリーンショットはまだ追加されていません。

## デプロイ

**Debian** でビルドした後、プロジェクトのルートディレクトリから `root` として次のコマンドを実行します。

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

`http://127.0.0.1:8080` を開きます。API は `/api`、OpenAI 互換クライアントは `/api/v1` を使用し、Swagger は `/api/doc/` にあります。モデルの重みは別途用意します。他のクライアントにアクセスを許可する場合は `--api-key` を使用してください。

## ビルド

まず Debian 13 上で `root` として依存パッケージをインストールします。

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

続いて、通常のユーザーでプロジェクトディレクトリから実行します。

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

スクリプトはローカルのソースを `_/build/` にビルドします。並列ジョブ数は `DGS_BUILD_JOBS=8` で指定します。追加の引数は CMake に渡されます。たとえば、対応する SDK があれば `-DGGML_CUDA=ON` または `-DGGML_VULKAN=ON` を指定できます。[ビルドの依存関係とオプション](MANUAL.ja-JP.md#local-build)を事前に準備してください。

## ドキュメント

次の文書を参照してください。

- [doc/MANUAL.ja-JP.md](MANUAL.ja-JP.md)：DGS のビルド、インストール、操作、MCP と Web UI の設定、問題の診断。
- [doc/CODE.ja-JP.md](CODE.ja-JP.md)：密な推論エンジン、ソース構成、バックエンドとフロントエンドのビルド契約。

## プロジェクトを支援する

- 新しい機能の開発は **[こちらからご依頼ください](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)**。
- **GitHub Sponsors。** GitHub で金銭的に支援することもできます：[github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)。
