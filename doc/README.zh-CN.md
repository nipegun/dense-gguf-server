# dense-gguf-server

本 README 还提供以下语言版本：\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), **zh-CN**

## 项目简介

通过网页聊天和 HTTP 推理 API 在本地运行稠密 GGUF 模型。

类似 llama.cpp，但以 NiPeGun 式手术刀般的精度精心打造，力求实现尽可能快的推理速度。

## 截图

尚未添加截图。

## 部署

在 **Debian** 上完成构建后，从项目根目录以 `root` 身份执行：

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

打开 `http://127.0.0.1:8080`。API 使用 `/api`，兼容 OpenAI 的客户端使用 `/api/v1`，Swagger 位于 `/api/doc/`。模型权重需另行提供；允许其他客户端访问时请使用 `--api-key`。

## 构建

首先在 Debian 13 上以 `root` 身份安装依赖：

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

然后使用普通用户在项目目录中执行：

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

脚本将本地源码构建到 `_/build/`。`DGS_BUILD_JOBS=8` 设置并行任务数；其他参数传给 CMake，例如安装相应 SDK 后使用 `-DGGML_CUDA=ON` 或 `-DGGML_VULKAN=ON`。[构建依赖和选项](MANUAL.zh-CN.md#local-build)必须预先准备好。

## 阅读文档

请参阅：

- [doc/MANUAL.zh-CN.md](MANUAL.zh-CN.md)：构建、安装和使用 DGS，配置 MCP 与网页，以及排查问题。
- [doc/CODE.zh-CN.md](CODE.zh-CN.md)：稠密推理引擎、源码组织及后端与前端的构建约定。

## 赞助本项目

- 可**[委托我](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)**添加新功能。
- **GitHub Sponsors。**也可以在 GitHub 上提供资金支持：[github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)。
