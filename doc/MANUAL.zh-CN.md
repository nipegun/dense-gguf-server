# DGS 使用手册

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · [ja-JP](MANUAL.ja-JP.md) · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · **zh-CN**

Dense GGUF Server 是由 **NiPeGun** 开发和维护的独立项目。

DGS 提供以下稠密模型的推理服务及配套网页聊天应用：

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

其他语言模型均不受支持，包括所有 GLM 和 Kimi 版本。MoE 模型仍被排除。服务器检查原始 GGUF 身份及各层维度，加载推测解码的草稿模型时也执行这些检查。允许模型的量化版本仍受支持。

<a id="local-build"></a>

## 在 Debian 上构建和运行

使用 Debian 13 或兼容系统，并准备 C++17、CMake 3.19+、Python 3.13+，以及 Node.js 20.19+ 或 22.12+。以 root 身份安装软件包：

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

打开 `http://127.0.0.1:8080`。权重需另行提供。处理图像和视频时，通过 `--mmproj` 指定匹配的 Qwen 投影器；视频处理还需要 FFmpeg。不支持音频模型或音频生成。

默认启用 Release 模式、本机 CPU 内核以及平台支持的链接时优化。构建 CUDA 或 Vulkan 后端时，安装对应 SDK 并使用 `-DGGML_CUDA=ON` 或 `-DGGML_VULKAN=ON`。应针对部署机器的 CPU 构建，或使用 `-DGGML_NATIVE=OFF`。尚未测量吞吐量提升。

## 仅支持当前格式

只接受 GGUF v3。`general.name`，或 `general.basename` 与 `general.size_label` 的组合，必须标识允许的模型；仅文件名符合要求并不足够。这是元数据和维度检查，并非对权重进行身份认证。即使架构维度相同，也不会将 Qwen3.5-27B 文件当作 Qwen3.8-27B 接受。

聊天需要模型的完整 Jinja 模板，或通过 `--chat-template-file` 提供模板。旧的固定模板引擎、`--jinja`/`--no-jinja` 开关、废弃的 C API、旧版长选项别名及旧推理模式已移除。API 保留当前的 `/api/v1/*` 约定；旧别名 `/completion` 和 `/embedding` 已删除。

网页使用当前的 `DgsUi` 存储命名空间和结构化消息，不读取或迁移旧命名空间、主题、消息标记或旧 JSON 对话导出。支持当前 JSONL 会话导出及 ZIP 压缩包，不会删除浏览器中的旧数据。MCP 使用 StreamableHTTP 或 WebSocket；旧 SSE 传输及自动回退已移除。剪贴板操作需要安全上下文中的当前 Clipboard API。

## 目录与文档

- `backend/`：服务器、Qwen 推理、共享张量内核和 Qwen 视觉处理。
- `frontend/`：当前 Svelte 应用及锁定版本的依赖。
- `deploy/`：构建辅助工具和本地网页资源嵌入。
- [技术地图](CODE.zh-CN.md)和[使用手册](MANUAL.zh-CN.md)。

常规构建会编译并嵌入本地网页。若要复用已通过 `cd frontend && npm ci && npm run build` 构建的网页，可设置 `-DDGS_USE_LOCAL_UI=ON`。修改源码后应重新生成资源。`-DDGS_BUILD_UI=OFF` 会省略嵌入资源；此时使用 `--path` 指向已构建的当前网页目录。

项目规则禁止测试。验证仅限编译和静态类型检查；未执行模型推理或基准测试。此项目不使用 Git；受保护的 `.git` 目录由其所有者手动删除。

## 日常使用

在网页中新建对话，选择已加载且允许的模型，然后发送文本或支持的图像、视频附件。设置控制采样、上下文和工具。`-c` 设置上下文长度，`-t` 设置 CPU 线程数，`-ngl auto` 启用自动 GPU 放置。需要认证时使用 `--api-key`。兼容客户端连接到 `http://127.0.0.1:8080/api/v1` 和 `/api/v1/chat/completions`。模型目录可以含有其他文件，但加载器会拒绝不支持的检查点。

加载失败时，检查模型身份、架构、维度和 GGUF 版本；不要通过修改另一模型的标签绕过验证。内存不足时，减小上下文、批大小或 GPU 卸载量。网页缺失时，重新构建网页或提供 `--path`。

## MCP 与网页功能

通过 `--local-shell-mcp-server` 或 `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1` 启用内置 shell MCP 服务器。网页会自动发现它，并在调用工具前请求授权。它提供已安装的 Bash、Zsh、sh 和 PowerShell 解释器，显示命令、stdout 和 stderr，并将每次调用记录为 JSONL。

网页包括 CPU/GPU 与内存监控、状态栏、服务器端共享设置、15 种界面语言、按所选地区语言回复的可选指令，以及区分网页和其他 API 客户端请求的排队提示。保留模型浏览、下载、对话标签页和稠密模型限制。

网页位于 `/`，API 默认位于 `/api` 下。兼容 OpenAI 的客户端必须将 `/api/v1` 作为基础路径。Swagger 文档位于 `/api/doc/`。新 API 路径不提供根路径别名。`--api-prefix` 可选择 `/api` 下的其他路径，由 `/api/config.js` 告知网页。

## 本地 shell MCP

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

打开网页并输入该 API 密钥。服务器启用后，`local shell mcp server` 条目自动出现，可全局禁用或按对话禁用。工具为 `bash.run`、`zsh.run`、`sh.run` 和 `psh.run`，仅提供已安装的解释器。端点为 `POST /api/mcp/shell`，使用 MCP Streamable HTTP、JSON 响应和 `2025-11-25` 协议版本。

每个工具接受 `command`，以及可选的 `cwd` 和 `timeout`。超时单位为秒，默认 60，最大 3600。每次调用在独立进程中运行，不提供交互式终端。工作目录和环境变更不会跨调用保留。取消或超时将终止整个进程组。命令以服务器账户权限运行；应使用 API 密钥保护访问。

聊天分别显示 stdin（命令）、stdout、stderr 和退出码。显示设置可自动展开这些区块。`~/.dgs/local-shell-mcp-server/` 下的日志保留完整命令和输出，每次调用对应一个带时间戳的 JSONL 文件；聊天可能截断过长输出。文件仅供服务器账户访问，不会自动轮转。

在已安装 sudo 的 Linux 系统上，密码请求会出现在输入框已聚焦的对话框中，并提供接受和取消按钮。密码仅发送给辅助程序，不发送给模型，也不写入对话或日志。等待用户时会暂停命令计时；五分钟未回答的请求将过期。本项目不会安装 sudo，生产运行也不需要它。

## 监控与共享偏好

点击侧栏的**监控**或底部状态栏。页面可见时，CPU 和 GPU 使用率每两秒更新一次。完整视图还显示驱动提供的 RAM、VRAM、温度和功耗。内存分为稠密权重、MTP/草稿权重、KV/循环状态缓存、计算缓冲区、已保存的提示词/检查点、其他使用量和空闲量。NVIDIA 动态使用 NVML，AMD 使用 sysfs；无法取得的数值保持未知。路由器模式汇总已加载的子进程，不会加载或唤醒模型。

设置保存在服务器账户的 `~/.dgs/config/ui-settings.json`，权限为 0600。访问同一服务器的浏览器共享配置、主题、语言、收藏和工具偏好。每次保存替换整个文档；多个浏览器同时编辑时，以最后一次成功保存为准。API 密钥、对话及临时导航和流状态仍保存在各浏览器本地，不迁移旧浏览器设置。

在“常规”设置中选择语言。更改会保存选择并重新加载界面，请先保存其他未提交的修改。支持的地区语言为 `de-DE`、`en-GB`、`en-US`、`es-AR`、`es-ES`、`fr-FR`、`he-IL`、`hi-IN`、`it-IT`、`ja-JP`、`ko-KR`、`pt-BR`、`pt-PT`、`ru-RU`、`zh-CN`。美式英语是默认语言，也是未翻译文本的回退语言。启用**使用界面语言回复** (Respond in the interface language)可为之后的聊天请求、工具轮次、生成标题及 KV 预填充添加地区语言指令。不会翻译已有消息；是否遵循指令取决于模型。

推理请求等待空闲槽位时，状态会显示前方请求数量，以及其中是否有直接来自 API 的请求。普通 API 客户端保留原有响应格式。只有标记了 `"dgs_client": "webui"` 的流式请求才会收到 `dgs_queue` 事件。

开发前端时，`npm run dev` 将 `/api` 代理到 `http://localhost:9931`；设置 `VITE_PUBLIC_SERVER_ORIGIN` 可使用其他服务器。生产构建嵌入网页及本地 Swagger 查看器，无需外部文档 CDN。

## 源码与构建目录

`build/` 包含 CMake 入口和 `build-local.sh`。脚本根据自身位置定位项目，配置 Release 构建，并接受其他 CMake 选项。依赖必须预先安装。`DGS_BUILD_JOBS` 默认值为 4。手动配置使用 `cmake -S build -B _/build`；`backend/CMakeLists.txt` 定义引擎项目。生成文件和 `bin/dgs-server` 位于 `_/build/`，与构建脚本分开存放。

编译后，以 root 身份使用 `cmake --install _/build --prefix /usr/local` 安装到生产环境。网页准备规则位于 `deploy/scripts/ui-assets.cmake`，`frontend/sources.cmake` 列出其输入，`frontend/embed.py` 生成嵌入的 C++ 资源。主 README 位于根目录，地区语言版本位于 `doc/`。


<a id="license"></a>

## 许可证与第三方声明

NiPeGun 的贡献采用 [PolyForm Noncommercial 1.0.0](../LICENSE) 许可。集成的第三方代码保留各自的许可证和版权声明，集中存放于 [build/licenses/](../build/licenses/README.md)。

主 LICENSE 保持逐字不变。其中，ggml 作者的 MIT 声明保存在 `build/licenses/LICENSE-llama.cpp`；Swagger UI 保留其 Apache 许可证和 NOTICE。清单还包括集成的 HTTP、JSON、哈希、多媒体、subprocess、base64、CPU 内核和数学组件。依赖源码及网页包中的声明仍然适用。模型权重和系统库各有自己的条款。

在默认数据目录布局下，`cmake --install _/build --prefix PREFIX` 将项目 LICENSE 和 `build/licenses/` 复制到 `PREFIX/share/doc/dgs-server/licenses/`。默认前缀 `/usr/local` 对应安装路径 `/usr/local/share/doc/dgs-server/licenses/`；用户级安装可使用 `~/.local`。`Licenses` 组件可单独安装。手动分发可执行文件时也应附上这些文件。
