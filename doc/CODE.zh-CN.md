# DGS — 代码地图

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · **zh-CN**

Dense GGUF Server 是由 **NiPeGun** 开发和维护的独立项目。

- [架构](#architecture)
- [模块](#modules)
- [关键符号](#symbols)
- [调用流程](#flows)
- [入口](#entrypoints)
- [影响分析](#impact)
- [扩展](#extension)
- [MCP、监控与共享界面](#integrated-features)
- [来源与验证](#validation)
- [仓库组织](#repository-organization)

- [许可证与第三方声明](#license)

<a id="architecture"></a>

## 架构

项目包含一个服务器和一个网页应用。工厂仅实例化 `dgs_model_qwen35`；Qwen3.8-27B 也使用该架构。保留 DeltaNet 内核、Qwen 投影器及共享视觉辅助代码。拒绝 MoE、旧 GGUF 版本和其他配置档案。允许列表独立于 HTTP 别名读取原始元数据，然后检查层、嵌入、FFN 和注意力头。仅凭维度无法区分 Qwen3.5-27B 和 Qwen3.8-27B。GGUF 中的身份声明不经过密码学认证。GGML 和外部依赖保留其当前原语及接口。

<a id="modules"></a>

## 模块

| 模块 | 路径 | 职责 | 依赖 | 使用方 |
|---|---|---|---|---|
| 模型策略 | `backend/src/dgs-model-policy.*` | Qwen 允许列表 | GGUF, hparams | 模型加载器 |
| Qwen | `backend/src/models/` | 稠密计算图 | DeltaNet, GGML | 服务器 |
| 视觉 | `backend/tools/mtmd/` | Qwen 图像/视频 | GGML、编解码器 | 服务器 |
| 聊天 | `backend/common/` | Jinja、XML 工具、采样 | 模型、vendor | 服务器 |
| HTTP | `backend/tools/server/` | 路由、槽位、工具 | common, mtmd, UI | 可执行文件 |
| 网页 | `frontend/` | 当前界面及存储 | 锁定版本的 npm 包 | 浏览器 |
| 构建 | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | 编译与嵌入 | 编译器、npm、Python | 部署 |
| Shell MCP | backend/tools/server/server-shell-mcp.* | MCP 命令执行与密码请求 | HTTP, POSIX, common_json | 服务器、MCP 网页 |
| 监控 | backend/tools/server/server-monitor.* | CPU/GPU 遥测与内存汇总 | procfs、sysfs、可选 NVML | server-context、路由器、网页 |
| MTP 权重 | backend/src/dgs-ext.h; dgs-model.cpp | 稠密/MTP 内存分配划分 | GGML 缓冲区元数据 | 监控快照 |
| 共享设置 | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | 原子保存的设置文档 | 文件系统、API、SettingsService | 全部网页状态存储 |
| API 文档 | backend/tools/server/server-api-doc.*; frontend/api.html | 实时 OpenAPI 与 Swagger | 路由注册表、模式、vendor/swagger-ui | /api/doc/ |
| 语言 | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 个语言目录及回复指令 | JSON、共享设置 | 网页组件、聊天服务 |
| 排队提示 | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | 按需发送的队列事件 | 任务及响应队列 | 流式聊天界面 |
| 网页资源 | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | 编译并嵌入界面 | npm, Python | `dgs-ui` |
| 许可证分发 | `backend/CMakeLists.txt`, `build/licenses/` | 区分项目条款与第三方条款 | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## 关键符号

| 符号 | 文件:行号 | 作用 |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | 验证 GGUF 身份 |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | 验证稠密模型维度 |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | 拒绝 MoE 张量 |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | 创建允许的模型 |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | 加载 Qwen 维度 |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | 渲染 Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | 创建 Qwen 视觉计算图 |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | 启动服务器并注册路由 |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | 分发 MCP JSON-RPC |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | 验证并执行 shell 调用 |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | 转发 sudo 密码请求 |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | 汇总硬件及模型内存 |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | 发布稠密内存分段 |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | 统计附加 MTP 权重 |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | 原子保存共享设置 |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | 提供 OpenAPI 和 Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | 发送队列位置 |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | 添加地区语言回复指令 |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | 擦除已使用的密码字符串 |

<a id="flows"></a>

## 调用流程

加载：`fDgsModelCreate` → 拒绝 MoE → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → 张量/计算图。聊天：HTTP → 队列/槽位 → Jinja → 分词 → 推理 → 流式响应。网页：直接初始化 `DgsUi` 模式，不进行迁移。构建：npm → `frontend/dist` → `frontend/embed.py` → 嵌入库。草稿模型遵循相同策略；MTP 使用允许模型的预测层。

<a id="entrypoints"></a>

## 入口

| 路由 | 处理器 | 文件 |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | 当前补全路由 | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | 当前 API 路由 | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | 服务器/路由器 | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | 嵌入网页 | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | 快照与路由器汇总 | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | 前缀及已保存语言 | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## 影响分析

策略变更影响所有加载，包括草稿和仅词表加载。修改 GGUF 元数据名称可能导致无法识别模型；不能用文件名代替。仍作为当前 GGUF 键或公共原语使用的历史名称，不代表支持旧模型。不要恢复已删除的 API、迁移或别名。模式变更需要前后端同步修改。

<a id="extension"></a>

## 扩展

增加模型必须明确决定允许列表中的架构、元数据及维度，不能仅按名称扩大匹配，也不能重新引入 MoE。在服务器注册新端点并更新网页类型。README、MANUAL、CODE 必须在全部 15 种界面语言中同步维护。

<a id="validation"></a>

## 来源与验证

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

上方官方配置定义模型档案。验证仅使用编译与静态分析；项目规则禁止测试。未运行模型推理或吞吐量测量。受保护的 `.git` 目录由所有者手动移除，代理不使用 Git。



<a id="integrated-features"></a>

## MCP、监控与共享界面

Shell：CLI 选项 → 检测解释器 → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → 授权检查 → `tools/call` → 独立进程组 → JSON 结果及每次调用的 JSONL → stdin/stdout/stderr 展示。askpass 旁路使用调用 ID 和私有 Unix socket；网页对话框绝不把密码放入模型消息。`common_json::mWipeString` 在使用后覆写辅助程序中的敏感 JSON 字符串。

监控：加载模型 → MTP 权重分配划分 → 主循环内存快照，推理期间每秒最多一次 → 互斥锁保护的副本 → `/api/monitor*` → 页面可见时每两秒轮询。主模型与独立草稿的权重各计一次。监控不编入专家缓冲区、专家缓存或预取分类。轮询不提交推理任务，也不唤醒休眠模型。硬件使用率独立于推理快照。

设置：浏览器 API 密钥 → `/api/ui/settings` → `UiSettingsStore.mLoad` → 状态存储初始化 → 最后一次更改后短暂防抖，再保存整个文档。当前文档含 `config`、`userOverrides`、`language`、`preferences`；偏好保留所属状态存储的序列化方式。设置在重新加载后共享，并非实时协同编辑；最后一次完整保存生效。对话、标签页和流恢复游标仍在本地。

语言：`/api/config.js` → JSON 语言目录 → 界面。`fApplyResponseLanguage` 仅修改待发送消息的副本；普通聊天、工具轮次、标题和预填充都使用它。新增 DGS 文本涵盖 en-US/en-GB/es-ES/es-AR；其他语言在导入目录缺少条目时回退到 en-US。

队列：`dgs_client=webui` → 补全模式 → 按来源计数 → `server_task_result_queue` → `dgs_queue` SSE → 按对话保存的 `QueueStore` → 待处理响应标签。OpenAI Responses 和 Anthropic 的事件格式不接收这些额外事件；未标记的 API 客户端不受影响。

API 基础路径默认为 `/api`，只能配置到 `/api` 下；网页保持在 `/`。处理器接收相对路径，路由器的子进程始终使用 `/api`，网页解析发布的前缀。API 调用访问网络而非 service worker 缓存。修改路径时，必须同步更新路由器转发、客户端 URL 构造及 OpenAPI。

扩展时同时添加 shell 候选项及前端工具名称；内存分段同时更新稠密快照、API 类型和颜色组；界面文本加入 JSON 语言目录。保留当前稠密允许列表和格式。修改后编译后端及前端；按项目规则不新增或运行测试。


集成功能验证：CPU Release、LTO 和 70 项嵌入网页资源构建成功。Svelte/TypeScript 为零错误、零警告。网页包超过建议大小；GCC/CMake 还报告了原有警告和文件系统时钟偏差。未执行测试、启动服务器、运行推理或构建 GPU 后端。

<a id="repository-organization"></a>

## 仓库组织

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

构建流程：`build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd、前端、服务器。网页流程：`frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → 嵌入库。外层入口和后端均可作为 CMake 源码根目录。移动文件时必须同步调整 include、CMake 路径、依赖列表及文档。


目录验证：从 build/ 和 backend/ 配置 CMake 均成功；通过 build/ 完成 CPU Release、LTO 和嵌入网页构建。Svelte/TypeScript 为零错误、零警告；已检查 Bash 语法及本地文档链接。未运行测试或模型推理。


<a id="license"></a>

## 许可证与第三方声明

NiPeGun 的贡献采用 [PolyForm Noncommercial 1.0.0](../LICENSE) 许可。集成的第三方代码保留各自的许可证和版权声明，集中存放于 [build/licenses/](../build/licenses/README.md)。

主 LICENSE 保持逐字不变。其中，ggml 作者的 MIT 声明保存在 `build/licenses/LICENSE-llama.cpp`；Swagger UI 保留其 Apache 许可证和 NOTICE。清单还包括集成的 HTTP、JSON、哈希、多媒体、subprocess、base64、CPU 内核和数学组件。依赖源码及网页包中的声明仍然适用。模型权重和系统库各有自己的条款。

在默认数据目录布局下，`cmake --install _/build --prefix PREFIX` 将项目 LICENSE 和 `build/licenses/` 复制到 `PREFIX/share/doc/dgs-server/licenses/`。默认前缀 `/usr/local` 对应安装路径 `/usr/local/share/doc/dgs-server/licenses/`；用户级安装可使用 `~/.local`。`Licenses` 组件可单独安装。手动分发可执行文件时也应附上这些文件。

分发流程：项目 LICENSE 与第三方声明 → CMake `Licenses` 组件 → `${prefix}/share/doc/dgs-server/licenses/`。修改依赖时必须复核相应声明。产品名称和 API 属性类型使用 DGS；第三方署名及 GGUF 标识符不改名。已删除未使用的外部目录适配器和不支持的模型预设；模型发现继续使用现有 Hugging Face 服务。

身份及许可验证：CPU Release、LTO 和嵌入网页构建成功；Svelte/TypeScript 为零错误、零警告。生成的 CMake 安装规则包括主许可证及汇总声明。根 LICENSE 与参考文件逐字节一致。未执行测试、推理或生产安装。
