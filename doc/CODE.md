# DGS — code map

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · **en-US** · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server is an independent project developed and maintained by **NiPeGun**.

- [Architecture](#architecture)
- [Modules](#modules)
- [Symbols](#symbols)
- [Flows](#flows)
- [Entrypoints](#entrypoints)
- [Impact](#impact)
- [Extension](#extension)
- [MCP, monitor and shared UI](#integrated-features)
- [Sources and validation](#validation)
- [Repository organization](#repository-organization)

- [License and third-party notices](#license)

<a id="architecture"></a>

## Architecture

One server and one web application. The factory only instantiates `dgs_model_qwen35`; Qwen3.8-27B shares that architecture. DeltaNet kernels and the Qwen projector with its shared vision helper are retained. MoE, old GGUF versions and other profiles are rejected. The allowlist reads original metadata independently of HTTP aliases, then checks layers, embedding, FFN and heads. Dimensions alone cannot distinguish Qwen3.5-27B from Qwen3.8-27B. Identity is asserted by the GGUF, not cryptographically authenticated. GGML and external dependencies retain their current primitives and interfaces.

<a id="modules"></a>

## Modules

| Module | Path | Responsibility | Depends on | Used by |
|---|---|---|---|---|
| Policy | `backend/src/dgs-model-policy.*` | Qwen allowlist | GGUF, hparams | model loader |
| Qwen | `backend/src/models/` | Dense graph | DeltaNet, GGML | server |
| Vision | `backend/tools/mtmd/` | Qwen image/video | GGML, codecs | server |
| Chat | `backend/common/` | Jinja, XML tools, sampling | model, vendor | server |
| HTTP | `backend/tools/server/` | Routes, slots, tools | common, mtmd, UI | executable |
| Web | `frontend/` | current UI/storage | pinned npm packages | browser |
| Build | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | Compile and embed | compiler, npm, Python | deployment |
| Shell MCP | backend/tools/server/server-shell-mcp.* | MCP command execution and password prompts | HTTP, POSIX, common_json | server, MCP web |
| Monitor | backend/tools/server/server-monitor.* | CPU/GPU telemetry and memory aggregation | procfs, sysfs, optional NVML | server-context, router, web |
| MTP weights | backend/src/dgs-ext.h; dgs-model.cpp | Dense/MTP allocation split | GGML buffer metadata | monitor snapshots |
| Shared settings | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | Atomic settings document | filesystem, API, SettingsService | all web stores |
| API docs | backend/tools/server/server-api-doc.*; frontend/api.html | Live OpenAPI and Swagger | route registry, schema, vendor/swagger-ui | /api/doc/ |
| Languages | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 catalogs and reply instruction | JSON, shared settings | web components, chat service |
| Queue notices | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | Opt-in queue events | task and response queues | chat streaming UI |
| Web assets | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | Build and embed UI | npm, Python | `dgs-ui` |
| License distribution | `backend/CMakeLists.txt`, `build/licenses/` | Keep project and third-party terms separate | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## Symbols

| Symbol | File:line | Purpose |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | Validate GGUF identity |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | Validate dense dimensions |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | Reject MoE tensors |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | Create allowed model |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Load Qwen dimensions |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Render Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Build Qwen vision graph |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | Start server and register routes |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | Dispatch MCP JSON-RPC |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | Validate and execute shell call |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | Forward sudo password prompts |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | Aggregate hardware and model memory |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | Publish dense memory segments |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | Count attached MTP weights |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | Save shared settings atomically |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | Serve OpenAPI and Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | Emit queue position |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | Add regional response instruction |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | Erase consumed password strings |

<a id="flows"></a>

## Flows

Load: `fDgsModelCreate` → MoE rejection → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → tensors/graph. Chat: HTTP → queue/slot → Jinja → tokenization → inference → streamed response. Web: direct initialization of the `DgsUi` schema without migrations. Build: npm → `frontend/dist` → `frontend/embed.py` → embedded library. Draft models use the same policy; MTP uses the allowed model’s prediction layers.

<a id="entrypoints"></a>

## Entrypoints

| Route | Handler | File |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | current completion routes | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | current API routes | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | server/router | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | embedded web | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | snapshots + router aggregation | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | Prefix and saved language | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## Impact

Policy changes affect every load, including drafts and vocabulary-only loads. Renaming GGUF metadata can prevent identification; do not substitute filenames. Historical names that remain current GGUF keys or shared primitives do not enable older models. Do not restore removed APIs, migrations or aliases. Schema changes require coordinated frontend/backend changes.

<a id="extension"></a>

## Extension

Adding another model requires an explicit allowlist decision covering architecture, metadata and dimensions. Do not broaden matching by name alone or reintroduce MoE. Register new endpoints in the server and update web types. Keep README, MANUAL and CODE current in all 15 interface languages.

<a id="validation"></a>

## Sources and validation

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

The official configurations above define the profiles. Validation uses compilation and static analysis only: project rules prohibit tests. No model inference or throughput measurements were run. The protected `.git` directory is removed manually by the owner; the agent does not use Git.



<a id="integrated-features"></a>

## MCP, monitor and shared UI

Shell: CLI flag → shell discovery → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → permission gate → `tools/call` → independent process group → JSON result and per-call JSONL → stdin/stdout/stderr renderer. The askpass side channel uses a call id and a private Unix socket; the web dialog never includes passwords in model messages. `common_json::mWipeString` overwrites helper JSON strings when consumed.

Monitor: model load → MTP weight allocation split → main-loop memory snapshot (at most once per second during inference) → mutex-protected snapshot copy → `/api/monitor*` → visible-page polling every two seconds. Target and separate draft weights are counted once. No expert buffers, caches or prefetch categories are compiled into this monitor. Polling does not post inference tasks or wake sleeping models. Hardware usage is independent of the inference snapshot.

Settings: browser API key → `/api/ui/settings` → `UiSettingsStore.mLoad` → existing store initialization → debounced whole-document save. The current document has `config`, `userOverrides`, `language` and `preferences`; preference values preserve their owning store's serialization. Settings are shared between browsers on reload, not live collaborative edits; the last whole-document save wins. Conversations, tabs and resumable-stream cursors remain local.

Language: `/api/config.js` → JSON catalog selection → UI. `fApplyResponseLanguage` changes only outgoing message copies; ordinary chat, tools, titles and prefill all use it. Added DGS UI strings include en-US/en-GB/es-ES/es-AR; other selected locales fall back to en-US where their imported catalog has no entry.

Queue: `dgs_client=webui` → completion schema → origin counters → `server_task_result_queue` → `dgs_queue` SSE → per-conversation `QueueStore` → pending response label. OpenAI Responses and Anthropic event grammars do not receive these extra events. Unmarked API clients are unaffected.

API base: `/api` by default, configurable only below `/api`; UI stays at `/`. Handlers receive relative paths, router children always use `/api`, and the web resolves the published prefix. API calls use the network rather than the service-worker cache. Changes to these paths must update router forwarding, client URL construction and OpenAPI together.

Extension: add a shell candidate and its frontend tool name together; add monitor segments in the dense snapshot, API type and color groups together; add UI text keys to JSON catalogs. Preserve the existing dense allowlist and current formats. Compile backend and web after changes; do not add or run tests under the project rules.


Validation of the integrated features: CPU Release build with LTO and 70 embedded web assets succeeded. Svelte/TypeScript: zero errors and zero warnings. The generated UI bundle exceeds the build size advisory; GCC/CMake also reported existing warnings and filesystem clock skew. No tests, server execution, model inference or GPU builds were performed.

<a id="repository-organization"></a>

## Repository organization

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

Build flow: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, frontend, server. UI flow: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → embedded library. The wrapper and the backend can both be CMake source roots. A relocation must update includes, CMake source paths, dependency lists and documentation together.


Layout validation: CMake configuration succeeded from both build/ and backend/. CPU Release compilation with LTO and the embedded web succeeded through the build/ entry point. Svelte/TypeScript reported zero errors and zero warnings; Bash syntax and local documentation links were checked. No tests or model inference were run.


<a id="license"></a>

## License and third-party notices

NiPeGun's contributions are licensed under [PolyForm Noncommercial 1.0.0](../LICENSE). Incorporated third-party code retains its own licenses and copyright notices, collected in [build/licenses/](../build/licenses/README.md).

The main LICENSE is kept verbatim. In particular, the MIT notice of the ggml authors remains in `build/licenses/LICENSE-llama.cpp`; Swagger UI retains its Apache license and NOTICE. The license inventory also includes the bundled HTTP, JSON, hashing, media, subprocess, base64, CPU-kernel and mathematics components. Notices embedded in dependency source files or web bundles remain applicable. Model weights and system libraries have their own terms.

`cmake --install _/build --prefix PREFIX` copies the project LICENSE and `build/licenses/` into `PREFIX/share/doc/dgs-server/licenses/` with the default data layout. The default `/usr/local` prefix therefore installs them at `/usr/local/share/doc/dgs-server/licenses/`; a user prefix can use `~/.local`. The `Licenses` component can be installed separately. Include these documents when distributing the executable manually.

Distribution flow: project LICENSE + third-party notices → CMake `Licenses` component → `${prefix}/share/doc/dgs-server/licenses/`. Changing a dependency requires reviewing its corresponding notice. Product naming and API property types use DGS; third-party attributions and GGUF identifiers are not renamed. The unused external catalog adapter and unsupported model presets were removed; model discovery continues through the existing Hugging Face service.

Validation of project identity and licensing: CPU Release compilation with LTO and embedded UI succeeded; Svelte/TypeScript reported zero errors and zero warnings. The generated CMake installation includes the root license and collected notices. The root LICENSE matches the reference byte-for-byte. No tests, model inference or production installation were run.
