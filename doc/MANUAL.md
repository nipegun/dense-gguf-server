# DGS manual

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · **en-US** · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · [ja-JP](MANUAL.ja-JP.md) · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server is an independent project developed and maintained by **NiPeGun**.

DGS serves the following dense models and their associated chat web application:

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

All other language models, including every GLM and Kimi version, are unsupported. MoE models remain prohibited. The server checks the original GGUF identity and layer dimensions, including when loading a speculative draft. Quantizations of the allowed models remain supported.

<a id="local-build"></a>

## Build and run on Debian

Use Debian 13 or a compatible system with C++17, CMake 3.19+, Python 3.13+ and Node.js 20.19+ or 22.12+. Install packages as root:

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. Model weights are supplied separately. For vision/video, supply the matching Qwen projector with `--mmproj`; video processing additionally uses FFmpeg. Audio models and audio generation are unsupported.

Release mode, native CPU kernels and supported link-time optimization are enabled by default. CUDA and Vulkan builds use `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` with the corresponding SDK. Build for the deployment CPU, or use `-DGGML_NATIVE=OFF`. No throughput improvement has been measured.

## Current formats only

Only GGUF v3 is accepted. `general.name`, or `general.basename` together with `general.size_label`, must identify an allowed model; the filename alone is not enough. This is a metadata/dimension policy, not authentication of the model weights. A Qwen3.5-27B file is not accepted as Qwen3.8-27B merely because the architectures have the same dimensions.

Chat requires the model's complete Jinja template or `--chat-template-file`. The old fixed-template engine, `--jinja`/`--no-jinja` switch, obsolete C APIs, alternate long CLI spellings and deprecated reasoning mode have been removed. The API retains its current `/api/v1/*` contracts; the old `/completion` and `/embedding` aliases are gone.

The web uses the current `DgsUi` storage namespace and structured messages. It does not read or migrate old namespaces, themes, message markers or legacy conversation JSON exports. Current JSONL session exports and ZIP archives are supported. Old browser data is not deleted. MCP uses StreamableHTTP or WebSocket; the retired SSE transport and automatic fallback are removed. Clipboard operations require the current Clipboard API in a secure context.

## Layout and documentation

- `backend/`: server, Qwen inference, shared tensor kernels and Qwen vision.
- `frontend/`: current Svelte application and its pinned dependencies.
- `deploy/`: build helpers and local web embedding.
- [Technical map](CODE.md) and [manual](MANUAL.md).

The normal build compiles and embeds the local web. To reuse a web already built by `cd frontend && npm ci && npm run build`, configure with `-DDGS_USE_LOCAL_UI=ON`. Rebuild the assets after editing their sources. `-DDGS_BUILD_UI=OFF` omits embedded assets; use `--path` with a current built web directory in that mode.

Project rules prohibit tests. Validation is limited to compilation and static type checking; no model inference or benchmarks were executed. Git is not used for this project; removal of the protected `.git` directory is handled manually by its owner.

## Daily operation

Start a conversation in the web, select a loaded allowed model, and send text or supported image/video attachments. Settings control sampling, context and tools. `-c` sets context length, `-t` CPU threads and `-ngl auto` automatic GPU placement. Use `--api-key` when authentication is needed. Compatible clients connect to `http://127.0.0.1:8080/api/v1` and `/api/v1/chat/completions`. Model directories may contain other files, but the loader rejects unsupported checkpoints.

If loading fails, check the model identity, architecture, dimensions and GGUF version; do not relabel a different model to bypass validation. Reduce context, batch size or offload if memory is insufficient. A missing interface requires rebuilding the web or providing `--path`.

## MCP and web features

Enable the built-in shell MCP server with `--local-shell-mcp-server` (or `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1`). The web discovers it automatically and asks permission for tool calls. It offers the installed Bash, Zsh, sh and PowerShell interpreters, displays the command, stdout and stderr, and records each call as JSONL.

The web includes a CPU/GPU and memory monitor, a status bar, shared settings stored by the server, a selector for 15 interface languages, an optional instruction to reply in the selected regional language, and queue notices that distinguish web requests from other API clients. Model browsing, downloads, conversation tabs and the dense-model policy are retained.

The web is served at `/`; the API is under `/api` by default. OpenAI-compatible clients must use `/api/v1` as their base path. Swagger documentation is available at `/api/doc/`. The new API paths have no root aliases. `--api-prefix` may select a path below `/api`; `/api/config.js` publishes it to the web.

## Local shell MCP

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

Open the web and enter that API key. The `local shell mcp server` entry appears automatically when enabled on the server; it can be disabled globally or per conversation. The available tools are `bash.run`, `zsh.run`, `sh.run` and `psh.run`, only for installed interpreters. The endpoint is `POST /api/mcp/shell`, using MCP Streamable HTTP with JSON responses and protocol version `2025-11-25`.

Each tool accepts `command`, optional `cwd` and optional `timeout` in seconds (60 by default, at most 3600). Calls use independent processes with no interactive terminal. Working directory and environment changes do not persist across calls. Cancellation or timeout terminates the process group. Commands run with the server account's permissions; protect access with an API key.

The chat shows stdin (the command), stdout and stderr separately, with the exit code. Display settings can expand these boxes automatically. Logs under `~/.dgs/local-shell-mcp-server/` contain the full command and output, one timestamped JSONL file per call; the chat may truncate long output. Files are private to the server account and are not automatically rotated.

On Linux machines that already have sudo, its password requests appear in a focused dialog with Accept/Cancel. The password is sent only to the helper, not to the model, conversation or logs. Waiting for the user pauses the command deadline; unanswered requests expire after five minutes. This project does not install sudo, and production operation does not require it.

## Monitor and shared preferences

Use **Monitor** in the sidebar, or click the bottom status bar. CPU and GPU usage refresh every two seconds while the page is visible. The full view adds RAM, VRAM, temperatures and power where the driver provides them. Memory is split into dense weights, MTP/draft weights, KV/recurrent caches, compute buffers, saved prompts/checkpoints, other use and free memory. NVIDIA uses NVML dynamically; AMD uses sysfs; unavailable measurements remain unknown. Router mode aggregates loaded children without loading or waking models.

Settings are saved in `~/.dgs/config/ui-settings.json` of the server account (0600). All browsers using the same server share configuration, theme, language, favorites and tool preferences. Saves replace the document, so the last successful save wins if browsers edit concurrently. API authentication keys stay in the browser; conversations and transient navigation/stream state remain browser-local. Old browser settings are not migrated.

Select the language under General settings. The change saves and reloads the interface; save other pending edits first. Supported locales are `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`; English (US) is the default and the fallback for untranslated strings. Enable **Respond in the interface language** to add a regional instruction to future chat requests, tool turns, generated titles and KV prefill. Existing messages are not translated; adherence depends on the model.

While an inference request waits for a slot, its status shows how many requests are ahead and whether any came directly from the API. Ordinary API clients retain their usual response format. Only streaming requests marked `"dgs_client": "webui"` receive `dgs_queue` events.

For frontend development, `npm run dev` proxies `/api` to `http://localhost:9931`; set `VITE_PUBLIC_SERVER_ORIGIN` to use another server. The normal production build embeds the web and the local Swagger viewer, requiring no external documentation CDN.

## Build and source directories

`build/` contains the CMake entry point and `build-local.sh`. The helper resolves the project relative to its own location, configures a Release build and accepts additional CMake options. Dependencies must already be installed. `DGS_BUILD_JOBS` defaults to 4. Run `cmake -S build -B _/build` when configuring manually; `backend/CMakeLists.txt` is the engine project. Generated files and `bin/dgs-server` belong to `_/build/`, separate from the build scripts.

After compilation, production installation as root uses `cmake --install _/build --prefix /usr/local`. The CMake web provisioning rules live in `deploy/scripts/ui-assets.cmake`; `frontend/sources.cmake` lists their inputs and `frontend/embed.py` generates the embedded C++ assets. The main README is in the root and its regional versions are in `doc/`.


<a id="license"></a>

## License and third-party notices

NiPeGun's contributions are licensed under [PolyForm Noncommercial 1.0.0](../LICENSE). Incorporated third-party code retains its own licenses and copyright notices, collected in [build/licenses/](../build/licenses/README.md).

The main LICENSE is kept verbatim. In particular, the MIT notice of the ggml authors remains in `build/licenses/LICENSE-llama.cpp`; Swagger UI retains its Apache license and NOTICE. The license inventory also includes the bundled HTTP, JSON, hashing, media, subprocess, base64, CPU-kernel and mathematics components. Notices embedded in dependency source files or web bundles remain applicable. Model weights and system libraries have their own terms.

`cmake --install _/build --prefix PREFIX` copies the project LICENSE and `build/licenses/` into `PREFIX/share/doc/dgs-server/licenses/` with the default data layout. The default `/usr/local` prefix therefore installs them at `/usr/local/share/doc/dgs-server/licenses/`; a user prefix can use `~/.local`. The `Licenses` component can be installed separately. Include these documents when distributing the executable manually.
