# dense-gguf-server

This README is also available in:\
[de-DE](doc/README.de-DE.md), [en-GB](doc/README.en-GB.md), **en-US**, [es-AR](doc/README.es-AR.md), [es-ES](doc/README.es-ES.md), [fr-FR](doc/README.fr-FR.md), [he-IL](doc/README.he-IL.md), [hi-IN](doc/README.hi-IN.md), [it-IT](doc/README.it-IT.md), [ja-JP](doc/README.ja-JP.md), [ko-KR](doc/README.ko-KR.md), [pt-BR](doc/README.pt-BR.md), [pt-PT](doc/README.pt-PT.md), [ru-RU](doc/README.ru-RU.md), [zh-CN](doc/README.zh-CN.md)

## What is it

Run GGUF dense models locally through a web chat and an HTTP inference API.

Like llama.cpp but crafted with nipegunian scalpel precision to achieve the fastest inference speed possible.

## Screenshots

Screenshots have not been added yet.

## Deploy

On **Debian**, after building the project, run these commands as `root` from its root directory:

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

Open `http://127.0.0.1:8080`. The API uses `/api`, OpenAI-compatible clients use `/api/v1`, and Swagger is at `/api/doc/`. Model weights are supplied separately; use `--api-key` when granting access to other clients.

## Build

Install the dependencies first, as `root` on Debian 13:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

Then run from the project directory with your regular user:

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

The helper builds the local sources in `_/build/`. `DGS_BUILD_JOBS=8` sets the job count; extra arguments are passed to CMake, for example `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` when the corresponding SDK is installed. [Build dependencies and options](doc/MANUAL.md#local-build) must already be available.

## Read the documentation

Explore:

- [doc/MANUAL.md](doc/MANUAL.md) to build, install and use DGS, configure MCP and the web, or diagnose problems.
- [doc/CODE.md](doc/CODE.md) to understand the dense engine, source organization and backend/frontend build contracts.

## Sponsor this project

- **[Hire me](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)** to add new features.
- **GitHub Sponsors.** You can also support the project financially on GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
