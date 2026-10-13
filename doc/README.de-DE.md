# dense-gguf-server

Diese README ist auch in folgenden Sprachen verfügbar:\
**de-DE**, [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Was ist das?

Führen Sie dichte GGUF-Modelle lokal über einen Webchat und eine HTTP-Inferenz-API aus.

Wie llama.cpp, aber mit nipegunianischer Skalpellpräzision gefertigt, um die höchstmögliche Inferenzgeschwindigkeit zu erreichen.

## Bildschirmfotos

Es wurden noch keine Bildschirmfotos hinzugefügt.

## Bereitstellung

Führen Sie unter **Debian** nach dem Kompilieren diese Befehle als `root` im Projektwurzelverzeichnis aus:

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

Öffnen Sie `http://127.0.0.1:8080`. Die API verwendet `/api`, OpenAI-kompatible Clients verwenden `/api/v1`, und Swagger liegt unter `/api/doc/`. Modellgewichte werden separat bereitgestellt; verwenden Sie `--api-key`, wenn Sie anderen Clients Zugriff gewähren.

## Kompilierung

Installieren Sie zunächst die Abhängigkeiten als `root` unter Debian 13:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

Führen Sie danach diese Befehle mit Ihrem normalen Benutzerkonto im Projektverzeichnis aus:

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

Das Skript kompiliert die lokalen Quellen in `_/build/`. `DGS_BUILD_JOBS=8` legt die Anzahl paralleler Aufgaben fest. Zusätzliche Argumente gehen an CMake, etwa `-DGGML_CUDA=ON` oder `-DGGML_VULKAN=ON`, wenn das entsprechende SDK installiert ist. Die [Abhängigkeiten und Build-Optionen](MANUAL.de-DE.md#local-build) müssen bereits verfügbar sein.

## Dokumentation lesen

Weiterführende Dokumentation:

- [doc/MANUAL.de-DE.md](MANUAL.de-DE.md) beschreibt Kompilierung, Installation und Verwendung von DGS, die Einrichtung von MCP und Weboberfläche sowie die Fehlerdiagnose.
- [doc/CODE.de-DE.md](CODE.de-DE.md) erläutert die dichte Inferenz-Engine, die Quellstruktur und die Build-Schnittstellen zwischen Backend und Frontend.

## Dieses Projekt unterstützen

- **[Beauftragen Sie mich](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)**, um neue Funktionen hinzuzufügen.
- **GitHub Sponsors.** Sie können das Projekt auch finanziell auf GitHub unterstützen: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
