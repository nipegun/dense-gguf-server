# dense-gguf-server

Questo README è disponibile anche nelle seguenti lingue:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), **it-IT**, [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Che cos'è

Esegui localmente modelli GGUF densi tramite una chat web e un'API HTTP di inferenza.

Come llama.cpp, ma realizzato con precisione da bisturi nipeguniana per raggiungere la massima velocità di inferenza possibile.

## Schermate

Non sono ancora state aggiunte schermate.

## Distribuzione

Su **Debian**, dopo aver compilato il progetto, esegui questi comandi come `root` dalla directory principale:

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

Apri `http://127.0.0.1:8080`. L'API usa `/api`, i client compatibili con OpenAI usano `/api/v1` e Swagger è disponibile su `/api/doc/`. I pesi dei modelli sono forniti separatamente; usa `--api-key` quando concedi accesso ad altri client.

## Compilazione

Installa prima le dipendenze, come `root` su Debian 13:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

Poi esegui dalla directory del progetto con il tuo utente abituale:

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

Lo script compila i sorgenti locali in `_/build/`. `DGS_BUILD_JOBS=8` imposta il numero di processi paralleli; gli argomenti aggiuntivi vengono passati a CMake, ad esempio `-DGGML_CUDA=ON` o `-DGGML_VULKAN=ON` se è installato il relativo SDK. Le [dipendenze e opzioni di compilazione](MANUAL.it-IT.md#local-build) devono essere già disponibili.

## Consulta la documentazione

Consulta:

- [doc/MANUAL.it-IT.md](MANUAL.it-IT.md) per compilare, installare e usare DGS, configurare MCP e l'interfaccia o diagnosticare problemi.
- [doc/CODE.it-IT.md](CODE.it-IT.md) per conoscere il motore denso, l'organizzazione dei sorgenti e i contratti di compilazione di backend e frontend.

## Sostieni questo progetto

- **[Incaricami](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)** per aggiungere nuove funzionalità.
- **GitHub Sponsors.** Puoi anche sostenere economicamente il progetto su GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
