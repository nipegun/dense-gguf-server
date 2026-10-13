# Manuale di DGS

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · **it-IT** · [ja-JP](MANUAL.ja-JP.md) · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server è un progetto indipendente sviluppato e mantenuto da **NiPeGun**.

DGS serve i seguenti modelli densi e la relativa applicazione web di chat:

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

Tutti gli altri modelli linguistici, comprese tutte le versioni di GLM e Kimi, non sono supportati. I modelli MoE restano esclusi. Il server controlla l'identità GGUF originale e le dimensioni degli strati, anche quando carica un modello draft per la decodifica speculativa. Restano supportate le quantizzazioni dei modelli ammessi.

<a id="local-build"></a>

## Compilazione ed esecuzione su Debian

Usa Debian 13 o un sistema compatibile con C++17, CMake 3.19+, Python 3.13+ e Node.js 20.19+ oppure 22.12+. Installa i pacchetti come root:

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

Apri `http://127.0.0.1:8080`. I pesi sono forniti separatamente. Per immagini e video, specifica il proiettore Qwen corrispondente con `--mmproj`; la gestione dei video richiede anche FFmpeg. I modelli audio e la generazione audio non sono supportati.

La modalità Release, i kernel CPU nativi e l'ottimizzazione al collegamento, se disponibile, sono attivi per impostazione predefinita. Le compilazioni CUDA e Vulkan usano `-DGGML_CUDA=ON` o `-DGGML_VULKAN=ON` con il relativo SDK. Compila per la CPU di destinazione oppure usa `-DGGML_NATIVE=OFF`. Non è stato misurato alcun miglioramento del throughput.

## Solo formati attuali

È accettato soltanto GGUF v3. `general.name`, oppure `general.basename` insieme a `general.size_label`, deve identificare un modello ammesso; il nome del file non basta. Questa politica verifica metadati e dimensioni, senza autenticare i pesi. Un file Qwen3.5-27B non viene accettato come Qwen3.8-27B solo perché le architetture hanno le stesse dimensioni.

La chat richiede il template Jinja completo del modello oppure `--chat-template-file`. Sono stati rimossi il vecchio motore a template fissi, l'opzione `--jinja`/`--no-jinja`, le API C obsolete, le varianti storiche delle opzioni CLI lunghe e la vecchia modalità di ragionamento. L'API mantiene i contratti attuali `/api/v1/*`; i vecchi alias `/completion` e `/embedding` sono stati eliminati.

L'interfaccia usa lo spazio di archiviazione attuale `DgsUi` e messaggi strutturati. Non legge né migra vecchi spazi, temi, marcatori di messaggi o esportazioni JSON storiche. Supporta le sessioni JSONL attuali e gli archivi ZIP. I vecchi dati del browser non vengono eliminati. MCP usa StreamableHTTP o WebSocket; il vecchio trasporto SSE e il ripiego automatico sono stati rimossi. Gli appunti richiedono l'API Clipboard attuale in un contesto sicuro.

## Struttura e documentazione

- `backend/`: server, inferenza Qwen, kernel tensoriali condivisi e visione Qwen.
- `frontend/`: applicazione Svelte attuale e versioni delle dipendenze fissate.
- `deploy/`: strumenti di compilazione e incorporamento dell'interfaccia locale.
- [Mappa tecnica](CODE.it-IT.md) e [manuale](MANUAL.it-IT.md).

La compilazione normale compila e incorpora l'interfaccia locale. Per riutilizzare un'interfaccia già generata con `cd frontend && npm ci && npm run build`, configura `-DDGS_USE_LOCAL_UI=ON`. Rigenera le risorse dopo aver modificato i sorgenti. `-DDGS_BUILD_UI=OFF` esclude le risorse incorporate; in questa modalità usa `--path` con una directory web attuale già compilata.

Le regole del progetto vietano i test. La verifica si limita alla compilazione e all'analisi statica dei tipi; non sono state eseguite inferenze né benchmark. Il progetto non usa Git; il proprietario rimuove manualmente la directory protetta `.git`.

## Uso quotidiano

Avvia una conversazione, seleziona un modello ammesso già caricato e invia testo o allegati immagine/video compatibili. Le impostazioni controllano campionamento, contesto e strumenti. `-c` imposta la lunghezza del contesto, `-t` i thread CPU e `-ngl auto` il posizionamento automatico sulla GPU. Usa `--api-key` per l'autenticazione. I client compatibili si collegano a `http://127.0.0.1:8080/api/v1` e `/api/v1/chat/completions`. Le directory dei modelli possono contenere altri file, ma il caricatore rifiuta checkpoint non supportati.

Se il caricamento fallisce, controlla identità, architettura, dimensioni e versione GGUF; non modificare l'identità di un modello diverso per aggirare la verifica. Se la memoria non basta, riduci contesto, dimensione dei batch o trasferimento su GPU. Se manca l'interfaccia, ricompilala o specifica `--path`.

## MCP e funzioni web

Attiva il server MCP di shell integrato con `--local-shell-mcp-server` oppure `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1`. L'interfaccia lo rileva automaticamente e chiede il permesso prima di usare gli strumenti. Espone gli interpreti Bash, Zsh, sh e PowerShell installati, mostra comando, stdout e stderr e registra ogni chiamata in JSONL.

L'interfaccia comprende monitor CPU/GPU e memoria, barra di stato, impostazioni condivise sul server, selettore di 15 lingue, istruzione facoltativa per rispondere nella variante regionale scelta e avvisi di coda che distinguono le richieste web dagli altri client API. Restano disponibili esplorazione e download dei modelli, schede delle conversazioni e politica dei soli modelli densi.

L'interfaccia è servita su `/`; l'API è sotto `/api` per impostazione predefinita. I client compatibili OpenAI devono usare `/api/v1` come percorso base. Swagger è disponibile su `/api/doc/`. I nuovi percorsi API non hanno alias nella radice. `--api-prefix` permette un percorso sotto `/api`; `/api/config.js` lo comunica all'interfaccia.

## MCP di shell locali

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

Apri l'interfaccia e inserisci la chiave API. La voce `local shell mcp server` compare automaticamente se abilitata sul server; può essere disattivata globalmente o per conversazione. Gli strumenti sono `bash.run`, `zsh.run`, `sh.run` e `psh.run`, soltanto per interpreti installati. L'endpoint è `POST /api/mcp/shell`, con MCP Streamable HTTP, risposte JSON e protocollo `2025-11-25`.

Ogni strumento accetta `command`, `cwd` facoltativo e `timeout` facoltativo in secondi: 60 per impostazione predefinita, massimo 3600. Le chiamate usano processi indipendenti senza terminale interattivo. Directory e ambiente non persistono tra le chiamate. Annullamento o scadenza termina il gruppo di processi. I comandi hanno i permessi dell'account del server; proteggi l'accesso con una chiave API.

La chat mostra separatamente stdin (il comando), stdout e stderr, con il codice di uscita. Le impostazioni di visualizzazione possono aprire automaticamente questi riquadri. I log in `~/.dgs/local-shell-mcp-server/` conservano comando e output completi, un JSONL con data e ora per chiamata; la chat può troncare output lunghi. I file sono privati dell'account del server e non vengono ruotati automaticamente.

Sui sistemi Linux dove sudo è già presente, le richieste di password appaiono in una finestra con campo attivo e pulsanti Accetta/Annulla. La password viene inviata solo all'ausiliario, non al modello, alla conversazione o ai log. Durante l'attesa dell'utente il limite temporale del comando è sospeso; le richieste senza risposta scadono dopo cinque minuti. Il progetto non installa sudo e non lo richiede in produzione.

## Monitor e preferenze condivise

Apri **Monitor** dalla barra laterale o fai clic sulla barra di stato inferiore. L'uso CPU/GPU si aggiorna ogni due secondi quando la pagina è visibile. La vista completa aggiunge RAM, VRAM, temperatura e consumo se disponibili dal driver. La memoria distingue pesi densi, pesi MTP/draft, cache KV/stato ricorrente, buffer di calcolo, prompt/checkpoint salvati, altri usi e spazio libero. NVIDIA usa NVML dinamicamente, AMD usa sysfs; le misure indisponibili restano sconosciute. Il router aggrega i processi caricati senza caricare o risvegliare modelli.

Le impostazioni vengono salvate in `~/.dgs/config/ui-settings.json` dell'account del server, con modalità 0600. Tutti i browser dello stesso server condividono configurazione, tema, lingua, preferiti e opzioni degli strumenti. Il salvataggio sostituisce il documento: se più browser modificano contemporaneamente, prevale l'ultimo salvataggio riuscito. Le chiavi API restano nel browser, come conversazioni e stato temporaneo di navigazione e flussi. Le vecchie impostazioni non vengono migrate.

Seleziona la lingua nelle impostazioni generali. La modifica salva la scelta e ricarica l'interfaccia; salva prima le altre modifiche in sospeso. Le lingue disponibili sono `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU` e `zh-CN`. L'inglese statunitense è predefinito e viene usato per i testi non tradotti. Attiva **Rispondi nella lingua dell'interfaccia** (Respond in the interface language) per aggiungere un'istruzione regionale a future richieste, turni degli strumenti, titoli e precompilazione KV. I messaggi esistenti non vengono tradotti; il rispetto dell'istruzione dipende dal modello.

Quando una richiesta aspetta uno slot, il suo stato indica quante richieste la precedono e se alcune provengono direttamente dall'API. I client API ordinari mantengono il formato abituale. Solo le richieste in streaming contrassegnate con `"dgs_client": "webui"` ricevono eventi `dgs_queue`.

Per lo sviluppo frontend, `npm run dev` inoltra `/api` a `http://localhost:9931`; imposta `VITE_PUBLIC_SERVER_ORIGIN` per usare un altro server. La compilazione di produzione incorpora interfaccia e Swagger locale senza richiedere una CDN esterna per la documentazione.

## Directory dei sorgenti e di compilazione

`build/` contiene l'ingresso CMake e `build-local.sh`. Lo script individua il progetto dalla propria posizione, configura Release e accetta ulteriori opzioni CMake. Le dipendenze devono essere già installate. `DGS_BUILD_JOBS` vale 4 per impostazione predefinita. Per la configurazione manuale esegui `cmake -S build -B _/build`; `backend/CMakeLists.txt` definisce il progetto del motore. I file generati e `bin/dgs-server` appartengono a `_/build/`, separati dagli script di compilazione.

Dopo la compilazione, l'installazione in produzione come root usa `cmake --install _/build --prefix /usr/local`. Le regole CMake per preparare l'interfaccia sono in `deploy/scripts/ui-assets.cmake`; `frontend/sources.cmake` ne elenca gli input e `frontend/embed.py` genera le risorse C++ incorporate. Il README principale è nella radice e le versioni regionali in `doc/`.


<a id="license"></a>

## Licenza e avvisi di terze parti

I contributi di NiPeGun sono distribuiti con licenza [PolyForm Noncommercial 1.0.0](../LICENSE). Il codice di terze parti conserva le proprie licenze e note di copyright, raccolte in [build/licenses/](../build/licenses/README.md).

Il LICENSE principale resta letteralmente invariato. In particolare, l'avviso MIT degli autori ggml rimane in `build/licenses/LICENSE-llama.cpp`; Swagger UI mantiene la licenza Apache e il NOTICE. L'inventario comprende anche i componenti HTTP, JSON, hash, multimediali, subprocess, base64, kernel CPU e librerie matematiche incorporati. Restano applicabili gli avvisi nei sorgenti delle dipendenze e nei pacchetti web. I pesi dei modelli e le librerie di sistema hanno termini propri.

`cmake --install _/build --prefix PREFIX` copia il LICENSE del progetto e `build/licenses/` in `PREFIX/share/doc/dgs-server/licenses/` con la disposizione dati predefinita. Con il prefisso `/usr/local` vengono quindi installati in `/usr/local/share/doc/dgs-server/licenses/`; per un'installazione utente si può usare `~/.local`. Il componente `Licenses` è installabile separatamente. Includi questi documenti quando distribuisci manualmente l'eseguibile.
