# DGS — mappa del codice

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · **it-IT** · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server è un progetto indipendente sviluppato e mantenuto da **NiPeGun**.

- [Architettura](#architecture)
- [Moduli](#modules)
- [Simboli](#symbols)
- [Flussi](#flows)
- [Punti di ingresso](#entrypoints)
- [Impatto](#impact)
- [Estensioni](#extension)
- [MCP, monitor e interfaccia condivisa](#integrated-features)
- [Fonti e verifica](#validation)
- [Organizzazione del repository](#repository-organization)

- [Licenza e avvisi di terze parti](#license)

<a id="architecture"></a>

## Architettura

Un server e un'applicazione web. La factory istanzia soltanto `dgs_model_qwen35`; Qwen3.8-27B condivide questa architettura. Restano i kernel DeltaNet e il proiettore Qwen con l'ausiliario visivo condiviso. MoE, vecchie versioni GGUF e altri profili vengono rifiutati. La lista ammessa legge i metadati originali indipendentemente dagli alias HTTP, poi verifica strati, embedding, FFN e teste. Le sole dimensioni non distinguono Qwen3.5-27B da Qwen3.8-27B. L'identità è dichiarata nel GGUF, non autenticata crittograficamente. GGML e le dipendenze esterne conservano primitive e interfacce attuali.

<a id="modules"></a>

## Moduli

| Modulo | Percorso | Responsabilità | Dipende da | Usato da |
|---|---|---|---|---|
| Politica | `backend/src/dgs-model-policy.*` | Lista Qwen ammessa | GGUF, hparams | caricatore dei modelli |
| Qwen | `backend/src/models/` | Grafo denso | DeltaNet, GGML | server |
| Visione | `backend/tools/mtmd/` | Immagini/video Qwen | GGML, codec | server |
| Chat | `backend/common/` | Jinja, strumenti XML, campionamento | modello, vendor | server |
| HTTP | `backend/tools/server/` | Percorsi, slot, strumenti | common, mtmd, UI | eseguibile |
| Web | `frontend/` | interfaccia e archiviazione attuali | versioni npm fissate | browser |
| Compilazione | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | Compilare e incorporare | compilatore, npm, Python | distribuzione |
| MCP di shell | backend/tools/server/server-shell-mcp.* | Comandi MCP e richieste di password | HTTP, POSIX, common_json | server, interfaccia MCP |
| Monitor | backend/tools/server/server-monitor.* | Telemetria CPU/GPU e aggregazione della memoria | procfs, sysfs, NVML facoltativo | server-context, router, web |
| Pesi MTP | backend/src/dgs-ext.h; dgs-model.cpp | Ripartizione delle allocazioni dense/MTP | metadati dei buffer GGML | istantanee del monitor |
| Impostazioni condivise | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | Documento delle impostazioni atomico | filesystem, API, SettingsService | tutti gli store dell'interfaccia |
| Documentazione API | backend/tools/server/server-api-doc.*; frontend/api.html | OpenAPI e Swagger aggiornati | registro dei percorsi, schema, vendor/swagger-ui | /api/doc/ |
| Lingue | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 cataloghi e istruzione di risposta | JSON, impostazioni condivise | componenti web, servizio chat |
| Avvisi di coda | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | Eventi di coda su richiesta | code di attività e risposte | interfaccia chat in streaming |
| Risorse web | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | Compilare e incorporare l'interfaccia | npm, Python | `dgs-ui` |
| Distribuzione delle licenze | `backend/CMakeLists.txt`, `build/licenses/` | Separare i termini del progetto da quelli dei terzi | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## Simboli

| Simbolo | File:riga | Funzione |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | Verificare l'identità GGUF |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | Verificare le dimensioni dense |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | Rifiutare tensori MoE |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | Creare un modello ammesso |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Caricare le dimensioni Qwen |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Elaborare Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Creare il grafo visivo Qwen |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | Avviare il server e registrare i percorsi |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | Smistare JSON-RPC MCP |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | Verificare ed eseguire un comando shell |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | Inoltrare le richieste di password sudo |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | Aggregare hardware e memoria dei modelli |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | Pubblicare segmenti di memoria densa |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | Contare i pesi MTP aggiunti |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | Salvare atomicamente le impostazioni condivise |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | Servire OpenAPI e Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | Inviare la posizione in coda |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | Aggiungere l'istruzione linguistica regionale |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | Cancellare le stringhe di password usate |

<a id="flows"></a>

## Flussi

Caricamento: `fDgsModelCreate` → rifiuto MoE → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → tensori/grafo. Chat: HTTP → coda/slot → Jinja → tokenizzazione → inferenza → risposta in streaming. Web: inizializzazione diretta dello schema `DgsUi` senza migrazioni. Compilazione: npm → `frontend/dist` → `frontend/embed.py` → libreria incorporata. I modelli draft seguono la stessa politica; MTP usa gli strati predittivi del modello ammesso.

<a id="entrypoints"></a>

## Punti di ingresso

| Percorso | Gestore | File |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | percorsi di completamento attuali | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | percorsi API attuali | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | server/router | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | interfaccia incorporata | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | istantanee e aggregazione del router | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | Prefisso e lingua salvata | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## Impatto

La politica interessa ogni caricamento, anche dei draft e del solo vocabolario. Rinominare metadati GGUF può impedire l'identificazione; non sostituirli con nomi di file. I nomi storici ancora usati come chiavi GGUF o primitive condivise non abilitano vecchi modelli. Non ripristinare API, migrazioni o alias eliminati. Gli schemi richiedono modifiche coordinate tra frontend e backend.

<a id="extension"></a>

## Estensioni

Aggiungere un modello richiede una decisione esplicita sulla lista ammessa, comprendente architettura, metadati e dimensioni. Non ampliare il riconoscimento basandosi solo sul nome e non reintrodurre MoE. Registra nuovi endpoint sul server e aggiorna i tipi web. Mantieni README, MANUAL e CODE aggiornati in tutte le 15 lingue dell'interfaccia.

<a id="validation"></a>

## Fonti e verifica

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

Le configurazioni ufficiali sopra elencate definiscono i profili. La verifica usa soltanto compilazione e analisi statica: i test sono vietati dalle regole del progetto. Non sono state eseguite inferenze o misure di throughput. Il proprietario elimina manualmente la directory protetta `.git`; l'agente non usa Git.



<a id="integrated-features"></a>

## MCP, monitor e interfaccia condivisa

Shell: opzione CLI → rilevamento degli interpreti → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → autorizzazione → `tools/call` → gruppo di processi indipendente → risultato JSON e JSONL per chiamata → visualizzazione stdin/stdout/stderr. Il canale askpass usa un identificatore di chiamata e un socket Unix privato; il dialogo non inserisce password nei messaggi del modello. `common_json::mWipeString` sovrascrive le stringhe JSON sensibili dell'ausiliario dopo l'uso.

Monitor: caricamento → ripartizione dei pesi MTP → istantanea dal ciclo principale, al massimo una al secondo durante l'inferenza → copia protetta da mutex → `/api/monitor*` → interrogazione ogni due secondi con pagina visibile. I pesi principali e del draft separato vengono contati una sola volta. Il monitor non include buffer, cache o prefetch di esperti. Le richieste non inseriscono attività di inferenza né risvegliano modelli. L'uso hardware è indipendente dall'istantanea di inferenza.

Impostazioni: chiave API del browser → `/api/ui/settings` → `UiSettingsStore.mLoad` → inizializzazione degli store → salvataggio del documento completo dopo una breve attesa dall'ultima modifica. Il documento contiene `config`, `userOverrides`, `language` e `preferences`; ogni preferenza conserva la serializzazione del proprio store. I browser condividono i valori al ricaricamento, senza modifica collaborativa in tempo reale; prevale l'ultimo documento salvato. Conversazioni, schede e cursori di ripresa dei flussi restano locali.

Lingua: `/api/config.js` → catalogo JSON → interfaccia. `fApplyResponseLanguage` modifica soltanto copie dei messaggi in uscita; chat, strumenti, titoli e precompilazione lo usano. I testi DGS aggiunti coprono en-US/en-GB/es-ES/es-AR; le altre lingue usano en-US se manca una voce nel catalogo importato.

Coda: `dgs_client=webui` → schema di completamento → contatori per origine → `server_task_result_queue` → SSE `dgs_queue` → `QueueStore` per conversazione → etichetta di attesa. I formati degli eventi OpenAI Responses e Anthropic non ricevono questi avvisi aggiuntivi. I client API senza contrassegno non sono interessati.

Base API: `/api` predefinita, configurabile solo sotto `/api`; l'interfaccia resta su `/`. I gestori ricevono percorsi relativi, i processi figli usano sempre `/api` e la web risolve il prefisso pubblicato. Le API usano la rete invece della cache del service worker. Cambiare i percorsi richiede aggiornare insieme inoltro del router, URL del client e OpenAPI.

Estensione: aggiungi insieme l'interprete candidato e il suo nome di strumento frontend; aggiungi i segmenti di memoria contemporaneamente nell'istantanea densa, nel tipo API e nei colori; aggiungi i testi ai cataloghi JSON. Conserva la lista dei modelli densi e i formati attuali. Ricompila backend e frontend dopo modifiche; non aggiungere né eseguire test, come richiesto dalle regole.


Verifica delle funzioni integrate: compilazione CPU Release con LTO e 70 risorse web incorporate riuscita. Svelte/TypeScript: zero errori e zero avvisi. Il pacchetto web supera la soglia indicativa di dimensione; GCC/CMake hanno anche segnalato avvisi esistenti e uno sfasamento dell'orologio del filesystem. Non sono stati eseguiti test, server, inferenza o compilazioni GPU.

<a id="repository-organization"></a>

## Organizzazione del repository

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

Flusso di compilazione: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, frontend, server. Interfaccia: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → libreria incorporata. Sia il wrapper sia il backend possono essere radici CMake. Uno spostamento richiede aggiornare insieme include, percorsi CMake, dipendenze e documentazione.


Verifica della struttura: configurazione CMake riuscita da build/ e backend/. Compilazione CPU Release con LTO e interfaccia incorporata riuscita da build/. Svelte/TypeScript senza errori o avvisi; controllati sintassi Bash e collegamenti locali. Non sono stati eseguiti test né inferenza.


<a id="license"></a>

## Licenza e avvisi di terze parti

I contributi di NiPeGun sono distribuiti con licenza [PolyForm Noncommercial 1.0.0](../LICENSE). Il codice di terze parti conserva le proprie licenze e note di copyright, raccolte in [build/licenses/](../build/licenses/README.md).

Il LICENSE principale resta letteralmente invariato. In particolare, l'avviso MIT degli autori ggml rimane in `build/licenses/LICENSE-llama.cpp`; Swagger UI mantiene la licenza Apache e il NOTICE. L'inventario comprende anche i componenti HTTP, JSON, hash, multimediali, subprocess, base64, kernel CPU e librerie matematiche incorporati. Restano applicabili gli avvisi nei sorgenti delle dipendenze e nei pacchetti web. I pesi dei modelli e le librerie di sistema hanno termini propri.

`cmake --install _/build --prefix PREFIX` copia il LICENSE del progetto e `build/licenses/` in `PREFIX/share/doc/dgs-server/licenses/` con la disposizione dati predefinita. Con il prefisso `/usr/local` vengono quindi installati in `/usr/local/share/doc/dgs-server/licenses/`; per un'installazione utente si può usare `~/.local`. Il componente `Licenses` è installabile separatamente. Includi questi documenti quando distribuisci manualmente l'eseguibile.

Distribuzione: LICENSE del progetto e avvisi di terze parti → componente CMake `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`. Ogni cambio di dipendenza richiede rivedere il relativo avviso. Nomi del prodotto e tipi API usano DGS; attribuzioni di terzi e identificatori GGUF non vengono rinominati. L'adattatore inutilizzato del catalogo esterno e i preset non supportati sono stati rimossi; la ricerca continua tramite il servizio Hugging Face esistente.

Verifica di identità e licenze: compilazione CPU Release con LTO e interfaccia incorporata riuscita; Svelte/TypeScript senza errori o avvisi. L'installazione CMake generata include licenza principale e avvisi raccolti. Il LICENSE principale coincide byte per byte con il riferimento. Non sono stati eseguiti test, inferenza o installazione in produzione.
