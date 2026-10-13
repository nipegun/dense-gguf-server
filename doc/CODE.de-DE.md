# DGS — Codeübersicht

**de-DE** · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server ist ein unabhängiges Projekt, das von **NiPeGun** entwickelt und gepflegt wird.

- [Architektur](#architecture)
- [Module](#modules)
- [Symbole](#symbols)
- [Abläufe](#flows)
- [Einstiegspunkte](#entrypoints)
- [Auswirkungen](#impact)
- [Erweiterungen](#extension)
- [MCP, Monitor und gemeinsame Oberfläche](#integrated-features)
- [Quellen und Validierung](#validation)
- [Verzeichnisstruktur](#repository-organization)

- [Lizenz und Hinweise zu Drittanbietern](#license)

<a id="architecture"></a>

## Architektur

Ein Server und eine Webanwendung. Die Factory erzeugt ausschließlich `dgs_model_qwen35`; Qwen3.8-27B teilt diese Architektur. DeltaNet-Kernels und der Qwen-Projektor mit gemeinsamer Bildverarbeitung bleiben erhalten. MoE, alte GGUF-Versionen und andere Profile werden abgewiesen. Die Positivliste liest ursprüngliche Metadaten unabhängig von HTTP-Aliasen und prüft danach Schichten, Embeddings, FFN und Köpfe. Dimensionen allein unterscheiden Qwen3.5-27B nicht von Qwen3.8-27B. Die GGUF-Datei behauptet eine Identität, die nicht kryptografisch authentifiziert ist. GGML und externe Abhängigkeiten behalten ihre aktuellen Primitive und Schnittstellen.

<a id="modules"></a>

## Module

| Modul | Pfad | Aufgabe | Abhängigkeiten | Verwendet von |
|---|---|---|---|---|
| Richtlinie | `backend/src/dgs-model-policy.*` | Qwen-Positivliste | GGUF, hparams | Modell-Loader |
| Qwen | `backend/src/models/` | Dichter Graph | DeltaNet, GGML | Server |
| Bildverarbeitung | `backend/tools/mtmd/` | Qwen-Bild/Video | GGML, Codecs | Server |
| Chat | `backend/common/` | Jinja, XML-Werkzeuge, Sampling | Modell, vendor | Server |
| HTTP | `backend/tools/server/` | Routen, Slots, Werkzeuge | common, mtmd, UI | Programmdatei |
| Web | `frontend/` | aktuelle Oberfläche/Speicherung | festgelegte npm-Versionen | Browser |
| Kompilierung | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | Kompilieren und einbetten | Compiler, npm, Python | Bereitstellung |
| Shell-MCP | backend/tools/server/server-shell-mcp.* | MCP-Befehle und Passwortabfragen | HTTP, POSIX, common_json | Server, MCP-Oberfläche |
| Monitor | backend/tools/server/server-monitor.* | CPU/GPU-Telemetrie und Speicheraggregation | procfs, sysfs, optional NVML | server-context, Router, Web |
| MTP-Gewichte | backend/src/dgs-ext.h; dgs-model.cpp | Aufteilung dichter/MTP-Speicherbelegung | GGML-Puffermetadaten | Monitor-Snapshots |
| Gemeinsame Einstellungen | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | Atomar gespeichertes Einstellungsdokument | Dateisystem, API, SettingsService | alle Web-Stores |
| API-Dokumentation | backend/tools/server/server-api-doc.*; frontend/api.html | Aktuelles OpenAPI und Swagger | Routenregister, Schema, vendor/swagger-ui | /api/doc/ |
| Sprachen | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 Kataloge und Antwortanweisung | JSON, gemeinsame Einstellungen | Webkomponenten, Chatdienst |
| Warteschlangenhinweise | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | Angeforderte Warteschlangenereignisse | Aufgaben- und Antwortwarteschlangen | Streaming-Chatoberfläche |
| Webressourcen | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | Oberfläche kompilieren und einbetten | npm, Python | `dgs-ui` |
| Lizenzverteilung | `backend/CMakeLists.txt`, `build/licenses/` | Projekt- und Drittbedingungen trennen | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## Symbole

| Symbol | Datei:Zeile | Funktion |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | GGUF-Identität prüfen |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | Dichte Dimensionen prüfen |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | MoE-Tensoren ablehnen |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | Zugelassenes Modell erstellen |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Qwen-Dimensionen laden |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Jinja rendern |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Qwen-Bildgraph erstellen |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | Server starten und Routen registrieren |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | MCP-JSON-RPC verteilen |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | Shell-Aufruf prüfen und ausführen |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | sudo-Passwortabfragen weiterleiten |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | Hardware- und Modellspeicher aggregieren |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | Dichte Speichersegmente veröffentlichen |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | Angehängte MTP-Gewichte zählen |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | Gemeinsame Einstellungen atomar speichern |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | OpenAPI und Swagger bereitstellen |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | Warteschlangenposition senden |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | Regionale Sprachanweisung ergänzen |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | Verbrauchte Passwortzeichenfolgen löschen |

<a id="flows"></a>

## Abläufe

Laden: `fDgsModelCreate` → MoE-Ablehnung → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → Tensoren/Graph. Chat: HTTP → Warteschlange/Slot → Jinja → Tokenisierung → Inferenz → Streaming-Antwort. Web: direkte Initialisierung des `DgsUi`-Schemas ohne Migration. Build: npm → `frontend/dist` → `frontend/embed.py` → eingebettete Bibliothek. Entwurfsmodelle folgen derselben Richtlinie; MTP nutzt die Vorhersageschichten des zugelassenen Modells.

<a id="entrypoints"></a>

## Einstiegspunkte

| Route | Handler | Datei |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | aktuelle Completion-Routen | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | aktuelle API-Routen | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | Server/Router | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | eingebettete Oberfläche | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | Snapshots und Router-Aggregation | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | Präfix und gespeicherte Sprache | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## Auswirkungen

Richtlinienänderungen betreffen jeden Ladevorgang, auch Entwürfe und reine Vokabularladungen. Umbenannte GGUF-Metadaten können die Identifikation verhindern; ersetzen Sie sie nicht durch Dateinamen. Historische Namen aktueller GGUF-Schlüssel oder gemeinsamer Primitive aktivieren keine alten Modelle. Stellen Sie entfernte APIs, Migrationen oder Aliase nicht wieder her. Schemaänderungen verlangen koordinierte Frontend-/Backend-Anpassungen.

<a id="extension"></a>

## Erweiterungen

Ein weiteres Modell erfordert eine ausdrückliche Entscheidung über Positivliste, Architektur, Metadaten und Dimensionen. Erweitern Sie die Erkennung nicht allein anhand des Namens und führen Sie MoE nicht wieder ein. Registrieren Sie neue Endpunkte im Server und aktualisieren Sie Webtypen. Halten Sie README, MANUAL und CODE in allen 15 Oberflächensprachen aktuell.

<a id="validation"></a>

## Quellen und Validierung

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

Die oben verlinkten offiziellen Konfigurationen definieren die Profile. Validiert wird ausschließlich durch Kompilierung und statische Analyse; Projektregeln verbieten Tests. Modellinferenz und Durchsatzmessungen wurden nicht ausgeführt. Der Eigentümer löscht das geschützte Verzeichnis `.git` manuell; der Agent verwendet Git nicht.



<a id="integrated-features"></a>

## MCP, Monitor und gemeinsame Oberfläche

Shell: CLI-Option → Shell-Erkennung → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → Berechtigungsprüfung → `tools/call` → unabhängige Prozessgruppe → JSON-Ergebnis und JSONL je Aufruf → stdin/stdout/stderr-Anzeige. Der askpass-Nebenkanal nutzt eine Aufruf-ID und einen privaten Unix-Socket. Der Dialog nimmt keine Passwörter in Modellnachrichten auf. `common_json::mWipeString` überschreibt die sensiblen JSON-Zeichenfolgen des Helfers nach Gebrauch.

Monitor: Modell laden → MTP-Gewichtsaufteilung → Snapshot der Hauptschleife, während Inferenz höchstens einmal pro Sekunde → mutexgeschützte Kopie → `/api/monitor*` → Abfrage der sichtbaren Seite alle zwei Sekunden. Haupt- und separate Entwurfsgewichte werden jeweils einmal gezählt. Expertenpuffer, -caches und -prefetch-Kategorien fehlen in diesem Monitor. Die Abfragen erzeugen keine Inferenzaufgaben und wecken keine schlafenden Modelle. Hardwareauslastung ist unabhängig vom Inferenz-Snapshot.

Einstellungen: Browser-API-Schlüssel → `/api/ui/settings` → `UiSettingsStore.mLoad` → Store-Initialisierung → verzögertes Speichern des gesamten Dokuments nach Ende einer Änderungsfolge. Es enthält `config`, `userOverrides`, `language` und `preferences`; Präferenzen behalten die Serialisierung ihres Stores. Browser teilen Einstellungen beim Neuladen, nicht durch gemeinsame Echtzeitbearbeitung. Das zuletzt gespeicherte Gesamtdokument gilt. Konversationen, Tabs und Stream-Fortsetzungscursor bleiben lokal.

Sprache: `/api/config.js` → JSON-Katalog → Oberfläche. `fApplyResponseLanguage` ändert nur Kopien ausgehender Nachrichten; Chat, Werkzeuge, Titel und Vorbefüllung verwenden die Funktion. Zusätzliche DGS-Texte liegen in en-US/en-GB/es-ES/es-AR vor. Andere Sprachen verwenden en-US, wenn der importierte Katalog keinen Eintrag enthält.

Warteschlange: `dgs_client=webui` → Completion-Schema → Herkunftszähler → `server_task_result_queue` → `dgs_queue`-SSE → `QueueStore` je Konversation → Warteanzeige. Die Ereignisformate von OpenAI Responses und Anthropic erhalten diese zusätzlichen Ereignisse nicht. Unmarkierte API-Clients sind nicht betroffen.

API-Basis: standardmäßig `/api`, nur unter `/api` konfigurierbar; die Oberfläche bleibt unter `/`. Handler erhalten relative Pfade, Router-Kindprozesse nutzen stets `/api`, und das Web löst das veröffentlichte Präfix auf. API-Aufrufe nutzen das Netzwerk statt des Service-Worker-Caches. Bei Pfadänderungen müssen Router-Weiterleitung, Client-URLs und OpenAPI gemeinsam angepasst werden.

Erweiterung: Shell-Kandidaten und Frontend-Werkzeugnamen gemeinsam ergänzen; Speichersegmente zugleich in dichtem Snapshot, API-Typ und Farbgruppen ergänzen; Texte in JSON-Katalogen ergänzen. Behalten Sie die Positivliste und aktuellen Formate bei. Kompilieren Sie Backend und Frontend nach Änderungen; Projektregeln verbieten neue oder ausgeführte Tests.


Validierung integrierter Funktionen: CPU-Release-Build mit LTO und 70 eingebetteten Webressourcen erfolgreich. Svelte/TypeScript: keine Fehler oder Warnungen. Das Webpaket überschreitet den Größenrichtwert; GCC/CMake meldeten außerdem bestehende Warnungen und einen Zeitversatz des Dateisystems. Tests, Serverausführung, Modellinferenz und GPU-Builds wurden nicht durchgeführt.

<a id="repository-organization"></a>

## Verzeichnisstruktur

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

Build-Ablauf: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, Frontend, Server. UI-Ablauf: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → eingebettete Bibliothek. Wrapper und Backend können jeweils CMake-Quellwurzel sein. Umzüge erfordern abgestimmte Includes, CMake-Pfade, Abhängigkeitslisten und Dokumentation.


Strukturvalidierung: CMake-Konfiguration von build/ und backend/ erfolgreich. CPU-Release-Build mit LTO und eingebetteter Oberfläche über build/ erfolgreich. Svelte/TypeScript meldete keine Fehler oder Warnungen; Bash-Syntax und lokale Dokumentationslinks wurden geprüft. Keine Tests oder Modellinferenz ausgeführt.


<a id="license"></a>

## Lizenz und Hinweise zu Drittanbietern

Die Beiträge von NiPeGun stehen unter [PolyForm Noncommercial 1.0.0](../LICENSE). Eingebundener Fremdcode behält seine eigenen Lizenzen und Urheberrechtshinweise; sie sind unter [build/licenses/](../build/licenses/README.md) gesammelt.

Die Hauptdatei LICENSE bleibt wortgetreu erhalten. Insbesondere bleibt der MIT-Hinweis der ggml-Autoren in `build/licenses/LICENSE-llama.cpp`; Swagger UI behält Apache-Lizenz und NOTICE. Das Verzeichnis enthält außerdem Hinweise für integrierte HTTP-, JSON-, Hash-, Medien-, subprocess-, base64-, CPU-Kernel- und Mathematikkomponenten. Hinweise in Abhängigkeitsquellen und Webpaketen gelten weiterhin. Modellgewichte und Systembibliotheken haben eigene Bedingungen.

`cmake --install _/build --prefix PREFIX` kopiert die Projektlizenz und `build/licenses/` bei Standard-Datenpfaden nach `PREFIX/share/doc/dgs-server/licenses/`. Mit dem Standardpräfix `/usr/local` landen sie unter `/usr/local/share/doc/dgs-server/licenses/`; für eine Benutzerinstallation kann `~/.local` verwendet werden. Die Komponente `Licenses` lässt sich separat installieren. Legen Sie diese Dokumente bei manueller Weitergabe der Programmdatei bei.

Verteilung: Projekt-LICENSE und Drittanbieterhinweise → CMake-Komponente `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`. Abhängigkeitsänderungen erfordern eine Prüfung der jeweiligen Hinweise. Produktname und API-Typen verwenden DGS; fremde Urheberangaben und GGUF-Kennungen werden nicht umbenannt. Der ungenutzte externe Katalogadapter und nicht unterstützte Modell-Presets wurden entfernt. Die Modellsuche nutzt weiterhin den vorhandenen Hugging-Face-Dienst.

Validierung von Identität und Lizenzen: CPU-Release-Build mit LTO und eingebetteter Oberfläche erfolgreich; Svelte/TypeScript ohne Fehler und Warnungen. Die generierte CMake-Installation enthält Hauptlizenz und gesammelte Hinweise. Die Hauptdatei LICENSE stimmt bytegenau mit der Referenz überein. Keine Tests, Modellinferenz oder Produktionsinstallation ausgeführt.
