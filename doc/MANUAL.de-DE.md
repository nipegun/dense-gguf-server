# DGS-Handbuch

**de-DE** · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · [ja-JP](MANUAL.ja-JP.md) · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server ist ein unabhängiges Projekt, das von **NiPeGun** entwickelt und gepflegt wird.

DGS stellt die folgenden dichten Modelle und die zugehörige Webchat-Anwendung bereit:

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

Alle anderen Sprachmodelle, einschließlich sämtlicher GLM- und Kimi-Versionen, werden nicht unterstützt. MoE-Modelle bleiben ausgeschlossen. Der Server prüft die ursprüngliche GGUF-Identität und die Schichtdimensionen, auch beim Laden eines spekulativen Entwurfsmodells. Quantisierungen der zugelassenen Modelle bleiben unterstützt.

<a id="local-build"></a>

## Unter Debian kompilieren und ausführen

Verwenden Sie Debian 13 oder ein kompatibles System mit C++17, CMake 3.19+, Python 3.13+ und Node.js 20.19+ oder 22.12+. Installieren Sie die Pakete als root:

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

Öffnen Sie `http://127.0.0.1:8080`. Modellgewichte werden separat bereitgestellt. Geben Sie für Bild und Video den passenden Qwen-Projektor mit `--mmproj` an; die Videoverarbeitung benötigt zusätzlich FFmpeg. Audiomodelle und Audioerzeugung werden nicht unterstützt.

Release-Modus, native CPU-Kernels und unterstützte Linkzeitoptimierung sind standardmäßig aktiviert. CUDA- und Vulkan-Builds verwenden `-DGGML_CUDA=ON` beziehungsweise `-DGGML_VULKAN=ON` mit dem jeweiligen SDK. Kompilieren Sie für die Ziel-CPU oder verwenden Sie `-DGGML_NATIVE=OFF`. Eine Steigerung des Durchsatzes wurde nicht gemessen.

## Nur aktuelle Formate

Es wird nur GGUF v3 akzeptiert. `general.name` oder `general.basename` zusammen mit `general.size_label` muss ein zugelassenes Modell identifizieren; der Dateiname allein reicht nicht. Diese Richtlinie prüft Metadaten und Dimensionen, authentifiziert aber nicht die Gewichte. Eine Qwen3.5-27B-Datei wird nicht allein wegen gleicher Architekturdimensionen als Qwen3.8-27B akzeptiert.

Der Chat benötigt die vollständige Jinja-Vorlage des Modells oder `--chat-template-file`. Die alte Engine für feste Vorlagen, der Schalter `--jinja`/`--no-jinja`, veraltete C-APIs, alternative lange CLI-Schreibweisen und der alte Denkmodus wurden entfernt. Die aktuellen `/api/v1/*`-API-Verträge bleiben bestehen; die alten Aliase `/completion` und `/embedding` sind entfernt.

Die Weboberfläche verwendet den aktuellen Speicher-Namensraum `DgsUi` und strukturierte Nachrichten. Alte Namensräume, Designs, Nachrichtenmarkierungen und frühere JSON-Konversationsexporte werden weder gelesen noch migriert. Aktuelle JSONL-Sitzungsexporte und ZIP-Archive werden unterstützt. Alte Browserdaten werden nicht gelöscht. MCP verwendet StreamableHTTP oder WebSocket; der frühere SSE-Transport und der automatische Rückfall darauf wurden entfernt. Die Zwischenablage benötigt die aktuelle Clipboard-API in einem sicheren Kontext.

## Struktur und Dokumentation

- `backend/`: Server, Qwen-Inferenz, gemeinsame Tensor-Kernels und Qwen-Bildverarbeitung.
- `frontend/`: aktuelle Svelte-Anwendung mit festgelegten Abhängigkeitsversionen.
- `deploy/`: Build-Hilfen und Einbettung der lokalen Weboberfläche.
- [Technische Übersicht](CODE.de-DE.md) und [Handbuch](MANUAL.de-DE.md).

Der normale Build kompiliert und integriert die lokale Weboberfläche. Um eine bereits mit `cd frontend && npm ci && npm run build` erzeugte Oberfläche wiederzuverwenden, konfigurieren Sie `-DDGS_USE_LOCAL_UI=ON`. Erzeugen Sie die Ressourcen nach Quelländerungen erneut. `-DDGS_BUILD_UI=OFF` lässt eingebettete Ressourcen weg; verwenden Sie dann `--path` mit einem aktuellen, bereits kompilierten Webverzeichnis.

Die Projektregeln verbieten Tests. Die Validierung beschränkt sich auf Kompilierung und statische Typprüfung; Modellinferenz und Benchmarks wurden nicht ausgeführt. Das Projekt verwendet Git nicht. Das geschützte Verzeichnis `.git` wird von seinem Eigentümer manuell gelöscht.

## Täglicher Betrieb

Starten Sie eine Konversation in der Weboberfläche, wählen Sie ein geladenes, zugelassenes Modell und senden Sie Text oder unterstützte Bild-/Videoanhänge. Einstellungen steuern Sampling, Kontext und Werkzeuge. `-c` setzt die Kontextlänge, `-t` die CPU-Threads und `-ngl auto` die automatische GPU-Platzierung. Verwenden Sie `--api-key` für die Authentifizierung. Kompatible Clients verbinden sich mit `http://127.0.0.1:8080/api/v1` und `/api/v1/chat/completions`. Modellverzeichnisse dürfen andere Dateien enthalten; der Loader weist nicht unterstützte Checkpoints zurück.

Prüfen Sie bei Ladefehlern Identität, Architektur, Dimensionen und GGUF-Version. Deklarieren Sie ein anderes Modell nicht um, um die Prüfung zu umgehen. Reduzieren Sie bei Speichermangel Kontext, Batchgröße oder GPU-Auslagerung. Fehlt die Oberfläche, bauen Sie sie neu oder geben Sie `--path` an.

## MCP und Webfunktionen

Aktivieren Sie den integrierten Shell-MCP-Server mit `--local-shell-mcp-server` oder `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1`. Die Oberfläche erkennt ihn automatisch und fragt vor Werkzeugaufrufen nach Erlaubnis. Er bietet die installierten Interpreter Bash, Zsh, sh und PowerShell an, zeigt Befehl, stdout und stderr und protokolliert jeden Aufruf als JSONL.

Die Oberfläche bietet CPU/GPU- und Speicherüberwachung, eine Statusleiste, serverseitig gespeicherte gemeinsame Einstellungen, 15 Oberflächensprachen, eine optionale Anweisung zur Antwort in der gewählten regionalen Sprache und Warteschlangenhinweise, die Webanfragen von anderen API-Clients unterscheiden. Modellübersicht, Downloads, Konversationsregister und die Beschränkung auf dichte Modelle bleiben erhalten.

Die Oberfläche wird unter `/` bereitgestellt, die API standardmäßig unter `/api`. OpenAI-kompatible Clients müssen `/api/v1` als Basis verwenden. Swagger ist unter `/api/doc/` verfügbar. Die neuen API-Pfade besitzen keine Aliase im Wurzelpfad. `--api-prefix` wählt einen Pfad unter `/api`; `/api/config.js` veröffentlicht ihn für die Oberfläche.

## Lokaler Shell-MCP

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

Öffnen Sie die Oberfläche und geben Sie den API-Schlüssel ein. Der Eintrag `local shell mcp server` erscheint automatisch bei Aktivierung auf dem Server; er lässt sich global oder pro Konversation ausschalten. Die Werkzeuge `bash.run`, `zsh.run`, `sh.run` und `psh.run` werden nur für installierte Interpreter angeboten. Der Endpunkt ist `POST /api/mcp/shell`, mit MCP Streamable HTTP, JSON-Antworten und Protokollversion `2025-11-25`.

Jedes Werkzeug akzeptiert `command`, optional `cwd` und optional `timeout` in Sekunden: standardmäßig 60, höchstens 3600. Aufrufe laufen in unabhängigen Prozessen ohne interaktives Terminal. Verzeichnis- und Umgebungsänderungen bleiben nicht zwischen Aufrufen erhalten. Abbruch oder Zeitüberschreitung beendet die Prozessgruppe. Befehle haben die Rechte des Serverkontos; schützen Sie den Zugriff mit einem API-Schlüssel.

Im Chat erscheinen stdin (der Befehl), stdout und stderr getrennt mit dem Rückgabecode. Die Anzeigeeinstellungen können diese Blöcke automatisch aufklappen. Protokolle unter `~/.dgs/local-shell-mcp-server/` enthalten Befehl und Ausgabe vollständig, eine datierte JSONL-Datei je Aufruf; lange Ausgaben können im Chat gekürzt werden. Die Dateien sind nur für das Serverkonto zugänglich und werden nicht automatisch rotiert.

Auf Linux-Systemen mit bereits vorhandenem sudo erscheinen dessen Passwortabfragen in einem fokussierten Dialog mit Bestätigen/Abbrechen. Das Passwort geht nur an den Helfer, nicht an Modell, Konversation oder Protokolle. Während der Eingabe wartet auch die Befehlsfrist; unbeantwortete Anfragen verfallen nach fünf Minuten. Das Projekt installiert sudo nicht und benötigt es nicht für den Produktionsbetrieb.

## Überwachung und gemeinsame Einstellungen

Öffnen Sie **Monitor** in der Seitenleiste oder klicken Sie auf die untere Statusleiste. CPU- und GPU-Auslastung werden bei sichtbarer Seite alle zwei Sekunden aktualisiert. Die vollständige Ansicht ergänzt RAM, VRAM, Temperatur und Leistungsaufnahme, soweit der Treiber sie meldet. Speicher wird nach dichten Gewichten, MTP-/Entwurfsgewichten, KV-/rekurrenten Caches, Rechenpuffern, gespeicherten Prompts/Checkpoints, sonstiger Nutzung und freiem Speicher aufgeteilt. NVIDIA wird dynamisch über NVML, AMD über sysfs abgefragt; nicht verfügbare Messwerte bleiben unbekannt. Der Router aggregiert geladene Kindprozesse, ohne Modelle zu laden oder aufzuwecken.

Einstellungen liegen in `~/.dgs/config/ui-settings.json` des Serverkontos mit Modus 0600. Alle Browser desselben Servers teilen Konfiguration, Design, Sprache, Favoriten und Werkzeugpräferenzen. Speichern ersetzt das gesamte Dokument; bei gleichzeitigen Änderungen gilt der letzte erfolgreiche Speichervorgang. API-Schlüssel bleiben im Browser, ebenso Konversationen und vorübergehende Navigations-/Streamzustände. Alte Browsereinstellungen werden nicht migriert.

Wählen Sie die Sprache unter Allgemein. Die Änderung speichert die Auswahl und lädt die Oberfläche neu; speichern Sie vorher andere offene Änderungen. Verfügbare Gebietsschemata sind `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU` und `zh-CN`. US-Englisch ist Standard und Ersatz für unübersetzte Texte. **In der Sprache der Oberfläche antworten** (Respond in the interface language) ergänzt künftige Chatanfragen, Werkzeugrunden, Titel und KV-Vorbefüllung um eine regionale Anweisung. Vorhandene Nachrichten werden nicht übersetzt; die Befolgung hängt vom Modell ab.

Wartet eine Inferenzanfrage auf einen Slot, zeigt ihr Status die Anzahl vorheriger Anfragen und ob darunter direkte API-Anfragen sind. Normale API-Clients behalten ihr Antwortformat. Nur Streaming-Anfragen mit `"dgs_client": "webui"` erhalten `dgs_queue`-Ereignisse.

Für die Frontend-Entwicklung leitet `npm run dev` den Pfad `/api` an `http://localhost:9931` weiter; setzen Sie `VITE_PUBLIC_SERVER_ORIGIN` für einen anderen Server. Der Produktionsbuild bettet Oberfläche und Swagger lokal ein und benötigt kein externes Dokumentations-CDN.

## Quell- und Build-Verzeichnisse

`build/` enthält den CMake-Einstieg und `build-local.sh`. Das Skript ermittelt das Projekt relativ zu seiner eigenen Position, konfiguriert Release und akzeptiert weitere CMake-Optionen. Abhängigkeiten müssen installiert sein. `DGS_BUILD_JOBS` ist standardmäßig 4. Konfigurieren Sie manuell mit `cmake -S build -B _/build`; `backend/CMakeLists.txt` definiert das Engine-Projekt. Erzeugte Dateien und `bin/dgs-server` gehören nach `_/build/`, getrennt von den Build-Skripten.

Nach der Kompilierung erfolgt die Produktionsinstallation als root mit `cmake --install _/build --prefix /usr/local`. Die CMake-Regeln für Webressourcen stehen in `deploy/scripts/ui-assets.cmake`; `frontend/sources.cmake` listet ihre Eingaben, und `frontend/embed.py` erzeugt die eingebetteten C++-Ressourcen. Das Haupt-README liegt in der Wurzel, regionale Fassungen in `doc/`.


<a id="license"></a>

## Lizenz und Hinweise zu Drittanbietern

Die Beiträge von NiPeGun stehen unter [PolyForm Noncommercial 1.0.0](../LICENSE). Eingebundener Fremdcode behält seine eigenen Lizenzen und Urheberrechtshinweise; sie sind unter [build/licenses/](../build/licenses/README.md) gesammelt.

Die Hauptdatei LICENSE bleibt wortgetreu erhalten. Insbesondere bleibt der MIT-Hinweis der ggml-Autoren in `build/licenses/LICENSE-llama.cpp`; Swagger UI behält Apache-Lizenz und NOTICE. Das Verzeichnis enthält außerdem Hinweise für integrierte HTTP-, JSON-, Hash-, Medien-, subprocess-, base64-, CPU-Kernel- und Mathematikkomponenten. Hinweise in Abhängigkeitsquellen und Webpaketen gelten weiterhin. Modellgewichte und Systembibliotheken haben eigene Bedingungen.

`cmake --install _/build --prefix PREFIX` kopiert die Projektlizenz und `build/licenses/` bei Standard-Datenpfaden nach `PREFIX/share/doc/dgs-server/licenses/`. Mit dem Standardpräfix `/usr/local` landen sie unter `/usr/local/share/doc/dgs-server/licenses/`; für eine Benutzerinstallation kann `~/.local` verwendet werden. Die Komponente `Licenses` lässt sich separat installieren. Legen Sie diese Dokumente bei manueller Weitergabe der Programmdatei bei.
