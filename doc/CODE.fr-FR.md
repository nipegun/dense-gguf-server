# DGS — carte du code

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · **fr-FR** · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server est un projet indépendant développé et maintenu par **NiPeGun**.

- [Architecture](#architecture)
- [Modules](#modules)
- [Symboles](#symbols)
- [Flux](#flows)
- [Points d'entrée](#entrypoints)
- [Impact](#impact)
- [Extensions](#extension)
- [MCP, moniteur et interface partagée](#integrated-features)
- [Sources et validation](#validation)
- [Organisation du dépôt](#repository-organization)

- [Licence et mentions tierces](#license)

<a id="architecture"></a>

## Architecture

Un serveur et une application web. La fabrique instancie uniquement `dgs_model_qwen35` ; Qwen3.8-27B partage cette architecture. Les noyaux DeltaNet et le projecteur Qwen avec son auxiliaire de vision commun sont conservés. MoE, les anciennes versions GGUF et les autres profils sont rejetés. La liste autorisée lit les métadonnées originales indépendamment des alias HTTP, puis vérifie couches, embeddings, FFN et têtes. Les dimensions seules ne distinguent pas Qwen3.5-27B de Qwen3.8-27B. L'identité déclarée dans le GGUF n'est pas authentifiée cryptographiquement. GGML et les dépendances externes conservent leurs primitives et interfaces actuelles.

<a id="modules"></a>

## Modules

| Module | Chemin | Responsabilité | Dépend de | Utilisé par |
|---|---|---|---|---|
| Politique | `backend/src/dgs-model-policy.*` | Liste Qwen autorisée | GGUF, hparams | chargeur de modèles |
| Qwen | `backend/src/models/` | Graphe dense | DeltaNet, GGML | serveur |
| Vision | `backend/tools/mtmd/` | Images/vidéos Qwen | GGML, codecs | serveur |
| Chat | `backend/common/` | Jinja, outils XML, échantillonnage | modèle, vendor | serveur |
| HTTP | `backend/tools/server/` | Routes, slots, outils | common, mtmd, UI | exécutable |
| Web | `frontend/` | interface et stockage actuels | versions npm fixées | navigateur |
| Compilation | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | Compiler et incorporer | compilateur, npm, Python | déploiement |
| MCP de shells | backend/tools/server/server-shell-mcp.* | Commandes MCP et demandes de mot de passe | HTTP, POSIX, common_json | serveur, interface MCP |
| Moniteur | backend/tools/server/server-monitor.* | Télémétrie CPU/GPU et agrégation mémoire | procfs, sysfs, NVML facultatif | server-context, routeur, web |
| Poids MTP | backend/src/dgs-ext.h; dgs-model.cpp | Répartition des allocations denses/MTP | métadonnées des tampons GGML | instantanés du moniteur |
| Paramètres partagés | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | Document de paramètres atomique | système de fichiers, API, SettingsService | tous les stores de l'interface |
| Documentation API | backend/tools/server/server-api-doc.*; frontend/api.html | OpenAPI et Swagger actualisés | registre des routes, schéma, vendor/swagger-ui | /api/doc/ |
| Langues | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 catalogues et consigne de réponse | JSON, paramètres partagés | composants web, service de chat |
| Avis de file | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | Événements de file sur demande | files de tâches et de réponses | interface du chat en streaming |
| Ressources web | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | Compiler et incorporer l'interface | npm, Python | `dgs-ui` |
| Distribution des licences | `backend/CMakeLists.txt`, `build/licenses/` | Séparer les conditions du projet de celles des tiers | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## Symboles

| Symbole | Fichier:ligne | Fonction |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | Valider l'identité GGUF |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | Valider les dimensions denses |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | Rejeter les tenseurs MoE |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | Créer un modèle autorisé |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Charger les dimensions Qwen |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Rendre Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Créer le graphe de vision Qwen |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | Démarrer le serveur et enregistrer les routes |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | Distribuer les appels JSON-RPC MCP |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | Valider et exécuter une commande shell |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | Transmettre les demandes de mot de passe sudo |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | Agréger matériel et mémoire des modèles |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | Publier les segments de mémoire dense |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | Compter les poids MTP annexes |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | Enregistrer atomiquement les paramètres partagés |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | Servir OpenAPI et Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | Émettre la position dans la file |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | Ajouter la consigne de langue régionale |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | Effacer les chaînes de mot de passe utilisées |

<a id="flows"></a>

## Flux

Chargement : `fDgsModelCreate` → rejet MoE → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → tenseurs/graphe. Chat : HTTP → file/slot → Jinja → tokenisation → inférence → réponse en streaming. Web : initialisation directe du schéma `DgsUi`, sans migration. Compilation : npm → `frontend/dist` → `frontend/embed.py` → bibliothèque incorporée. Les brouillons suivent la même politique ; MTP utilise les couches de prédiction du modèle autorisé.

<a id="entrypoints"></a>

## Points d'entrée

| Route | Gestionnaire | Fichier |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | routes de complétion actuelles | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | routes API actuelles | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | serveur/routeur | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | interface incorporée | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | instantanés et agrégation du routeur | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | Préfixe et langue enregistrée | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## Impact

La politique affecte tous les chargements, y compris ceux de brouillons ou du seul vocabulaire. Renommer les métadonnées GGUF peut empêcher l'identification ; ne les remplacez pas par des noms de fichiers. Les noms historiques qui restent des clés GGUF ou des primitives communes n'activent pas les anciens modèles. Ne rétablissez pas les API, migrations ou alias supprimés. Toute modification de schéma exige une coordination frontend/backend.

<a id="extension"></a>

## Extensions

L'ajout d'un modèle exige une décision explicite sur la liste autorisée, son architecture, ses métadonnées et ses dimensions. N'élargissez pas la correspondance au seul nom et ne réintroduisez pas MoE. Enregistrez les nouveaux points d'accès dans le serveur et mettez à jour les types web. Maintenez README, MANUAL et CODE dans les 15 langues de l'interface.

<a id="validation"></a>

## Sources et validation

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

Les configurations officielles ci-dessus définissent les profils. La validation utilise uniquement compilation et analyse statique : les règles interdisent les tests. Aucune inférence ni mesure de débit n'a été exécutée. Le propriétaire supprime manuellement le répertoire protégé `.git` ; l'agent n'utilise pas Git.



<a id="integrated-features"></a>

## MCP, moniteur et interface partagée

Shell : option CLI → détection des shells → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → autorisation → `tools/call` → groupe de processus indépendant → résultat JSON et JSONL par appel → affichage stdin/stdout/stderr. Le canal askpass utilise un identifiant d'appel et un socket Unix privé ; le dialogue n'inclut jamais les mots de passe dans les messages du modèle. `common_json::mWipeString` écrase les chaînes JSON sensibles de l'auxiliaire après utilisation.

Moniteur : chargement → répartition des poids MTP → instantané de la boucle principale, au plus une fois par seconde durant l'inférence → copie protégée par mutex → `/api/monitor*` → interrogation toutes les deux secondes lorsque la page est visible. Les poids du modèle principal et du brouillon séparé sont comptés une seule fois. Aucun tampon, cache ou préchargement d'experts n'est intégré à ce moniteur. L'interrogation ne crée pas de tâches d'inférence et ne réveille pas les modèles. L'usage matériel est indépendant de l'instantané d'inférence.

Paramètres : clé API du navigateur → `/api/ui/settings` → `UiSettingsStore.mLoad` → initialisation des stores → enregistrement différé du document complet après stabilisation des changements. Le document contient `config`, `userOverrides`, `language` et `preferences` ; chaque préférence conserve la sérialisation de son store. Le partage prend effet au rechargement, sans édition collaborative en direct ; le dernier document enregistré prévaut. Conversations, onglets et curseurs de reprise des flux restent locaux.

Langue : `/api/config.js` → catalogue JSON → interface. `fApplyResponseLanguage` modifie seulement les copies des messages sortants ; chat, outils, titres et préremplissage l'utilisent. Les nouveaux textes DGS couvrent en-US/en-GB/es-ES/es-AR ; les autres locales utilisent en-US lorsqu'une entrée manque dans leur catalogue importé.

File : `dgs_client=webui` → schéma de complétion → compteurs par origine → `server_task_result_queue` → SSE `dgs_queue` → `QueueStore` par conversation → libellé d'attente. Les formats d'événements OpenAI Responses et Anthropic ne reçoivent pas ces avis supplémentaires. Les clients API non marqués ne sont pas affectés.

Base API : `/api` par défaut, configurable seulement sous `/api` ; l'interface reste à `/`. Les handlers reçoivent des chemins relatifs, les processus enfants utilisent toujours `/api` et l'interface résout le préfixe publié. Les appels API passent par le réseau, pas par le cache du service worker. Les changements de chemins exigent de mettre à jour ensemble le transfert du routeur, les URL du client et OpenAPI.

Extension : ajoutez ensemble un shell candidat et son nom d'outil frontend ; ajoutez ensemble les segments de mémoire, leur type API et leur groupe de couleurs ; ajoutez les textes aux catalogues JSON. Conservez la liste de modèles denses et les formats actuels. Recompilez backend et frontend après modification ; les règles interdisent d'ajouter ou d'exécuter des tests.


Validation des fonctions intégrées : compilation CPU Release avec LTO et 70 ressources web incorporées réussie. Svelte/TypeScript : zéro erreur et zéro avertissement. Le paquet web dépasse le seuil indicatif de taille ; GCC/CMake ont aussi signalé des avertissements existants et un décalage d'horloge du système de fichiers. Aucun test, lancement du serveur, inférence ou build GPU n'a été effectué.

<a id="repository-organization"></a>

## Organisation du dépôt

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

Compilation : `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, frontend, serveur. Interface : `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → bibliothèque incorporée. Le wrapper et le backend peuvent tous deux être la racine CMake. Un déplacement exige de coordonner les includes, chemins CMake, listes de dépendances et documentation.


Validation de l'organisation : configuration CMake réussie depuis build/ et backend/. Compilation CPU Release avec LTO et interface incorporée réussie depuis build/. Svelte/TypeScript : zéro erreur et zéro avertissement ; syntaxe Bash et liens locaux vérifiés. Aucun test ni inférence n'a été exécuté.


<a id="license"></a>

## Licence et mentions tierces

Les contributions de NiPeGun sont sous licence [PolyForm Noncommercial 1.0.0](../LICENSE). Le code tiers intégré conserve ses propres licences et mentions de copyright, regroupées dans [build/licenses/](../build/licenses/README.md).

Le LICENSE principal est conservé à l'identique. La mention MIT des auteurs de ggml reste notamment dans `build/licenses/LICENSE-llama.cpp` ; Swagger UI conserve sa licence Apache et son NOTICE. L'inventaire comprend aussi les composants HTTP, JSON, de hachage, multimédias, subprocess, base64, noyaux CPU et bibliothèques mathématiques incorporés. Les mentions dans les sources de dépendances et les paquets web restent applicables. Les poids des modèles et les bibliothèques système ont leurs propres conditions.

`cmake --install _/build --prefix PREFIX` copie le LICENSE principal et `build/licenses/` vers `PREFIX/share/doc/dgs-server/licenses/` avec l'organisation de données par défaut. Le préfixe `/usr/local` les installe donc dans `/usr/local/share/doc/dgs-server/licenses/` ; une installation utilisateur peut utiliser `~/.local`. Le composant `Licenses` peut être installé séparément. Joignez ces documents lorsque vous distribuez manuellement l'exécutable.

Distribution : LICENSE du projet + mentions tierces → composant CMake `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`. Toute modification de dépendance exige de vérifier sa mention. Le nom du produit et les types API utilisent DGS ; les attributions tierces et identifiants GGUF ne sont pas renommés. L'adaptateur de catalogue externe inutilisé et les presets non pris en charge ont été supprimés ; la découverte de modèles utilise toujours le service Hugging Face existant.

Validation de l'identité et des licences : compilation CPU Release avec LTO et interface incorporée réussie ; Svelte/TypeScript ne signale ni erreur ni avertissement. L'installation CMake générée contient la licence principale et les mentions collectées. Le LICENSE principal est identique octet par octet à la référence. Aucun test, inférence ou déploiement en production n'a été exécuté.
