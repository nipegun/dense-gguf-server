# Manuel de DGS

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · **fr-FR** · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · [ja-JP](MANUAL.ja-JP.md) · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server est un projet indépendant développé et maintenu par **NiPeGun**.

DGS sert les modèles denses suivants et l'application web de chat associée :

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

Tous les autres modèles de langage, y compris toutes les versions de GLM et Kimi, ne sont pas pris en charge. Les modèles MoE restent exclus. Le serveur vérifie l'identité GGUF d'origine et les dimensions des couches, y compris lors du chargement d'un modèle brouillon spéculatif. Les quantifications des modèles autorisés restent prises en charge.

<a id="local-build"></a>

## Compiler et exécuter sous Debian

Utilisez Debian 13 ou un système compatible avec C++17, CMake 3.19+, Python 3.13+ et Node.js 20.19+ ou 22.12+. Installez les paquets en tant que root :

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

Ouvrez `http://127.0.0.1:8080`. Les poids sont fournis séparément. Pour les images et les vidéos, indiquez le projecteur Qwen correspondant avec `--mmproj` ; le traitement vidéo utilise aussi FFmpeg. Les modèles audio et la génération audio ne sont pas pris en charge.

Le mode Release, les noyaux CPU natifs et l'optimisation à l'édition de liens, lorsqu'elle est disponible, sont activés par défaut. Pour CUDA ou Vulkan, utilisez `-DGGML_CUDA=ON` ou `-DGGML_VULKAN=ON` avec le SDK correspondant. Compilez pour le CPU de déploiement ou utilisez `-DGGML_NATIVE=OFF`. Aucun gain de débit n'a été mesuré.

## Formats actuels uniquement

Seul GGUF v3 est accepté. `general.name`, ou `general.basename` avec `general.size_label`, doit identifier un modèle autorisé ; le nom du fichier ne suffit pas. Cette politique vérifie les métadonnées et les dimensions, sans authentifier les poids. Un fichier Qwen3.5-27B n'est pas accepté comme Qwen3.8-27B au seul motif que leurs architectures ont les mêmes dimensions.

Le chat exige le modèle Jinja complet du modèle ou `--chat-template-file`. L'ancien moteur de modèles fixes, l'option `--jinja`/`--no-jinja`, les API C obsolètes, les variantes anciennes des options longues et l'ancien mode de raisonnement ont été supprimés. L'API conserve ses contrats actuels `/api/v1/*` ; les anciens alias `/completion` et `/embedding` ont disparu.

L'interface utilise l'espace de stockage actuel `DgsUi` et des messages structurés. Elle ne lit ni ne migre les anciens espaces, thèmes, marqueurs de messages ou exports JSON historiques. Les exports de sessions JSONL actuels et les archives ZIP sont pris en charge. Les anciennes données du navigateur ne sont pas supprimées. MCP utilise StreamableHTTP ou WebSocket ; l'ancien transport SSE et le repli automatique ont été retirés. Le presse-papiers exige l'API Clipboard actuelle dans un contexte sécurisé.

## Organisation et documentation

- `backend/` : serveur, inférence Qwen, noyaux tensoriels partagés et vision Qwen.
- `frontend/` : application Svelte actuelle et versions de dépendances fixées.
- `deploy/` : outils de compilation et intégration de l'interface web locale.
- [Carte technique](CODE.fr-FR.md) et [manuel](MANUAL.fr-FR.md).

La compilation normale compile et incorpore l'interface locale. Pour réutiliser une interface déjà générée avec `cd frontend && npm ci && npm run build`, configurez `-DDGS_USE_LOCAL_UI=ON`. Régénérez les ressources après avoir modifié leurs sources. `-DDGS_BUILD_UI=OFF` omet les ressources incorporées ; dans ce mode, utilisez `--path` avec un répertoire d'interface actuelle déjà compilée.

Les règles du projet interdisent les tests. La validation se limite à la compilation et à l'analyse statique des types ; aucune inférence ni mesure de performances n'a été exécutée. Ce projet n'utilise pas Git ; son propriétaire supprime manuellement le répertoire protégé `.git`.

## Utilisation quotidienne

Commencez une conversation dans l'interface, sélectionnez un modèle autorisé déjà chargé et envoyez du texte ou des pièces jointes image/vidéo compatibles. Les paramètres contrôlent l'échantillonnage, le contexte et les outils. `-c` définit la longueur du contexte, `-t` les threads CPU et `-ngl auto` le placement automatique sur GPU. Utilisez `--api-key` pour l'authentification. Les clients compatibles se connectent à `http://127.0.0.1:8080/api/v1` et `/api/v1/chat/completions`. Les répertoires de modèles peuvent contenir d'autres fichiers, mais le chargeur rejette les checkpoints non pris en charge.

Si le chargement échoue, vérifiez l'identité, l'architecture, les dimensions et la version GGUF ; ne renommez pas les métadonnées d'un autre modèle pour contourner la validation. En cas de manque de mémoire, réduisez le contexte, les lots ou le déport sur GPU. Si l'interface manque, recompilez-la ou indiquez `--path`.

## MCP et fonctions web

Activez le serveur MCP de shells intégré avec `--local-shell-mcp-server` ou `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1`. L'interface le détecte automatiquement et demande une autorisation pour appeler ses outils. Il expose les interpréteurs Bash, Zsh, sh et PowerShell installés, affiche la commande, stdout et stderr et enregistre chaque appel en JSONL.

L'interface comprend un moniteur CPU/GPU et mémoire, une barre d'état, des paramètres partagés conservés par le serveur, un sélecteur de 15 langues, une consigne facultative de réponse dans la variante régionale choisie et des avis de file distinguant les requêtes web des autres clients API. L'exploration et le téléchargement de modèles, les onglets de conversation et la politique de modèles denses sont conservés.

L'interface est servie à `/` ; l'API se trouve par défaut sous `/api`. Les clients compatibles OpenAI doivent utiliser `/api/v1` comme chemin de base. Swagger est disponible à `/api/doc/`. Les nouvelles routes n'ont pas d'alias à la racine. `--api-prefix` permet de choisir un chemin sous `/api` ; `/api/config.js` le communique à l'interface.

## MCP de shells locaux

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

Ouvrez l'interface et saisissez cette clé API. L'entrée `local shell mcp server` apparaît automatiquement si le serveur l'active ; elle peut être désactivée globalement ou pour une conversation. Les outils disponibles sont `bash.run`, `zsh.run`, `sh.run` et `psh.run`, uniquement pour les interpréteurs installés. Le point d'accès est `POST /api/mcp/shell`, avec MCP Streamable HTTP, des réponses JSON et la version de protocole `2025-11-25`.

Chaque outil accepte `command`, ainsi que `cwd` et `timeout` facultatifs ; le délai est en secondes, 60 par défaut et 3600 au maximum. Chaque appel utilise un processus indépendant sans terminal interactif. Les changements de répertoire et d'environnement ne persistent pas entre les appels. L'annulation ou l'expiration termine le groupe de processus. Les commandes ont les droits du compte du serveur ; protégez l'accès avec une clé API.

Le chat présente séparément stdin (la commande), stdout et stderr, avec le code de sortie. Les paramètres d'affichage peuvent ouvrir ces blocs automatiquement. Les journaux de `~/.dgs/local-shell-mcp-server/` contiennent la commande et la sortie complètes, dans un fichier JSONL horodaté par appel ; le chat peut tronquer les longues sorties. Les fichiers sont privés au compte du serveur et ne sont pas renouvelés automatiquement.

Sur les machines Linux où sudo existe déjà, ses demandes de mot de passe apparaissent dans une boîte de dialogue active avec Accepter/Annuler. Le mot de passe est envoyé seulement à l'auxiliaire, jamais au modèle, à la conversation ou aux journaux. L'attente de l'utilisateur suspend le délai de la commande ; les demandes sans réponse expirent après cinq minutes. Le projet n'installe pas sudo et son fonctionnement en production n'en dépend pas.

## Moniteur et préférences partagées

Ouvrez **Moniteur** dans la barre latérale ou cliquez sur la barre d'état inférieure. L'usage CPU/GPU est actualisé toutes les deux secondes lorsque la page est visible. La vue complète ajoute RAM, VRAM, température et consommation si le pilote les fournit. La mémoire distingue les poids denses, MTP/brouillon, les caches KV/état récurrent, les tampons de calcul, les prompts/checkpoints conservés, les autres usages et la mémoire libre. NVIDIA utilise NVML dynamiquement et AMD utilise sysfs ; les mesures indisponibles restent inconnues. Le mode routeur agrège les processus chargés sans charger ni réveiller de modèles.

Les paramètres sont enregistrés dans `~/.dgs/config/ui-settings.json` du compte du serveur, avec le mode 0600. Tous les navigateurs de ce serveur partagent configuration, thème, langue, favoris et préférences d'outils. Chaque enregistrement remplace le document ; en cas de modifications simultanées, le dernier enregistrement réussi prévaut. Les clés API restent dans le navigateur, comme les conversations et l'état transitoire de navigation et des flux. Les anciens paramètres du navigateur ne sont pas migrés.

Choisissez la langue dans les paramètres généraux. Le changement enregistre le choix et recharge l'interface ; enregistrez d'abord les autres modifications en attente. Les locales proposées sont `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU` et `zh-CN`. L'anglais américain est la langue par défaut et de repli pour les textes non traduits. Activez **Répondre dans la langue de l'interface** (Respond in the interface language) pour ajouter une consigne régionale aux futures requêtes, tours d'outils, titres et préremplissages KV. Les messages existants ne sont pas traduits ; le respect de la consigne dépend du modèle.

Lorsqu'une requête attend un slot, son état indique combien de requêtes la précèdent et si certaines viennent directement de l'API. Les clients API ordinaires conservent leur format habituel. Seules les requêtes en streaming portant `"dgs_client": "webui"` reçoivent des événements `dgs_queue`.

En développement frontend, `npm run dev` relaie `/api` vers `http://localhost:9931` ; définissez `VITE_PUBLIC_SERVER_ORIGIN` pour utiliser un autre serveur. La compilation de production incorpore l'interface et Swagger localement, sans CDN de documentation externe.

## Répertoires des sources et de compilation

`build/` contient l'entrée CMake et `build-local.sh`. Le script résout le projet par rapport à sa propre position, configure Release et accepte d'autres options CMake. Les dépendances doivent être installées. `DGS_BUILD_JOBS` vaut 4 par défaut. Pour une configuration manuelle, lancez `cmake -S build -B _/build` ; `backend/CMakeLists.txt` définit le projet du moteur. Les fichiers générés et `bin/dgs-server` se trouvent dans `_/build/`, séparés des scripts de compilation.

Après compilation, l'installation en production en tant que root utilise `cmake --install _/build --prefix /usr/local`. Les règles CMake de préparation web sont dans `deploy/scripts/ui-assets.cmake` ; `frontend/sources.cmake` liste leurs entrées et `frontend/embed.py` génère les ressources C++ incorporées. Le README principal est à la racine et ses versions régionales dans `doc/`.


<a id="license"></a>

## Licence et mentions tierces

Les contributions de NiPeGun sont sous licence [PolyForm Noncommercial 1.0.0](../LICENSE). Le code tiers intégré conserve ses propres licences et mentions de copyright, regroupées dans [build/licenses/](../build/licenses/README.md).

Le LICENSE principal est conservé à l'identique. La mention MIT des auteurs de ggml reste notamment dans `build/licenses/LICENSE-llama.cpp` ; Swagger UI conserve sa licence Apache et son NOTICE. L'inventaire comprend aussi les composants HTTP, JSON, de hachage, multimédias, subprocess, base64, noyaux CPU et bibliothèques mathématiques incorporés. Les mentions dans les sources de dépendances et les paquets web restent applicables. Les poids des modèles et les bibliothèques système ont leurs propres conditions.

`cmake --install _/build --prefix PREFIX` copie le LICENSE principal et `build/licenses/` vers `PREFIX/share/doc/dgs-server/licenses/` avec l'organisation de données par défaut. Le préfixe `/usr/local` les installe donc dans `/usr/local/share/doc/dgs-server/licenses/` ; une installation utilisateur peut utiliser `~/.local`. Le composant `Licenses` peut être installé séparément. Joignez ces documents lorsque vous distribuez manuellement l'exécutable.
