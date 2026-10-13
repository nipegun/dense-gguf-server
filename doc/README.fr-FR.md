# dense-gguf-server

Ce README est également disponible dans les langues suivantes :\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), **fr-FR**, [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Présentation

Exécutez localement des modèles GGUF denses au moyen d'un chat web et d'une API HTTP d'inférence.

Comme llama.cpp, mais façonné avec une précision de scalpel nipegunienne pour atteindre la vitesse d'inférence la plus élevée possible.

## Captures d'écran

Aucune capture d'écran n'a encore été ajoutée.

## Déploiement

Sous **Debian**, après avoir compilé le projet, exécutez ces commandes en tant que `root` depuis son répertoire racine :

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

Ouvrez `http://127.0.0.1:8080`. L'API utilise `/api`, les clients compatibles OpenAI utilisent `/api/v1` et Swagger se trouve à `/api/doc/`. Les poids des modèles sont fournis séparément ; utilisez `--api-key` lorsque vous accordez l'accès à d'autres clients.

## Compilation

Installez d'abord les dépendances, en tant que `root` sous Debian 13 :

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

Exécutez ensuite les commandes depuis le répertoire du projet avec votre utilisateur habituel :

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

Le script compile les sources locales dans `_/build/`. `DGS_BUILD_JOBS=8` définit le nombre de tâches ; les arguments supplémentaires sont transmis à CMake, par exemple `-DGGML_CUDA=ON` ou `-DGGML_VULKAN=ON` si le SDK correspondant est installé. Les [dépendances et options de compilation](MANUAL.fr-FR.md#local-build) doivent être disponibles au préalable.

## Consulter la documentation

À consulter :

- [doc/MANUAL.fr-FR.md](MANUAL.fr-FR.md) pour compiler, installer et utiliser DGS, configurer MCP et l'interface web ou diagnostiquer des problèmes.
- [doc/CODE.fr-FR.md](CODE.fr-FR.md) pour comprendre le moteur dense, l'organisation des sources et les contrats de compilation du backend et du frontend.

## Soutenir ce projet

- **[Faites appel à moi](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)** pour ajouter de nouvelles fonctionnalités.
- **GitHub Sponsors.** Vous pouvez également soutenir financièrement le projet sur GitHub : [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
