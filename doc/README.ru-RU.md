# dense-gguf-server

Этот README также доступен на следующих языках:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), **ru-RU**, [zh-CN](README.zh-CN.md)

## О проекте

Запускайте плотные модели GGUF локально через веб-чат и HTTP API инференса.

Как llama.cpp, но созданный с нипегунианской точностью скальпеля, чтобы достичь максимально возможной скорости инференса.

## Скриншоты

Скриншоты пока не добавлены.

## Развёртывание

В **Debian** после сборки проекта выполните эти команды от имени `root` из корневого каталога:

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

Откройте `http://127.0.0.1:8080`. API использует `/api`, OpenAI-совместимые клиенты — `/api/v1`, а Swagger расположен по адресу `/api/doc/`. Веса моделей предоставляются отдельно; при доступе других клиентов используйте `--api-key`.

## Сборка

Сначала установите зависимости от имени `root` в Debian 13:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

Затем выполните команды из каталога проекта под обычной учётной записью:

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

Скрипт собирает локальные исходники в `_/build/`. `DGS_BUILD_JOBS=8` задаёт число параллельных заданий. Остальные аргументы передаются CMake, например `-DGGML_CUDA=ON` или `-DGGML_VULKAN=ON` при установленном соответствующем SDK. [Зависимости и параметры сборки](MANUAL.ru-RU.md#local-build) должны быть подготовлены заранее.

## Документация

См. следующие документы:

- [doc/MANUAL.ru-RU.md](MANUAL.ru-RU.md) — сборка, установка и использование DGS, настройка MCP и веб-интерфейса, диагностика проблем.
- [doc/CODE.ru-RU.md](CODE.ru-RU.md) — плотный движок, организация исходников и контракты сборки backend и frontend.

## Поддержать проект

- **[Закажите у меня](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)** разработку новых функций.
- **GitHub Sponsors.** Также можно поддержать проект финансово на GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
