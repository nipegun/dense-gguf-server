# DGS — карта кода

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · **ru-RU** · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server — независимый проект, который разрабатывает и поддерживает **NiPeGun**.

- [Архитектура](#architecture)
- [Модули](#modules)
- [Символы](#symbols)
- [Потоки вызовов](#flows)
- [Точки входа](#entrypoints)
- [Влияние изменений](#impact)
- [Расширение](#extension)
- [MCP, монитор и общие настройки интерфейса](#integrated-features)
- [Источники и проверка](#validation)
- [Организация репозитория](#repository-organization)

- [Лицензия и уведомления третьих сторон](#license)

<a id="architecture"></a>

## Архитектура

Один сервер и одна веб-программа. Фабрика создаёт только `dgs_model_qwen35`; Qwen3.8-27B использует ту же архитектуру. Ядра DeltaNet и проектор Qwen с общим визуальным помощником сохранены. MoE, старые версии GGUF и другие профили отвергаются. Список разрешённых моделей читает исходные метаданные независимо от HTTP-псевдонимов, затем проверяет слои, эмбеддинги, FFN и головы внимания. По одним размерностям нельзя различить Qwen3.5-27B и Qwen3.8-27B. Идентичность заявляется в GGUF, но не подтверждается криптографически. GGML и внешние зависимости сохраняют текущие примитивы и интерфейсы.

<a id="modules"></a>

## Модули

| Модуль | Путь | Назначение | Зависит от | Используется |
|---|---|---|---|---|
| Политика | `backend/src/dgs-model-policy.*` | Разрешённые модели Qwen | GGUF, hparams | загрузчик моделей |
| Qwen | `backend/src/models/` | Плотный граф | DeltaNet, GGML | сервер |
| Зрение | `backend/tools/mtmd/` | Изображения/видео Qwen | GGML, кодеки | сервер |
| Чат | `backend/common/` | Jinja, XML-инструменты, сэмплинг | модель, vendor | сервер |
| HTTP | `backend/tools/server/` | Маршруты, слоты, инструменты | common, mtmd, UI | исполняемый файл |
| Веб | `frontend/` | актуальный интерфейс и хранилище | закреплённые пакеты npm | браузер |
| Сборка | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | Компиляция и встраивание | компилятор, npm, Python | развёртывание |
| Shell MCP | backend/tools/server/server-shell-mcp.* | MCP-команды и запросы паролей | HTTP, POSIX, common_json | сервер, веб-интерфейс MCP |
| Монитор | backend/tools/server/server-monitor.* | Телеметрия CPU/GPU и агрегация памяти | procfs, sysfs, необязательный NVML | server-context, маршрутизатор, веб |
| Веса MTP | backend/src/dgs-ext.h; dgs-model.cpp | Разделение выделений плотных/MTP-весов | метаданные буферов GGML | снимки монитора |
| Общие настройки | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | Атомарный документ настроек | файловая система, API, SettingsService | все хранилища состояния веб-интерфейса |
| Документация API | backend/tools/server/server-api-doc.*; frontend/api.html | Актуальные OpenAPI и Swagger | реестр маршрутов, схема, vendor/swagger-ui | /api/doc/ |
| Языки | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 каталогов и инструкция для ответов | JSON, общие настройки | веб-компоненты, сервис чата |
| Уведомления очереди | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | События очереди по запросу | очереди задач и ответов | интерфейс потокового чата |
| Веб-ресурсы | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | Сборка и встраивание интерфейса | npm, Python | `dgs-ui` |
| Распространение лицензий | `backend/CMakeLists.txt`, `build/licenses/` | Разделение условий проекта и третьих сторон | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## Символы

| Символ | Файл:строка | Назначение |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | Проверить идентичность GGUF |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | Проверить плотные размерности |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | Отклонить тензоры MoE |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | Создать разрешённую модель |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Загрузить размерности Qwen |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Обработать Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Построить граф зрения Qwen |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | Запустить сервер и зарегистрировать маршруты |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | Обработать JSON-RPC MCP |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | Проверить и выполнить shell-команду |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | Передать запрос пароля sudo |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | Объединить данные оборудования и памяти моделей |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | Опубликовать сегменты плотной памяти |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | Подсчитать добавленные веса MTP |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | Атомарно сохранить общие настройки |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | Предоставить OpenAPI и Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | Передать позицию в очереди |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | Добавить региональную языковую инструкцию |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | Стереть использованные строки паролей |

<a id="flows"></a>

## Потоки вызовов

Загрузка: `fDgsModelCreate` → запрет MoE → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → тензоры/граф. Чат: HTTP → очередь/слот → Jinja → токенизация → инференс → потоковый ответ. Веб: прямая инициализация схемы `DgsUi` без миграций. Сборка: npm → `frontend/dist` → `frontend/embed.py` → встроенная библиотека. Черновые модели проходят ту же политику; MTP использует предсказывающие слои разрешённой модели.

<a id="entrypoints"></a>

## Точки входа

| Маршрут | Обработчик | Файл |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | актуальные маршруты дополнений | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | актуальные API-маршруты | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | сервер/маршрутизатор | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | встроенный веб-интерфейс | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | снимки и агрегация маршрутизатора | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | Префикс и сохранённый язык | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## Влияние изменений

Политика влияет на все загрузки, включая черновики и загрузку только словаря. Переименование метаданных GGUF может помешать идентификации; не заменяйте их именами файлов. Исторические названия текущих GGUF-ключей или общих примитивов не включают старые модели. Не возвращайте удалённые API, миграции или псевдонимы. Изменения схемы требуют согласованного обновления frontend/backend.

<a id="extension"></a>

## Расширение

Новая модель требует явного решения по списку разрешений, архитектуре, метаданным и размерностям. Не расширяйте распознавание только по имени и не возвращайте MoE. Регистрируйте новые конечные точки на сервере и обновляйте типы интерфейса. Поддерживайте README, MANUAL и CODE на всех 15 языках интерфейса.

<a id="validation"></a>

## Источники и проверка

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

Указанные официальные конфигурации определяют профили. Проверка ограничена компиляцией и статическим анализом: правила запрещают тесты. Инференс и измерения пропускной способности не выполнялись. Защищённый каталог `.git` удаляет вручную владелец; агент не использует Git.



<a id="integrated-features"></a>

## MCP, монитор и общие настройки интерфейса

Shell: параметр CLI → обнаружение оболочек → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → проверка разрешения → `tools/call` → отдельная группа процессов → JSON-результат и JSONL каждого вызова → показ stdin/stdout/stderr. Канал askpass использует идентификатор вызова и приватный Unix-сокет; диалог никогда не включает пароли в сообщения модели. `common_json::mWipeString` перезаписывает чувствительные JSON-строки помощника после использования.

Монитор: загрузка → разделение весов MTP → снимок основного цикла, не чаще раза в секунду при инференсе → копия под mutex → `/api/monitor*` → опрос видимой страницы каждые две секунды. Веса основной и отдельной черновой моделей считаются по одному разу. Буферы, кэши и предвыборка экспертов в монитор не включены. Опрос не создаёт задач инференса и не будит модели. Использование оборудования не зависит от снимка инференса.

Настройки: API-ключ браузера → `/api/ui/settings` → `UiSettingsStore.mLoad` → инициализация хранилищ → сохранение целого документа после короткой задержки от последнего изменения. Документ содержит `config`, `userOverrides`, `language` и `preferences`; значения сохраняют сериализацию своего хранилища. Обмен между браузерами происходит при перезагрузке, без совместного редактирования в реальном времени; действует последнее сохранение. Диалоги, вкладки и курсоры возобновления потоков остаются локальными.

Язык: `/api/config.js` → JSON-каталог → интерфейс. `fApplyResponseLanguage` изменяет только копии исходящих сообщений; его используют чат, инструменты, заголовки и предварительное заполнение. Дополнительные строки DGS включают en-US/en-GB/es-ES/es-AR; остальные локали используют en-US при отсутствии записи в импортированном каталоге.

Очередь: `dgs_client=webui` → схема дополнений → счётчики источников → `server_task_result_queue` → SSE `dgs_queue` → `QueueStore` для каждого диалога → метка ожидания. Форматы событий OpenAI Responses и Anthropic не получают этих дополнительных событий. API-клиенты без метки не затрагиваются.

Базовый путь API — `/api`, его можно задавать только под `/api`; интерфейс остаётся на `/`. Обработчики получают относительные пути, дочерние процессы всегда используют `/api`, а интерфейс подставляет опубликованный префикс. API идёт в сеть, а не в кэш service worker. Меняя пути, синхронно обновляйте проксирование маршрутизатора, формирование URL клиента и OpenAPI.

Расширение: добавляйте кандидата оболочки вместе с именем frontend-инструмента; сегменты памяти — одновременно в плотный снимок, API-тип и цветовые группы; строки — в JSON-каталоги. Сохраняйте список плотных моделей и актуальные форматы. После изменений компилируйте backend и frontend; по правилам не добавляйте и не запускайте тесты.


Проверка встроенных функций: CPU Release с LTO и 70 встроенными веб-ресурсами собран успешно. Svelte/TypeScript: ноль ошибок и предупреждений. Веб-пакет превышает рекомендованный размер; GCC/CMake также сообщили о существующих предупреждениях и расхождении часов файловой системы. Тесты, запуск сервера, инференс и GPU-сборки не выполнялись.

<a id="repository-organization"></a>

## Организация репозитория

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

Сборка: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, frontend, сервер. Интерфейс: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → встроенная библиотека. Обёртка и backend могут быть корнями исходников CMake. Перемещение требует согласованно обновить include, CMake-пути, зависимости и документацию.


Проверка структуры: CMake успешно настроен из build/ и backend/. CPU Release с LTO и интерфейсом успешно собран через build/. Svelte/TypeScript без ошибок и предупреждений; проверены Bash-синтаксис и локальные ссылки. Тесты и инференс не выполнялись.


<a id="license"></a>

## Лицензия и уведомления третьих сторон

Вклад NiPeGun распространяется по лицензии [PolyForm Noncommercial 1.0.0](../LICENSE). Включённый сторонний код сохраняет собственные лицензии и уведомления об авторских правах, собранные в [build/licenses/](../build/licenses/README.md).

Главный LICENSE сохраняется дословно. В частности, уведомление MIT авторов ggml находится в `build/licenses/LICENSE-llama.cpp`; Swagger UI сохраняет лицензию Apache и NOTICE. Перечень также включает встроенные компоненты HTTP, JSON, хэширования, мультимедиа, subprocess, base64, ядер CPU и математических библиотек. Уведомления в исходниках зависимостей и веб-пакетах остаются действующими. Веса моделей и системные библиотеки имеют собственные условия.

`cmake --install _/build --prefix PREFIX` копирует LICENSE проекта и `build/licenses/` в `PREFIX/share/doc/dgs-server/licenses/` при стандартном расположении данных. С префиксом `/usr/local` они устанавливаются в `/usr/local/share/doc/dgs-server/licenses/`; пользовательская установка может использовать `~/.local`. Компонент `Licenses` устанавливается и отдельно. При ручном распространении исполняемого файла приложите эти документы.

Распространение: LICENSE проекта и сторонние уведомления → компонент CMake `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`. Изменение зависимости требует пересмотра её уведомления. Имя продукта и API-типы используют DGS; атрибуции третьих сторон и идентификаторы GGUF не переименовываются. Неиспользуемый адаптер внешнего каталога и неподдерживаемые пресеты удалены; поиск моделей использует существующий сервис Hugging Face.

Проверка идентичности и лицензий: CPU Release с LTO и интерфейсом собран успешно; Svelte/TypeScript без ошибок и предупреждений. Созданные правила установки CMake включают главную лицензию и собранные уведомления. LICENSE совпадает с эталоном побайтно. Тесты, инференс и установка в производственную среду не выполнялись.
