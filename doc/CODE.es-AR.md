# DGS — mapa técnico

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · **es-AR** · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server es un proyecto independiente desarrollado y mantenido por **NiPeGun**.

- [Arquitectura](#architecture)
- [Módulos](#modules)
- [Símbolos](#symbols)
- [Flujos](#flows)
- [Entradas](#entrypoints)
- [Impacto](#impact)
- [Extensión](#extension)
- [Fuentes y verificación](#validation)
- [MCP, monitor y web compartida](#integrated-features)

- [Organización del repositorio](#repository-organization)

- [Licencia y avisos de terceros](#license)

<a id="architecture"></a>

## Arquitectura

Un solo servidor y una web. La factoría sólo instancia `dgs_model_qwen35`; Qwen3.8-27B comparte esta arquitectura. Se conservan el núcleo DeltaNet y el proyector Qwen con su auxiliar visual compartido. Se rechazan MoE, GGUF antiguos y otros perfiles. La lista blanca utiliza metadatos originales, independientes de los alias HTTP; después se contrastan capas, embedding, FFN y cabezas. Las dimensiones no distinguen por sí solas Qwen3.5-27B de Qwen3.8-27B. La identidad sigue siendo declarada por el GGUF, no autenticada criptográficamente. GGML y las bibliotecas externas conservan sus primitivas e interfaces actuales.

<a id="modules"></a>

## Módulos

| Módulo | Ruta | Responsabilidad | Depende de | Usado por |
|---|---|---|---|---|
| Política | `backend/src/dgs-model-policy.*` | Lista de modelos Qwen permitidos | GGUF, hparams | Cargador de modelos |
| Qwen | `backend/src/models/` | Grafo denso | DeltaNet, GGML | Servidor |
| Visión | `backend/tools/mtmd/` | Imágenes y vídeo Qwen | GGML, codecs | Servidor |
| Chat | `backend/common/` | Jinja, herramientas XML y muestreo | Modelo, vendor | Servidor |
| HTTP | `backend/tools/server/` | Rutas, slots y herramientas | common, mtmd, UI | Ejecutable |
| Web | `frontend/` | Interfaz y almacenamiento actuales | Paquetes npm con versiones fijadas | Navegador |
| Compilación | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | Compilar e incorporar | Compilador, npm, Python | Despliegue |
| MCP de shells | backend/tools/server/server-shell-mcp.* | Comandos MCP y peticiones de contraseña | HTTP, POSIX, common_json | Servidor y web MCP |
| Monitor | backend/tools/server/server-monitor.* | Telemetría de CPU/GPU y agregación de memoria | procfs, sysfs y NVML opcional | server-context, router y web |
| Pesos MTP | backend/src/dgs-ext.h; dgs-model.cpp | Desglose de asignaciones densas/MTP | Metadatos de búferes GGML | Instantáneas del monitor |
| Ajustes compartidos | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | Documento de ajustes atómico | Sistema de archivos, API y SettingsService | Todos los stores de la web |
| Documentación API | backend/tools/server/server-api-doc.*; frontend/api.html | OpenAPI y Swagger actuales | Registro de rutas, esquema y vendor/swagger-ui | /api/doc/ |
| Idiomas | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 catálogos e instrucción de respuesta | JSON y ajustes compartidos | Componentes web y servicio de chat |
| Avisos de cola | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | Eventos de cola solicitados | Colas de tareas y respuestas | Interfaz de chat en streaming |
| Recursos web | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | Compilar e incorporar la web | npm, Python | `dgs-ui` |
| Distribución de licencias | `backend/CMakeLists.txt`, `build/licenses/` | Separar las licencias del proyecto y de terceros | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## Símbolos

| Símbolo | Archivo:línea | Función |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | Validar identidad GGUF |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | Validar dimensiones densas |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | Rechazar tensores MoE |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | Crear modelo admitido |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Cargar dimensiones Qwen |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Procesar Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Crear grafo de visión Qwen |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | Arrancar servidor y registrar rutas |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | Procesar JSON-RPC MCP |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | Validar y ejecutar un comando |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | Comunicar peticiones de contraseña |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | Agregar hardware y memoria de modelos |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | Publicar segmentos de memoria densa |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | Contar pesos MTP anexos |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | Guardar ajustes compartidos de forma atómica |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | Servir OpenAPI y Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | Emitir posición en la cola |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | Añadir instrucción de idioma regional |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | Borrar cadenas de contraseña utilizadas |

<a id="flows"></a>

## Flujos

Carga: `fDgsModelCreate` → rechazo MoE → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → tensores/grafo. Chat: HTTP → cola/slot → Jinja → tokenización → inferencia → respuesta progresiva. Web: inicialización directa del esquema `DgsUi`, sin migraciones. Compilación: npm → `frontend/dist` → `frontend/embed.py` → biblioteca incorporada. Los borradores pasan por la misma política; MTP utiliza las capas predictoras del modelo permitido.

<a id="entrypoints"></a>

## Entradas

| Ruta | Handler | Archivo |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | Rutas de completado actuales | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | Rutas API actuales | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | Servidor/router | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | Web incorporada | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | Instantáneas y agregación del router | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | Prefijo e idioma guardado | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## Impacto

La política afecta a todas las cargas, incluidos borradores y vocabularios. Cambiar nombres GGUF puede impedir identificar un modelo válido; no usar nombres de archivo como sustituto. Los nombres históricos que siguen siendo claves del GGUF actual o primitivas compartidas no implican soporte de modelos antiguos. No restaurar APIs, migraciones ni alias eliminados. Los cambios de esquemas requieren frontend/backend coordinados.

<a id="extension"></a>

## Extensión

Añadir otro modelo exige una decisión explícita sobre la lista blanca, su arquitectura, metadatos y dimensiones. No ampliar por coincidencias de nombre ni reintroducir MoE. Los endpoints nuevos se registran en el servidor y sus tipos se actualizan en la web. Mantener README, MANUAL y CODE en los 15 idiomas de la interfaz.

<a id="validation"></a>

## Fuentes y verificación

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

Las configuraciones oficiales anteriores sustentan los perfiles. Sólo se realizan compilación y análisis estático: las reglas prohíben tests. No se ha ejecutado inferencia ni medido rendimiento. La carpeta `.git` protegida se elimina manualmente por el propietario; el agente no utiliza Git.



Verificación final: compilación CPU Release con LTO y 70 recursos web incorporados correcta. Svelte/TypeScript: cero errores y cero avisos. No se ejecutaron tests, inferencia ni benchmarks; los backends GPU no se compilaron en este entorno.

<a id="integrated-features"></a>

## MCP, monitor y web compartida

Shell: opción de arranque → detección de intérpretes → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → autorización de la herramienta → `tools/call` → grupo de procesos independiente → resultado JSON y registro JSONL por llamada → presentación de stdin/stdout/stderr. El canal de contraseñas usa un identificador de llamada y un socket Unix privado. El diálogo nunca añade la contraseña a los mensajes del modelo. `common_json::mWipeString` sobrescribe las cadenas JSON sensibles del auxiliar después de usarlas.

Monitor: carga del modelo → desglose de pesos MTP → instantánea de memoria desde el bucle principal (como máximo una por segundo durante la inferencia) → copia protegida por mutex → `/api/monitor*` → consulta de la página visible cada dos segundos. Los pesos compartidos por modelo principal y borrador sólo se cuentan una vez. El monitor no contiene categorías ni búferes de expertos, caché de expertos o precarga. Consultarlo no crea tareas de inferencia ni despierta modelos. El uso del hardware se mide independientemente de las instantáneas del modelo.

Ajustes: clave API del navegador → `/api/ui/settings` → `UiSettingsStore.mLoad` → inicialización de los stores → guardado del documento completo tras un breve intervalo. El esquema actual contiene `config`, `userOverrides`, `language` y `preferences`; cada preferencia conserva la serialización de su store. Los navegadores comparten los ajustes al recargar: no se implementa edición colaborativa en directo y prevalece el último guardado completo. Las conversaciones, pestañas y posiciones de reanudación de flujos siguen siendo locales.

Idioma: `/api/config.js` → selección del catálogo JSON → interfaz. `fApplyResponseLanguage` sólo modifica copias de los mensajes salientes; se aplica a chat, herramientas, títulos y preparación de la caché. Los textos nuevos de DGS incluyen en-US/en-GB/es-ES/es-AR. Los demás idiomas utilizan en-US cuando su catálogo importado no contiene una entrada.

Cola: `dgs_client=webui` → esquema de peticiones → contadores por origen → `server_task_result_queue` → evento SSE `dgs_queue` → `QueueStore` por conversación → estado de la respuesta pendiente. Los formatos de eventos de OpenAI Responses y Anthropic no reciben estos avisos adicionales. Los clientes de API sin esa marca conservan el formato habitual.

Rutas: `/api` por defecto, configurable sólo dentro de `/api`; la web permanece en `/`. Los handlers reciben rutas relativas y los procesos de modelos del router usan siempre `/api`, sin heredar la clave de la API pública ni el MCP de shells. La web utiliza el prefijo publicado por el servidor. Las llamadas a la API van a la red y no a la caché del service worker. Un cambio de rutas debe coordinar router, resolución de URL del cliente y OpenAPI.

Extensión: para añadir un intérprete, actualizar sus candidatos y el nombre de herramienta del frontend. Para añadir un segmento de memoria, actualizar a la vez instantánea densa, tipo de API y colores. Añadir los textos a los catálogos JSON. Conservar la lista blanca de modelos densos y los formatos actuales. Compilar backend y web tras los cambios; las reglas del proyecto prohíben crear o ejecutar tests.


Validación de las funciones integradas: compilación CPU Release con LTO y 70 recursos web incorporados correcta. Svelte/TypeScript: cero errores y cero avisos. El paquete web supera el umbral informativo de tamaño; GCC/CMake también emitieron avisos existentes y de desfase horario del sistema de archivos. No se han ejecutado tests, el servidor, inferencia ni compilaciones GPU.

<a id="repository-organization"></a>

## Organización del repositorio

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

Flujo de compilación: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, frontend y servidor. Flujo web: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → biblioteca incorporada. Tanto el punto de entrada de compilación como el backend pueden ser raíces de CMake. Al mover archivos deben actualizarse los includes, las rutas CMake, las dependencias y la documentación conjuntamente.


Validación de la organización: CMake se configuró correctamente desde build/ y backend/. La compilación CPU Release con LTO y web incorporada terminó correctamente desde build/. Svelte/TypeScript informó de cero errores y cero avisos; se revisaron la sintaxis Bash y los enlaces locales de documentación. No se ejecutaron tests ni inferencia.


<a id="license"></a>

## Licencia y avisos de terceros

Las contribuciones de NiPeGun se distribuyen bajo [PolyForm Noncommercial 1.0.0](../LICENSE). El código de terceros incorporado conserva sus propias licencias y avisos de copyright, reunidos en [build/licenses/](../build/licenses/README.md).

El LICENSE principal se conserva literalmente. En particular, el aviso MIT de los autores de ggml permanece en `build/licenses/LICENSE-llama.cpp`; Swagger UI conserva su licencia Apache y su NOTICE. El inventario incluye también los componentes HTTP, JSON, hash, multimedia, subprocess, base64, kernels de CPU y bibliotecas matemáticas incorporados. Los avisos incluidos en fuentes de dependencias y paquetes web siguen siendo aplicables. Los pesos de modelos y las bibliotecas del sistema tienen sus propios términos.

`cmake --install _/build --prefix PREFIX` copia el LICENSE principal y `build/licenses/` a `PREFIX/share/doc/dgs-server/licenses/` con la disposición de datos predeterminada. Con `/usr/local`, la ruta es `/usr/local/share/doc/dgs-server/licenses/`; una instalación de usuario puede utilizar `~/.local`. El componente `Licenses` permite instalar esos documentos por separado. Al distribuir el ejecutable manualmente deben acompañarlo estos documentos.

Flujo de distribución: LICENSE principal + avisos de terceros → componente CMake `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`. Al cambiar una dependencia se revisa el aviso correspondiente. La identidad del producto y los tipos de propiedades de API usan DGS; las atribuciones de terceros y los identificadores GGUF no se renombran. Se han retirado el adaptador de catálogo externo sin consumidores y los presets de modelos no admitidos; el descubrimiento de modelos utiliza el servicio Hugging Face existente.

Validación de identidad y licencias: compilación CPU Release con LTO y web incorporada correcta; Svelte/TypeScript informó de cero errores y cero avisos. La instalación CMake generada incluye la licencia principal y los avisos recopilados. El LICENSE principal coincide byte por byte con la referencia. No se ejecutaron tests, inferencia ni instalación en producción.
