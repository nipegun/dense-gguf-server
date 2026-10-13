# Manual de DGS

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · **es-AR** · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · [ja-JP](MANUAL.ja-JP.md) · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server es un proyecto independiente desarrollado y mantenido por **NiPeGun**.

DGS sirve estos modelos densos y su aplicación web de chat:

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

No se admiten otros modelos de lenguaje, incluidas todas las versiones de GLM y Kimi. Los modelos MoE siguen prohibidos. Se comprueban la identidad GGUF original y las dimensiones de las capas, también al cargar un borrador especulativo. Se conservan las cuantizaciones de los modelos admitidos.

<a id="local-build"></a>

## Compilar y ejecutar en Debian

Usá Debian 13 o un sistema compatible con C++17, CMake 3.19+, Python 3.13+ y Node.js 20.19+ o 22.12+. Instalá los paquetes como root:

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

Abrí `http://127.0.0.1:8080`. Los pesos se proporcionan por separado. Para imágenes y vídeo, indicá el proyector Qwen correspondiente con `--mmproj`; el vídeo necesita además FFmpeg. No se admiten modelos de audio ni generación de audio.

El modo Release, los kernels nativos de CPU y la optimización durante el enlace están activados por defecto cuando se admiten. Para CUDA o Vulkan, usá `-DGGML_CUDA=ON` o `-DGGML_VULKAN=ON` y el SDK correspondiente. Compilá para la CPU de destino o usá `-DGGML_NATIVE=OFF`. No se ha medido una mejora de velocidad de inferencia.

## Sólo formatos actuales

Se exige GGUF v3. `general.name`, o `general.basename` junto con `general.size_label`, debe identificar un modelo permitido; no basta con el nombre del archivo. Esta política comprobá metadatos y dimensiones, no autentica los pesos. Un Qwen3.5-27B no se acepta como Qwen3.8-27B sólo porque sus arquitecturas tengan las mismas dimensiones.

El chat necesita la plantilla Jinja completa del modelo o `--chat-template-file`. Se han retirado el motor antiguo de plantillas fijas, `--jinja`/`--no-jinja`, las APIs C obsoletas, las grafías largas alternativas de argumentos y el modo de razonamiento antiguo. Se mantienen los contratos actuales `/api/v1/*`; ya no existen los alias `/completion` y `/embedding`.

La web usá el espacio de almacenamiento actual `DgsUi` y mensajes estructurados. No lee ni migra los espacios anteriores, temas, marcadores de mensajes ni exportaciones JSON antiguas. Admite sesiones JSONL actuales y archivos ZIP. No elimina los datos antiguos del navegador. MCP usá StreamableHTTP o WebSocket; se han retirado el transporte SSE antiguo y su selección automática como alternativa. El portapapeles necesita la API actual en un contexto seguro.

## Estructura y documentación

- `backend/`: servidor, inferencia Qwen, kernels compartidos y visión Qwen.
- `frontend/`: aplicación Svelte actual y dependencias fijadas.
- `deploy/`: auxiliares de compilación e incorporación de la web local.
- [Mapa técnico](CODE.es-AR.md) y [manual](MANUAL.es-AR.md).

La compilación normal compila e incorpora la web local. Para reutilizar una web generada con `cd frontend && npm ci && npm run build`, configurá `-DDGS_USE_LOCAL_UI=ON`. Recompilá los recursos tras modificar sus fuentes. `-DDGS_BUILD_UI=OFF` omite los recursos incorporados; en ese modo, usá `--path` con una web actual compilada.

Las reglas prohíben los tests. La verificación se limita a compilación y análisis estático; no se han ejecutado inferencia ni benchmarks. Este proyecto no usá Git; el propietario gestiona manualmente el borrado de la carpeta `.git` protegida.

## Uso diario

Iniciá una conversación en la web, seleccioná un modelo permitido y enviá texto o adjuntos de imagen/vídeo compatibles. Los ajustes controlan muestreo, contexto y herramientas. `-c` indica el contexto; `-t`, los hilos de CPU; `-ngl auto`, la distribución automática en GPU. `--api-key` activa la autenticación. Los clientes compatibles usan `http://127.0.0.1:8080/api/v1` y `/api/v1/chat/completions`. El directorio puede contener otros archivos, pero el cargador rechaza los checkpoints no admitidos.

Si falla la carga, comprobá identidad, arquitectura, dimensiones y versión GGUF; no cambies las etiquetas de otro modelo para sortear la validación. Si falta memoria, reducí contexto, lotes o descarga a GPU. Si falta la interfaz, recompilá la web o configurá `--path`.

## MCP y funciones de la web

Activá el servidor MCP de shells con `--local-shell-mcp-server` (o `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1`). La web lo detecta y solicita permiso para ejecutar herramientas. Ofrece los intérpretes Bash, Zsh, sh y PowerShell instalados, muestra el comando, stdout y stderr y registra cada llamada en JSONL.

La web incorpora un monitor de CPU/GPU y memoria, barra de estado, ajustes compartidos guardados en el servidor, selector de 15 idiomas, instrucción opcional para responder en el idioma y variante regional elegidos y avisos de cola que distinguen peticiones de la web y de otros clientes de la API. Se conservan el explorador y las descargas de modelos, las pestañas de conversaciones y la política de modelos densos.

La web se sirve en `/`; la API utiliza `/api` por defecto. Los clientes compatibles con OpenAI deben usar `/api/v1` como ruta base. La documentación Swagger está en `/api/doc/`. Las nuevas rutas de API no tienen alias en la raíz. `--api-prefix` permite elegir una ruta dentro de `/api`; `/api/config.js` se la comunica a la web.

## MCP de shells locales

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

Abrí la web e ingresá esa clave API. La entrada `local shell mcp server` aparece cuando el servidor lo ofrece; podés desactivarla globalmente o por conversación. Las herramientas son `bash.run`, `zsh.run`, `sh.run` y `psh.run`, sólo para intérpretes instalados. El endpoint es `POST /api/mcp/shell`, con MCP Streamable HTTP, respuestas JSON y versión de protocolo `2025-11-25`.

Cada herramienta recibe `command`, `cwd` opcional y `timeout` opcional en segundos (60 por defecto, máximo 3600). Cada llamada usa un proceso independiente, sin terminal interactiva. Los cambios de directorio y variables no pasan a la siguiente llamada. La cancelación o el tiempo límite terminan el grupo de procesos. Los comandos tienen los permisos de la cuenta que ejecuta el servidor; protegé el acceso con una clave API.

El chat muestra stdin (el comando), stdout y stderr por separado, con el código de salida. Los ajustes de visualización permiten desplegar las cajas automáticamente. Los registros de `~/.dgs/local-shell-mcp-server/` guardan el comando y la salida completos, en un JSONL fechado por llamada; el chat puede recortar salidas largas. Los archivos son privados de la cuenta del servidor y no se rotan automáticamente.

En equipos Linux que ya tengan sudo, sus peticiones de contraseña aparecen en un diálogo con el foco en la entrada y botones Aceptar/Cancelar. La contraseña sólo se enviá al auxiliar: no llega al modelo, a la conversación ni a los registros. La espera pausa el tiempo límite del comando; una petición sin respuesta caduca a los cinco minutos. Este proyecto no instala sudo y la ejecución en producción no lo necesita.

## Monitor y preferencias compartidas

Abrí **Monitor** desde la barra lateral o pulsá la barra de estado inferior. El uso de CPU y GPU se actualiza cada dos segundos mientras la página es visible. La vista completa añade RAM, VRAM, temperaturas y consumo cuando el controlador los proporciona. La memoria se desglosa en pesos densos, pesos MTP/borrador, cachés KV/estado recurrente, búferes de cálculo, prompts/checkpoints guardados, otros usos y espacio libre. NVIDIA utiliza NVML dinámicamente y AMD, sysfs; los datos no disponibles quedan como desconocidos. En modo router se agregan los procesos cargados sin cargar ni despertar modelos.

Los ajustes se guardan en `~/.dgs/config/ui-settings.json` de la cuenta del servidor (0600). Los navegadores comparten configuración, tema, idioma, favoritos y preferencias de herramientas. Cada guardado sustituye el documento; si varios navegadores editan a la vez, prevalece el último guardado correcto. La clave API se queda en el navegador, al igual que las conversaciones y el estado transitorio de navegación y flujos. No se migran ajustes antiguos del navegador.

Seleccioná el idioma en los ajustes generales. El cambio guarda la selección y recarga la interfaz; guardá primero otras ediciones pendientes. Se ofrecen `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU` y `zh-CN`. El idioma predeterminado y de respaldo para textos sin traducir es en-US. **Responder en el idioma de la interfaz** añade una instrucción regional a las nuevas peticiones, turnos de herramientas, títulos y preparación de caché KV. No traduce mensajes existentes; su cumplimiento depende del modelo.

Cuando una petición espera un slot, su estado muestra cuántas consultas tiene por delante y si alguna procede directamente de la API. Los clientes de API habituales conservan su formato de respuesta. Sólo las peticiones en streaming marcadas con `"dgs_client": "webui"` reciben eventos `dgs_queue`.

Durante el desarrollo, `npm run dev` reenviá `/api` a `http://localhost:9931`; `VITE_PUBLIC_SERVER_ORIGIN` permite elegir otro servidor. La compilación de producción incorpora la web y el visor Swagger local, sin depender de una CDN de documentación.

## Carpetas de fuentes y compilación

`build/` contiene la entrada de CMake y `build-local.sh`. El script resuelve el proyecto desde su propia ubicación, configurá Release y acepta opciones adicionales de CMake. Las dependencias deben estar instaladas previamente. `DGS_BUILD_JOBS` vale 4 por defecto. Para configurar manualmente se usa `cmake -S build -B _/build`; `backend/CMakeLists.txt` define el proyecto del motor. Los archivos generados y `bin/dgs-server` van en `_/build/`, separados de los scripts.

Después de compilar, la instalación en producción como root se realiza con `cmake --install _/build --prefix /usr/local`. Las reglas de preparación web están en `deploy/scripts/ui-assets.cmake`; `frontend/sources.cmake` enumera sus entradas y `frontend/embed.py` genera los recursos C++ incorporados. El README principal está en la raíz y sus versiones regionales, en `doc/`.


<a id="license"></a>

## Licencia y avisos de terceros

Las contribuciones de NiPeGun se distribuyen bajo [PolyForm Noncommercial 1.0.0](../LICENSE). El código de terceros incorporado conserva sus propias licencias y avisos de copyright, reunidos en [build/licenses/](../build/licenses/README.md).

El LICENSE principal se conserva literalmente. En particular, el aviso MIT de los autores de ggml permanece en `build/licenses/LICENSE-llama.cpp`; Swagger UI conserva su licencia Apache y su NOTICE. El inventario incluye también los componentes HTTP, JSON, hash, multimedia, subprocess, base64, kernels de CPU y bibliotecas matemáticas incorporados. Los avisos incluidos en fuentes de dependencias y paquetes web siguen siendo aplicables. Los pesos de modelos y las bibliotecas del sistema tienen sus propios términos.

`cmake --install _/build --prefix PREFIX` copia el LICENSE principal y `build/licenses/` a `PREFIX/share/doc/dgs-server/licenses/` con la disposición de datos predeterminada. Con `/usr/local`, la ruta es `/usr/local/share/doc/dgs-server/licenses/`; una instalación de usuario puede utilizar `~/.local`. El componente `Licenses` permite instalar esos documentos por separado. Al distribuir el ejecutable manualmente deben acompañarlo estos documentos.
