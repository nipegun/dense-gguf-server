# dense-gguf-server

Este README también está disponible en:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), **es-AR**, [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## Qué es

Ejecutá modelos GGUF densos localmente mediante un chat web y una API HTTP de inferencia.

Como llama.cpp, pero creado con precisión de bisturí nipeguniana para alcanzar la máxima velocidad de inferencia posible.

## Capturas de pantalla

Todavía no se han añadido capturas de pantalla.

## Despliegue

En **Debian**, después de compilar el proyecto, ejecutá estos comandos como `root` desde la carpeta raíz:

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

Abrí `http://127.0.0.1:8080`. La API usa `/api`, los clientes compatibles con OpenAI usan `/api/v1` y Swagger está en `/api/doc/`. Los pesos se proporcionan por separado; usá `--api-key` al dar acceso a otros clientes.

## Compilación

Instalá primero las dependencias, como `root` en Debian 13:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

Después, ejecutá desde la carpeta del proyecto con tu usuario habitual:

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

El script compila las fuentes locales en `_/build/`. `DGS_BUILD_JOBS=8` indica el número de tareas simultáneas. Los argumentos adicionales se pasan a CMake, por ejemplo `-DGGML_CUDA=ON` o `-DGGML_VULKAN=ON` si está instalado el SDK correspondiente. Las [dependencias y opciones de compilación](MANUAL.es-AR.md#local-build) deben estar disponibles previamente.

## Leé la documentación

Explorá:

- [doc/MANUAL.es-AR.md](MANUAL.es-AR.md) para compilar, instalar y usar DGS, configurar MCP y la web o diagnosticar problemas.
- [doc/CODE.es-AR.md](CODE.es-AR.md) para entender el motor denso, la organización del código y los contratos de compilación del backend y del frontend.

## Patrociná este proyecto

- **[Contratame](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)** para añadir nuevas funcionalidades.
- **GitHub Sponsors.** También podés apoyar el proyecto económicamente en GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
