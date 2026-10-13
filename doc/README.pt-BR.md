# dense-gguf-server

Este README também está disponível em:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), **pt-BR**, [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## O que é

Execute modelos GGUF densos localmente por meio de um chat web e de uma API HTTP de inferência.

Como llama.cpp, mas criado com precisão de bisturi nipeguniana para alcançar a maior velocidade de inferência possível.

## Capturas de tela

Ainda não foram adicionadas capturas de tela.

## Implantação

No **Debian**, depois de compilar o projeto, execute estes comandos como `root` no diretório raiz:

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

Abra `http://127.0.0.1:8080`. A API usa `/api`, os clientes compatíveis com OpenAI usam `/api/v1`, e o Swagger fica em `/api/doc/`. Os pesos dos modelos são fornecidos separadamente; use `--api-key` ao conceder acesso a outros clientes.

## Compilação

Primeiro instale as dependências, como `root` no Debian 13:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

Depois execute no diretório do projeto com seu usuário habitual:

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

O script compila os fontes locais em `_/build/`. `DGS_BUILD_JOBS=8` define o número de tarefas paralelas; argumentos adicionais são enviados ao CMake, por exemplo `-DGGML_CUDA=ON` ou `-DGGML_VULKAN=ON` quando o SDK correspondente está instalado. As [dependências e opções de compilação](MANUAL.pt-BR.md#local-build) devem estar disponíveis antes.

## Leia a documentação

Consulte:

- [doc/MANUAL.pt-BR.md](MANUAL.pt-BR.md) para compilar, instalar e usar DGS, configurar MCP e a interface ou diagnosticar problemas.
- [doc/CODE.pt-BR.md](CODE.pt-BR.md) para entender o mecanismo denso, a organização dos fontes e os contratos de compilação do backend e do frontend.

## Patrocine este projeto

- **[Contrate-me](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)** para adicionar novas funcionalidades.
- **GitHub Sponsors.** Você também pode apoiar financeiramente o projeto no GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
