# Manual do DGS

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · [ja-JP](MANUAL.ja-JP.md) · [ko-KR](MANUAL.ko-KR.md) · [pt-BR](MANUAL.pt-BR.md) · **pt-PT** · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server é um projeto independente desenvolvido e mantido por **NiPeGun**.

O DGS disponibiliza os seguintes modelos densos e a aplicação web de chat associada:

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

Todos os outros modelos de linguagem, incluindo todas as versões de GLM e Kimi, não são suportados. Modelos MoE continuam excluídos. O servidor verifica a identidade GGUF original e as dimensões das camadas, inclusive ao carregar um modelo de rascunho especulativo. As quantizações dos modelos permitidos continuam suportadas.

<a id="local-build"></a>

## Compilar e executar no Debian

Use Debian 13 ou um sistema compatível com C++17, CMake 3.19+, Python 3.13+ e Node.js 20.19+ ou 22.12+. Instale os pacotes como root:

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

Abra `http://127.0.0.1:8080`. Os pesos são fornecidos separadamente. Para imagens e vídeo, informe o projetor Qwen correspondente com `--mmproj`; o processamento de vídeo também usa FFmpeg. Modelos de áudio e geração de áudio não são suportados.

O modo Release, os kernels nativos de CPU e a otimização durante a vinculação, quando suportada, ficam ativos por predefinição. Compilações CUDA e Vulkan usam `-DGGML_CUDA=ON` ou `-DGGML_VULKAN=ON` com o SDK correspondente. Compile para a CPU de destino ou use `-DGGML_NATIVE=OFF`. Não foi medido nenhum ganho de débito.

## Apenas formatos atuais

Apenas GGUF v3 é aceite. `general.name`, ou `general.basename` junto com `general.size_label`, deve identificar um modelo permitido; o nome do ficheiro não basta. Esta política verifica metadados e dimensões, sem autenticar os pesos. Um ficheiro Qwen3.5-27B não é aceite como Qwen3.8-27B só porque as arquiteturas têm as mesmas dimensões.

O chat exige o modelo de formatação Jinja completo do modelo ou `--chat-template-file`. Foram removidos o mecanismo antigo de modelos de formatação fixos, a opção `--jinja`/`--no-jinja`, as APIs C obsoletas, as grafias alternativas das opções CLI longas e o modo antigo de raciocínio. A API mantém os contratos atuais `/api/v1/*`; os aliases antigos `/completion` e `/embedding` foram removidos.

A interface usa o namespace de armazenamento atual `DgsUi` e mensagens estruturadas. Não lê nem migra namespaces, temas, marcadores ou exportações JSON de conversas antigas. São suportadas as exportações atuais de sessões JSONL e de ficheiros ZIP. Os dados antigos do navegador não são apagados. MCP usa StreamableHTTP ou WebSocket; o transporte SSE antigo e o mecanismo alternativo automático foram removidos. A área de transferência exige a API Clipboard atual num contexto seguro.

## Estrutura e documentação

- `backend/`: servidor, inferência Qwen, kernels tensoriais partilhados e visão Qwen.
- `frontend/`: aplicação Svelte atual e versões fixadas das dependências.
- `deploy/`: auxiliares de compilação e incorporação da interface local.
- [Mapa técnico](CODE.pt-PT.md) e [manual](MANUAL.pt-PT.md).

A compilação normal compila e incorpora a interface local. Para reutilizar uma interface já gerada com `cd frontend && npm ci && npm run build`, configure `-DDGS_USE_LOCAL_UI=ON`. Gere os recursos novamente após alterar o respetivo código-fonte. `-DDGS_BUILD_UI=OFF` omite os recursos incorporados; nesse modo use `--path` com um diretório de interface atual já compilada.

As regras do projeto proíbem testes. A verificação limita-se à compilação e à análise estática de tipos; não foram executadas inferência nem medições de desempenho. O projeto não usa Git; o seu proprietário remove manualmente o diretório protegido `.git`.

## Uso diário

Inicie uma conversa na interface, selecione um modelo permitido já carregado e envie texto ou anexos de imagem/vídeo compatíveis. As definições controlam amostragem, contexto e ferramentas. `-c` define o tamanho do contexto, `-t` as threads de CPU e `-ngl auto` o posicionamento automático na GPU. Use `--api-key` quando precisar de autenticação. Os clientes compatíveis ligam-se a `http://127.0.0.1:8080/api/v1` e `/api/v1/chat/completions`. Os diretórios podem conter outros ficheiros, mas o carregador rejeita checkpoints não suportados.

Se o carregamento falhar, verifique identidade, arquitetura, dimensões e versão GGUF; não altere as etiquetas de outro modelo para contornar a validação. Se faltar memória, reduza contexto, tamanho dos lotes ou transferência para GPU. Se a interface estiver ausente, recompile-a ou informe `--path`.

## MCP e funções web

Ative o servidor MCP de shells integrado com `--local-shell-mcp-server` ou `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1`. A interface deteta-o automaticamente e pede autorização para usar as ferramentas. São oferecidos os interpretadores Bash, Zsh, sh e PowerShell instalados; a interface mostra comando, stdout e stderr e regista cada chamada em JSONL.

A interface oferece monitor de CPU/GPU e memória, barra de estado, definições partilhadas no servidor, seletor de 15 idiomas, instrução opcional para responder na variante regional escolhida e avisos de fila que distinguem consultas web de outros clientes API. São mantidas a navegação e a transferência de modelos, os separadores de conversas e a política de modelos densos.

A interface é servida em `/`; a API fica sob `/api` por predefinição. Clientes compatíveis com OpenAI devem usar `/api/v1` como caminho-base. A documentação Swagger fica em `/api/doc/`. Os novos caminhos não têm aliases na raiz. `--api-prefix` permite escolher um caminho sob `/api`; `/api/config.js` publica-o para a interface.

## MCP de shells locais

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

Abra a interface e informe a chave API. A entrada `local shell mcp server` aparece automaticamente quando habilitada no servidor; pode ser desativada globalmente ou por conversa. As ferramentas são `bash.run`, `zsh.run`, `sh.run` e `psh.run`, apenas para interpretadores instalados. O endpoint é `POST /api/mcp/shell`, com MCP Streamable HTTP, respostas JSON e protocolo `2025-11-25`.

Cada ferramenta aceita `command`, `cwd` opcional e `timeout` opcional em segundos: 60 por predefinição, no máximo 3600. As chamadas usam processos independentes sem terminal interativo. Mudanças de diretório e ambiente não persistem entre chamadas. Cancelamento ou prazo esgotado encerra o grupo de processos. Os comandos usam as permissões da conta do servidor; proteja o acesso com uma chave API.

O chat mostra stdin (o comando), stdout e stderr separados, com o código de saída. As definições de apresentação podem abrir essas caixas automaticamente. Os logs em `~/.dgs/local-shell-mcp-server/` guardam o comando e toda a saída num JSONL com data e hora por chamada; o chat pode truncar saídas longas. Os ficheiros são privados da conta do servidor e não têm rotação automática.

Em sistemas Linux onde sudo já existe, os seus pedidos de palavra-passe aparecem num diálogo com foco na entrada e botões Aceitar/Cancelar. A palavra-passe só é enviada ao auxiliar, nunca ao modelo, à conversa ou aos logs. A espera pelo utilizador pausa o prazo do comando; pedidos sem resposta expiram em cinco minutos. O projeto não instala sudo e não o exige para funcionar em produção.

## Monitor e preferências partilhadas

Abra **Monitor** na barra lateral ou clique na barra de estado inferior. O uso de CPU/GPU é atualizado a cada dois segundos enquanto a página estiver visível. A visão completa acrescenta RAM, VRAM, temperatura e consumo quando fornecidos pelo driver. A memória distingue pesos densos, pesos MTP/rascunho, caches KV/estado recorrente, buffers de cálculo, prompts/checkpoints guardados, outros usos e espaço livre. NVIDIA usa NVML dinamicamente, e AMD usa sysfs; medições indisponíveis ficam desconhecidas. O modo router agrega os processos carregados sem carregar nem despertar modelos.

As definições ficam em `~/.dgs/config/ui-settings.json` da conta do servidor, com modo 0600. Todos os navegadores do mesmo servidor partilham configuração, tema, idioma, favoritos e preferências de ferramentas. A gravação substitui o documento; em edições simultâneas, prevalece a última gravação bem-sucedida. As chaves API ficam no navegador, assim como as conversas e o estado transitório de navegação e fluxos. Definições antigas não são migradas.

Escolha o idioma nas definições gerais. A mudança guarda a escolha e recarrega a interface; guarde antes outras alterações pendentes. As localidades suportadas são `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU` e `zh-CN`. O inglês dos EUA é a predefinição e a alternativa para textos não traduzidos. Ative **Responder no idioma da interface** (Respond in the interface language) para adicionar uma instrução regional a futuras consultas, ciclos de ferramentas, títulos e preenchimento prévio do KV. Mensagens existentes não são traduzidas; o cumprimento depende do modelo.

Enquanto uma consulta aguarda um slot, o seu estado mostra quantas consultas estão à frente e se alguma veio diretamente da API. Clientes API comuns mantêm o formato habitual. Apenas consultas em streaming marcadas com `"dgs_client": "webui"` recebem eventos `dgs_queue`.

No desenvolvimento do frontend, `npm run dev` encaminha `/api` para `http://localhost:9931`; defina `VITE_PUBLIC_SERVER_ORIGIN` para usar outro servidor. A compilação de produção incorpora a interface e o Swagger local, sem precisar de CDN externa de documentação.

## Diretórios de fontes e compilação

`build/` contém a entrada do CMake e `build-local.sh`. O script localiza o projeto a partir da sua própria localização, configura Release e aceita opções adicionais do CMake. As dependências devem estar instaladas. `DGS_BUILD_JOBS` vale 4 por predefinição. Na configuração manual execute `cmake -S build -B _/build`; `backend/CMakeLists.txt` define o projeto do mecanismo. Os ficheiros gerados e `bin/dgs-server` ficam em `_/build/`, separados dos scripts de compilação.

Após compilar, a instalação em produção como root usa `cmake --install _/build --prefix /usr/local`. As regras CMake de preparação da interface ficam em `deploy/scripts/ui-assets.cmake`; `frontend/sources.cmake` lista as respetivas entradas e `frontend/embed.py` gera os recursos C++ incorporados. O README principal fica na raiz e as suas versões regionais em `doc/`.


<a id="license"></a>

## Licença e avisos de terceiros

As contribuições de NiPeGun estão sob a licença [PolyForm Noncommercial 1.0.0](../LICENSE). O código de terceiros incorporado mantém suas próprias licenças e avisos de direitos autorais, reunidos em [build/licenses/](../build/licenses/README.md).

O LICENSE principal permanece literal. Em especial, o aviso MIT dos autores do ggml permanece em `build/licenses/LICENSE-llama.cpp`; Swagger UI mantém a licença Apache e o NOTICE. O inventário também inclui componentes HTTP, JSON, hash, multimédia, subprocess, base64, kernels de CPU e bibliotecas matemáticas incorporados. Os avisos no código-fonte das dependências e nos pacotes web continuam aplicáveis. Pesos de modelos e bibliotecas do sistema têm os seus próprios termos.

`cmake --install _/build --prefix PREFIX` copia o LICENSE do projeto e `build/licenses/` para `PREFIX/share/doc/dgs-server/licenses/` com a organização de dados predefinida. Com o prefixo `/usr/local`, eles ficam em `/usr/local/share/doc/dgs-server/licenses/`; uma instalação de utilizador pode usar `~/.local`. O componente `Licenses` pode ser instalado separadamente. Inclua esses documentos ao distribuir manualmente o executável.
