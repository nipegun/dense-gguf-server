# DGS — mapa do código

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · **pt-PT** · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server é um projeto independente desenvolvido e mantido por **NiPeGun**.

- [Arquitetura](#architecture)
- [Módulos](#modules)
- [Símbolos](#symbols)
- [Fluxos](#flows)
- [Pontos de entrada](#entrypoints)
- [Impacto](#impact)
- [Extensão](#extension)
- [MCP, monitor e interface partilhada](#integrated-features)
- [Fontes e verificação](#validation)
- [Organização do repositório](#repository-organization)

- [Licença e avisos de terceiros](#license)

<a id="architecture"></a>

## Arquitetura

Um servidor e uma aplicação web. A fábrica instancia apenas `dgs_model_qwen35`; Qwen3.8-27B partilha essa arquitetura. São mantidos os kernels DeltaNet e o projetor Qwen com o seu auxiliar visual partilhado. MoE, versões GGUF antigas e outros perfis são rejeitados. A lista permitida lê metadados originais independentemente dos aliases HTTP e depois verifica camadas, embeddings, FFN e cabeças de atenção. Apenas as dimensões não distinguem Qwen3.5-27B de Qwen3.8-27B. A identidade é declarada pelo GGUF, não autenticada criptograficamente. GGML e dependências externas mantêm as primitivas e interfaces atuais.

<a id="modules"></a>

## Módulos

| Módulo | Caminho | Responsabilidade | Depende de | Usado por |
|---|---|---|---|---|
| Política | `backend/src/dgs-model-policy.*` | Lista de Qwen permitidos | GGUF, hparams | carregador de modelos |
| Qwen | `backend/src/models/` | Grafo denso | DeltaNet, GGML | servidor |
| Visão | `backend/tools/mtmd/` | Imagem/vídeo Qwen | GGML, codecs | servidor |
| Chat | `backend/common/` | Jinja, ferramentas XML, amostragem | modelo, vendor | servidor |
| HTTP | `backend/tools/server/` | Rotas, slots, ferramentas | common, mtmd, UI | executável |
| Web | `frontend/` | interface e armazenamento atuais | versões npm fixadas | navegador |
| Compilação | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | Compilar e incorporar | compilador, npm, Python | instalação |
| MCP de shells | backend/tools/server/server-shell-mcp.* | Comandos MCP e pedidos de palavra-passe | HTTP, POSIX, common_json | servidor, interface MCP |
| Monitor | backend/tools/server/server-monitor.* | Telemetria CPU/GPU e agregação de memória | procfs, sysfs, NVML opcional | server-context, router, web |
| Pesos MTP | backend/src/dgs-ext.h; dgs-model.cpp | Divisão da alocação densa/MTP | metadados dos buffers GGML | instantâneos do monitor |
| Definições partilhadas | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | Documento de definições atómico | sistema de ficheiros, API, SettingsService | todos os stores da interface |
| Documentação API | backend/tools/server/server-api-doc.*; frontend/api.html | OpenAPI e Swagger atualizados | registo de rotas, esquema, vendor/swagger-ui | /api/doc/ |
| Idiomas | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 catálogos e instrução de resposta | JSON, definições partilhadas | componentes web, serviço de chat |
| Avisos de fila | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | Eventos de fila sob solicitação | filas de tarefas e respostas | interface de chat em streaming |
| Recursos web | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | Compilar e incorporar a interface | npm, Python | `dgs-ui` |
| Distribuição das licenças | `backend/CMakeLists.txt`, `build/licenses/` | Separar os termos do projeto e dos terceiros | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## Símbolos

| Símbolo | Ficheiro:linha | Função |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | Validar a identidade GGUF |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | Validar as dimensões densas |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | Rejeitar tensores MoE |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | Criar um modelo permitido |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Carregar dimensões Qwen |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Renderizar Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Criar o grafo visual Qwen |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | Iniciar o servidor e registar rotas |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | Encaminhar JSON-RPC MCP |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | Validar e executar um comando shell |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | Encaminhar pedidos de palavra-passe sudo |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | Agregar hardware e memória dos modelos |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | Publicar segmentos de memória densa |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | Contar pesos MTP anexos |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | Guardar definições partilhadas atomicamente |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | Servir OpenAPI e Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | Emitir a posição na fila |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | Adicionar uma instrução regional de idioma |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | Apagar cadeias de caracteres de palavra-passe já utilizadas |

<a id="flows"></a>

## Fluxos

Carregamento: `fDgsModelCreate` → rejeição de MoE → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → tensores/grafo. Chat: HTTP → fila/slot → Jinja → tokenização → inferência → resposta em streaming. Web: inicialização direta do esquema `DgsUi`, sem migrações. Compilação: npm → `frontend/dist` → `frontend/embed.py` → biblioteca incorporada. Modelos de rascunho seguem a mesma política; MTP usa as camadas preditivas do modelo permitido.

<a id="entrypoints"></a>

## Pontos de entrada

| Rota | Handler | Ficheiro |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | rotas atuais de completação | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | rotas API atuais | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | servidor/router | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | interface incorporada | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | instantâneos e agregação do router | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | Prefixo e idioma guardado | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## Impacto

Mudar a política afeta todos os carregamentos, incluindo rascunhos e apenas vocabulários. Renomear metadados GGUF pode impedir a identificação; não os substitua por nomes de ficheiro. Nomes históricos que continuam como chaves GGUF ou primitivas partilhadas não habilitam modelos antigos. Não restaure APIs, migrações ou aliases removidos. Alterações de esquema exigem coordenação entre frontend e backend.

<a id="extension"></a>

## Extensão

Adicionar um modelo exige uma decisão explícita sobre a lista permitida, incluindo arquitetura, metadados e dimensões. Não amplie o reconhecimento apenas pelo nome nem reintroduza MoE. Registe novos endpoints no servidor e atualize os tipos web. Mantenha README, MANUAL e CODE atualizados nos 15 idiomas da interface.

<a id="validation"></a>

## Fontes e verificação

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

As definições oficiais acima definem os perfis. A verificação usa apenas compilação e análise estática: as regras proíbem testes. Não foram executadas inferências ou medições de débito. O proprietário apaga manualmente o diretório protegido `.git`; o agente não usa Git.



<a id="integrated-features"></a>

## MCP, monitor e interface partilhada

Shell: opção CLI → descoberta dos shells → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → autorização → `tools/call` → grupo de processos independente → resultado JSON e JSONL por chamada → apresentação de stdin/stdout/stderr. O canal askpass usa um identificador de chamada e um socket Unix privado; o diálogo nunca coloca palavras-passe nas mensagens do modelo. `common_json::mWipeString` sobrescreve as cadeias de caracteres JSON sensíveis do auxiliar após o uso.

Monitor: carregamento → divisão dos pesos MTP → instantâneo no ciclo principal, no máximo uma vez por segundo durante a inferência → cópia protegida por mutex → `/api/monitor*` → consulta a cada dois segundos com página visível. Pesos do modelo principal e do rascunho separado são contados uma única vez. Não são incorporados buffers, caches ou prefetch de especialistas. As consultas não criam tarefas de inferência nem despertam modelos adormecidos. O uso do hardware independe do instantâneo de inferência.

Definições: chave API do navegador → `/api/ui/settings` → `UiSettingsStore.mLoad` → inicialização dos stores → gravação do documento completo após uma breve espera desde a última alteração. O documento contém `config`, `userOverrides`, `language` e `preferences`; cada preferência mantém a serialização do próprio store. Os valores são partilhados ao recarregar, sem edição colaborativa ao vivo; prevalece o último documento guardado. Conversas, separadores e cursores de retomada de streams ficam locais.

Idioma: `/api/config.js` → catálogo JSON → interface. `fApplyResponseLanguage` altera apenas cópias das mensagens de saída; chat, ferramentas, títulos e preenchimento prévio o usam. Os textos DGS adicionados incluem en-US/en-GB/es-ES/es-AR; outras localidades recorrem ao en-US quando o catálogo importado não tem a entrada.

Fila: `dgs_client=webui` → esquema de completação → contadores por origem → `server_task_result_queue` → SSE `dgs_queue` → `QueueStore` por conversa → etiqueta de resposta pendente. Os formatos de eventos OpenAI Responses e Anthropic não recebem esses avisos extras. Clientes API sem a marca não são afetados.

Base API: `/api` por predefinição, configurável apenas sob `/api`; a interface permanece em `/`. Os handlers recebem caminhos relativos, os processos filhos sempre usam `/api`, e a web resolve o prefixo publicado. As chamadas API usam a rede, não o cache do service worker. Mudanças de caminho devem atualizar juntas as rotas do router, URLs do cliente e OpenAPI.

Extensão: adicione juntos um shell candidato e o nome de sua ferramenta no frontend; adicione segmentos de memória ao instantâneo denso, tipo API e grupos de cores ao mesmo tempo; adicione textos aos catálogos JSON. Preserve a lista de modelos densos e os formatos atuais. Recompile backend e frontend após mudanças; não crie nem execute testes, conforme as regras.


Verificação das funções integradas: compilação CPU Release com LTO e 70 recursos web incorporados concluída. Svelte/TypeScript: zero erros e zero avisos. O pacote web excede o limite informativo de tamanho; GCC/CMake também indicaram avisos existentes e diferença de relógio no sistema de ficheiros. Não foram executados testes, servidor, inferência ou compilações GPU.

<a id="repository-organization"></a>

## Organização do repositório

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

Fluxo de compilação: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, frontend, servidor. Interface: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → biblioteca incorporada. Tanto o wrapper quanto o backend podem ser raízes CMake. Uma mudança de local deve atualizar includes, caminhos CMake, dependências e documentação em conjunto.


Verificação da estrutura: configuração CMake concluída de build/ e backend/. Compilação CPU Release com LTO e interface incorporada concluída por build/. Svelte/TypeScript sem erros nem avisos; sintaxe Bash e ligações locais verificados. Não foram executados testes nem inferência.


<a id="license"></a>

## Licença e avisos de terceiros

As contribuições de NiPeGun estão sob a licença [PolyForm Noncommercial 1.0.0](../LICENSE). O código de terceiros incorporado mantém suas próprias licenças e avisos de direitos autorais, reunidos em [build/licenses/](../build/licenses/README.md).

O LICENSE principal permanece literal. Em especial, o aviso MIT dos autores do ggml permanece em `build/licenses/LICENSE-llama.cpp`; Swagger UI mantém a licença Apache e o NOTICE. O inventário também inclui componentes HTTP, JSON, hash, multimédia, subprocess, base64, kernels de CPU e bibliotecas matemáticas incorporados. Os avisos no código-fonte das dependências e nos pacotes web continuam aplicáveis. Pesos de modelos e bibliotecas do sistema têm os seus próprios termos.

`cmake --install _/build --prefix PREFIX` copia o LICENSE do projeto e `build/licenses/` para `PREFIX/share/doc/dgs-server/licenses/` com a organização de dados predefinida. Com o prefixo `/usr/local`, eles ficam em `/usr/local/share/doc/dgs-server/licenses/`; uma instalação de utilizador pode usar `~/.local`. O componente `Licenses` pode ser instalado separadamente. Inclua esses documentos ao distribuir manualmente o executável.

Distribuição: LICENSE do projeto e avisos de terceiros → componente CMake `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`. Uma mudança de dependência exige rever o respetivo aviso. O nome do produto e os tipos API usam DGS; atribuições de terceiros e identificadores GGUF não são renomeados. Foram removidos o adaptador de catálogo externo sem uso e os presets não suportados; a descoberta de modelos continua no serviço Hugging Face existente.

Verificação de identidade e licenças: compilação CPU Release com LTO e interface incorporada concluída; Svelte/TypeScript sem erros nem avisos. A instalação CMake gerada inclui a licença principal e os avisos recolhidos. O LICENSE principal é idêntico byte por byte à referência. Não foram executados testes, inferência nem instalação em produção.
