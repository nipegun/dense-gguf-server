# DGS — 코드 지도

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · **ko-KR** · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server는 **NiPeGun**이 개발하고 유지 관리하는 독립 프로젝트입니다.

- [아키텍처](#architecture)
- [모듈](#modules)
- [주요 심볼](#symbols)
- [호출 흐름](#flows)
- [진입점](#entrypoints)
- [영향 분석](#impact)
- [확장](#extension)
- [MCP, 모니터 및 공유 UI](#integrated-features)
- [출처와 검증](#validation)
- [저장소 구조](#repository-organization)

- [라이선스와 서드파티 고지](#license)

<a id="architecture"></a>

## 아키텍처

서버 하나와 웹 애플리케이션 하나로 구성됩니다. 팩터리는 `dgs_model_qwen35`만 만들며 Qwen3.8-27B도 같은 아키텍처를 사용합니다. DeltaNet 커널과 공통 시각 보조 코드를 쓰는 Qwen 프로젝터를 유지합니다. MoE, 옛 GGUF 버전, 다른 프로파일은 거부됩니다. 허용 목록은 HTTP 별칭과 무관하게 원래 메타데이터를 읽은 뒤 레이어, 임베딩, FFN, 헤드를 검사합니다. 차원만으로 Qwen3.5-27B와 Qwen3.8-27B를 구별할 수 없습니다. GGUF의 식별 정보는 선언값이며 암호학적으로 인증되지 않습니다. GGML과 외부 의존성의 현재 기본 연산과 인터페이스는 유지됩니다.

<a id="modules"></a>

## 모듈

| 모듈 | 경로 | 책임 | 의존 대상 | 사용 주체 |
|---|---|---|---|---|
| 정책 | `backend/src/dgs-model-policy.*` | Qwen 허용 목록 | GGUF, hparams | 모델 로더 |
| Qwen | `backend/src/models/` | 밀집 그래프 | DeltaNet, GGML | 서버 |
| 시각 처리 | `backend/tools/mtmd/` | Qwen 이미지/영상 | GGML, 코덱 | 서버 |
| 채팅 | `backend/common/` | Jinja, XML 도구, 샘플링 | 모델, vendor | 서버 |
| HTTP | `backend/tools/server/` | 경로, 슬롯, 도구 | common, mtmd, UI | 실행 파일 |
| 웹 | `frontend/` | 현재 UI 및 저장 형식 | 버전이 고정된 npm 패키지 | 브라우저 |
| 빌드 | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | 빌드 및 내장 | 컴파일러, npm, Python | 배포 |
| 셸 MCP | backend/tools/server/server-shell-mcp.* | MCP 명령과 암호 요청 | HTTP, POSIX, common_json | 서버, MCP 웹 |
| 모니터 | backend/tools/server/server-monitor.* | CPU/GPU 측정 및 메모리 집계 | procfs, sysfs, 선택적 NVML | server-context, 라우터, 웹 |
| MTP 가중치 | backend/src/dgs-ext.h; dgs-model.cpp | 밀집/MTP 할당 분류 | GGML 버퍼 메타데이터 | 모니터 스냅샷 |
| 공유 설정 | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | 원자적으로 저장되는 설정 문서 | 파일 시스템, API, SettingsService | 모든 웹 상태 저장소 |
| API 문서 | backend/tools/server/server-api-doc.*; frontend/api.html | 현재 OpenAPI와 Swagger | 경로 등록부, 스키마, vendor/swagger-ui | /api/doc/ |
| 언어 | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15개 카탈로그와 응답 지시 | JSON, 공유 설정 | 웹 구성 요소, 채팅 서비스 |
| 대기열 알림 | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | 요청한 클라이언트의 큐 이벤트 | 작업 및 응답 큐 | 스트리밍 채팅 UI |
| 웹 자원 | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | UI 빌드 및 내장 | npm, Python | `dgs-ui` |
| 라이선스 배포 | `backend/CMakeLists.txt`, `build/licenses/` | 프로젝트와 서드파티 조건 분리 | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## 주요 심볼

| 심볼 | 파일:줄 | 기능 |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | GGUF 식별 검증 |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | 밀집 모델 차원 검증 |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | MoE 텐서 거부 |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | 허용 모델 생성 |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Qwen 차원 로드 |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Jinja 렌더링 |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Qwen 시각 그래프 생성 |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | 서버 시작 및 경로 등록 |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | MCP JSON-RPC 전달 |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | 셸 호출 검증 및 실행 |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | sudo 암호 요청 전달 |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | 하드웨어와 모델 메모리 집계 |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | 밀집 메모리 구간 게시 |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | 추가 MTP 가중치 집계 |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | 공유 설정의 원자적 저장 |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | OpenAPI와 Swagger 제공 |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | 대기열 위치 전송 |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | 지역 언어 응답 지시 추가 |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | 사용한 암호 문자열 지우기 |

<a id="flows"></a>

## 호출 흐름

로드: `fDgsModelCreate` → MoE 거부 → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → 텐서/그래프. 채팅: HTTP → 큐/슬롯 → Jinja → 토큰화 → 추론 → 스트리밍 응답. 웹: 이전 없이 `DgsUi` 스키마를 직접 초기화합니다. 빌드: npm → `frontend/dist` → `frontend/embed.py` → 내장 라이브러리. 초안 모델도 같은 정책을 따르며 MTP는 허용 모델의 예측 레이어를 사용합니다.

<a id="entrypoints"></a>

## 진입점

| 경로 | 핸들러 | 파일 |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | 현재 완성 API 경로 | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | 현재 API 경로 | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | 서버/라우터 | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | 내장 웹 | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | 스냅샷 및 라우터 집계 | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | 접두사와 저장된 언어 | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## 영향 분석

정책 변경은 초안과 어휘 전용 로드를 포함한 모든 로드에 영향을 줍니다. GGUF 메타데이터 이름을 바꾸면 모델을 식별하지 못할 수 있으므로 파일명으로 대체하지 마세요. 현재 GGUF 키나 공통 연산에 남은 옛 이름은 과거 모델 지원을 의미하지 않습니다. 삭제된 API, 마이그레이션, 별칭을 복원하지 마세요. 스키마 변경은 프론트엔드와 백엔드를 함께 수정해야 합니다.

<a id="extension"></a>

## 확장

모델을 추가하려면 아키텍처, 메타데이터, 차원을 포함한 허용 목록에 명시적으로 결정해야 합니다. 이름만으로 대상을 넓히거나 MoE를 재도입하지 마세요. 새 엔드포인트를 서버에 등록하고 웹 타입도 갱신합니다. README, MANUAL, CODE를 15개 인터페이스 언어 모두에서 최신으로 유지하세요.

<a id="validation"></a>

## 출처와 검증

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

위의 공식 설정이 프로파일을 정의합니다. 검증은 컴파일과 정적 분석만 사용하며 프로젝트 규칙은 테스트를 금지합니다. 모델 추론과 처리량 측정은 수행하지 않았습니다. 보호된 `.git` 디렉터리는 소유자가 직접 제거하며 에이전트는 Git을 사용하지 않습니다.



<a id="integrated-features"></a>

## MCP, 모니터 및 공유 UI

셸: CLI 옵션 → 인터프리터 발견 → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → 권한 확인 → `tools/call` → 독립 프로세스 그룹 → JSON 결과와 호출별 JSONL → stdin/stdout/stderr 표시. askpass 별도 채널은 호출 ID와 비공개 Unix 소켓을 쓰며 웹 대화상자는 암호를 모델 메시지에 넣지 않습니다. `common_json::mWipeString`은 사용 후 보조 프로그램의 민감한 JSON 문자열을 덮어씁니다.

모니터: 모델 로드 → MTP 할당 분리 → 추론 중 최대 초당 1회의 메인 루프 스냅샷 → mutex 보호 복사 → `/api/monitor*` → 보이는 페이지에서 2초 간격 폴링. 기본 및 별도 초안 가중치는 한 번만 집계합니다. 전문가 버퍼, 캐시, 프리페치 범주는 이 모니터에 포함되지 않습니다. 폴링은 추론 작업을 넣거나 잠든 모델을 깨우지 않습니다. 하드웨어 사용률은 추론 스냅샷과 독립적입니다.

설정: 브라우저 API 키 → `/api/ui/settings` → `UiSettingsStore.mLoad` → 상태 저장소 초기화 → 마지막 변경 후 잠시 기다려 전체 문서 저장. 문서는 `config`, `userOverrides`, `language`, `preferences`를 포함하며 각 값은 담당 저장소의 직렬화 형식을 유지합니다. 공유는 새로 고칠 때 반영되고 실시간 공동 편집은 아닙니다. 마지막 전체 문서 저장이 우선하며 대화, 탭, 재개 커서는 로컬에 남습니다.

언어: `/api/config.js` → JSON 카탈로그 → UI. `fApplyResponseLanguage`은 전송할 메시지 복사본만 바꾸며 일반 채팅, 도구, 제목, 사전 채우기에서 사용됩니다. 추가된 DGS 문자열에는 en-US/en-GB/es-ES/es-AR가 있고, 다른 로캘은 가져온 카탈로그에 항목이 없으면 en-US로 대체됩니다.

대기열: `dgs_client=webui` → 완성 스키마 → 출처별 카운터 → `server_task_result_queue` → `dgs_queue` SSE → 대화별 `QueueStore` → 대기 중 응답 표시. OpenAI Responses와 Anthropic 이벤트 형식에는 이런 추가 이벤트를 보내지 않습니다. 표시가 없는 API 클라이언트에는 영향이 없습니다.

API 기본 경로는 `/api`이며 `/api` 아래에서만 변경할 수 있고 UI는 `/`에 남습니다. 핸들러는 상대 경로를 받고 라우터 자식 프로세스는 항상 `/api`을 사용하며 웹은 게시된 접두사를 해석합니다. API 호출은 service worker 캐시 대신 네트워크를 사용합니다. 경로 변경 시 라우터 전달, 클라이언트 URL 구성, OpenAPI를 함께 수정해야 합니다.

확장: 셸 후보와 프론트엔드 도구 이름을 같이 추가하고, 메모리 구간은 밀집 스냅샷·API 타입·색상 그룹을 함께 갱신하세요. UI 문구는 JSON 카탈로그에 추가합니다. 기존 밀집 허용 목록과 현재 형식을 유지하고 변경 후 양쪽을 컴파일하세요. 규칙에 따라 테스트를 추가하거나 실행하지 마세요.


통합 기능 검증: CPU Release, LTO, 70개 내장 웹 자원 빌드 성공. Svelte/TypeScript 오류와 경고는 모두 0입니다. 웹 번들은 권장 크기를 넘었고 GCC/CMake는 기존 경고와 파일 시스템 시계 차이도 보고했습니다. 테스트, 서버 실행, 추론 또는 GPU 빌드는 수행하지 않았습니다.

<a id="repository-organization"></a>

## 저장소 구조

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

빌드 흐름: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, 프론트엔드, 서버. UI 흐름: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → 내장 라이브러리. 래퍼와 백엔드 모두 CMake 소스 루트가 될 수 있습니다. 파일 이동 시 include, CMake 경로, 의존성 목록, 문서를 함께 수정해야 합니다.


구조 검증: build/와 backend/에서 CMake 설정 성공. build/ 진입점으로 CPU Release, LTO, 내장 UI 빌드 성공. Svelte/TypeScript 오류와 경고 0건이며 Bash 구문과 로컬 문서 링크를 확인했습니다. 테스트와 모델 추론은 실행하지 않았습니다.


<a id="license"></a>

## 라이선스와 서드파티 고지

NiPeGun의 기여분에는 [PolyForm Noncommercial 1.0.0](../LICENSE) 라이선스가 적용됩니다. 포함된 서드파티 코드는 자체 라이선스와 저작권 고지를 유지하며, 관련 문서는 [build/licenses/](../build/licenses/README.md)에 모아 두었습니다.

기본 LICENSE는 원문 그대로 유지합니다. 특히 ggml 저작자의 MIT 고지는 `build/licenses/LICENSE-llama.cpp`에 보존되며 Swagger UI의 Apache 라이선스와 NOTICE도 유지됩니다. 목록에는 포함된 HTTP, JSON, 해시, 미디어, subprocess, base64, CPU 커널, 수학 구성 요소도 있습니다. 의존성 소스와 웹 번들에 있는 고지도 계속 적용됩니다. 모델 가중치와 시스템 라이브러리는 각자의 조건을 따릅니다.

기본 데이터 배치에서 `cmake --install _/build --prefix PREFIX`은 프로젝트 LICENSE와 `build/licenses/`을 `PREFIX/share/doc/dgs-server/licenses/`에 복사합니다. 기본 접두사 `/usr/local`을 쓰면 `/usr/local/share/doc/dgs-server/licenses/`에 설치되며 사용자 설치는 `~/.local`를 사용할 수 있습니다. `Licenses` 구성 요소는 따로 설치할 수도 있습니다. 실행 파일을 수동 배포할 때도 이 문서를 함께 제공하세요.

배포: 프로젝트 LICENSE와 서드파티 고지 → CMake `Licenses` 구성 요소 → `${prefix}/share/doc/dgs-server/licenses/`. 의존성을 바꾸면 관련 고지를 검토해야 합니다. 제품명과 API 타입은 DGS를 사용하고 서드파티 귀속 표시 및 GGUF 식별자는 이름을 바꾸지 않습니다. 사용하지 않는 외부 카탈로그 어댑터와 미지원 모델 프리셋을 제거했으며 모델 탐색은 기존 Hugging Face 서비스를 사용합니다.

프로젝트 정체성과 라이선스 검증: CPU Release, LTO, 내장 UI 빌드 성공. Svelte/TypeScript 오류와 경고는 0입니다. 생성된 CMake 설치 규칙에는 기본 라이선스와 수집한 고지가 포함됩니다. 루트 LICENSE는 참조와 바이트 단위로 동일합니다. 테스트, 추론, 운영 환경 설치는 수행하지 않았습니다.
