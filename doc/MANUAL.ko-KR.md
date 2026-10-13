# DGS 매뉴얼

[de-DE](MANUAL.de-DE.md) · [en-GB](MANUAL.en-GB.md) · [en-US](MANUAL.md) · [es-AR](MANUAL.es-AR.md) · [es-ES](MANUAL.es-ES.md) · [fr-FR](MANUAL.fr-FR.md) · [he-IL](MANUAL.he-IL.md) · [hi-IN](MANUAL.hi-IN.md) · [it-IT](MANUAL.it-IT.md) · [ja-JP](MANUAL.ja-JP.md) · **ko-KR** · [pt-BR](MANUAL.pt-BR.md) · [pt-PT](MANUAL.pt-PT.md) · [ru-RU](MANUAL.ru-RU.md) · [zh-CN](MANUAL.zh-CN.md)

Dense GGUF Server는 **NiPeGun**이 개발하고 유지 관리하는 독립 프로젝트입니다.

DGS는 다음 밀집 모델과 관련 웹 채팅 애플리케이션을 제공합니다.

- Qwen3.8-27B
- Qwen3.5-9B
- Qwen3.5-4B
- Qwen3.5-2B

모든 GLM 및 Kimi 버전을 포함한 다른 언어 모델은 지원하지 않습니다. MoE 모델도 계속 제외됩니다. 서버는 원본 GGUF 식별 정보와 레이어 차원을 검증하며, 추측 디코딩용 초안 모델을 로드할 때도 동일하게 확인합니다. 허용된 모델의 양자화 버전은 지원됩니다.

<a id="local-build"></a>

## Debian에서 빌드하고 실행하기

Debian 13 또는 C++17, CMake 3.19+, Python 3.13+, Node.js 20.19+ 또는 22.12+를 갖춘 호환 시스템을 사용하세요. root로 패키지를 설치합니다.

```sh
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
cmake -S build -B _/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _/build --target dgs-server -j 8
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080
```

`http://127.0.0.1:8080`을 여세요. 가중치는 별도로 제공됩니다. 이미지나 영상에는 `--mmproj`로 해당 Qwen 프로젝터를 지정하세요. 영상 처리에는 FFmpeg도 필요합니다. 오디오 모델과 오디오 생성은 지원하지 않습니다.

Release 모드, 네이티브 CPU 커널, 지원되는 링크 시점 최적화는 기본으로 활성화됩니다. CUDA와 Vulkan 빌드는 해당 SDK와 `-DGGML_CUDA=ON` 또는 `-DGGML_VULKAN=ON`을 사용합니다. 배포할 CPU에 맞춰 빌드하거나 `-DGGML_NATIVE=OFF`를 지정하세요. 처리량 향상은 측정하지 않았습니다.

## 현재 형식만 지원

GGUF v3만 허용됩니다. `general.name` 또는 `general.basename`과 `general.size_label`의 조합이 허용된 모델을 식별해야 하며 파일명만으로는 부족합니다. 이는 메타데이터와 차원을 확인하는 정책이지 가중치의 진위를 인증하는 기능이 아닙니다. 아키텍처 차원이 같다는 이유만으로 Qwen3.5-27B 파일을 Qwen3.8-27B로 받아들이지 않습니다.

채팅에는 모델의 완전한 Jinja 템플릿 또는 `--chat-template-file`이 필요합니다. 과거 고정 템플릿 엔진, `--jinja`/`--no-jinja` 스위치, 폐기된 C API, 오래된 긴 CLI 옵션 표기, 기존 추론 모드는 제거되었습니다. 현재 `/api/v1/*` API 계약은 유지되지만 옛 별칭 `/completion`와 `/embedding`는 없습니다.

웹은 현재 저장소 네임스페이스 `DgsUi`과 구조화된 메시지를 사용합니다. 옛 네임스페이스, 테마, 메시지 마커, 이전 JSON 대화 내보내기를 읽거나 마이그레이션하지 않습니다. 현재 JSONL 세션 내보내기와 ZIP 파일은 지원합니다. 브라우저의 옛 데이터는 삭제하지 않습니다. MCP는 StreamableHTTP 또는 WebSocket을 사용하며, 기존 SSE 전송과 자동 폴백은 제거되었습니다. 클립보드는 보안 컨텍스트의 현재 Clipboard API가 필요합니다.

## 구조와 문서

- `backend/`: 서버, Qwen 추론, 공유 텐서 커널, Qwen 시각 처리.
- `frontend/`: 현재 Svelte 애플리케이션과 버전이 고정된 의존성.
- `deploy/`: 빌드 보조 도구와 로컬 웹 임베딩.
- [기술 지도](CODE.ko-KR.md)와 [매뉴얼](MANUAL.ko-KR.md).

일반 빌드는 로컬 웹을 컴파일하고 포함합니다. `cd frontend && npm ci && npm run build`으로 이미 만든 웹을 재사용하려면 `-DDGS_USE_LOCAL_UI=ON`로 설정하세요. 소스를 바꾸면 자원을 다시 생성해야 합니다. `-DDGS_BUILD_UI=OFF`는 내장 자원을 제외합니다. 이 모드에서는 `--path`로 현재 버전의 빌드된 웹 디렉터리를 지정하세요.

프로젝트 규칙은 테스트를 금지합니다. 검증은 컴파일과 정적 타입 검사로 제한되며 모델 추론이나 벤치마크는 실행하지 않았습니다. 이 프로젝트는 Git을 사용하지 않습니다. 보호된 `.git` 디렉터리는 소유자가 직접 삭제합니다.

## 일상적인 사용

웹에서 대화를 만들고, 로드된 허용 모델을 선택한 뒤 텍스트나 지원되는 이미지·영상 첨부를 보내세요. 설정은 샘플링, 컨텍스트, 도구를 제어합니다. `-c`은 컨텍스트 길이, `-t`은 CPU 스레드, `-ngl auto`는 자동 GPU 배치를 지정합니다. 인증이 필요하면 `--api-key`을 사용하세요. 호환 클라이언트는 `http://127.0.0.1:8080/api/v1`와 `/api/v1/chat/completions`로 연결합니다. 모델 디렉터리에 다른 파일이 있어도 지원하지 않는 체크포인트는 로더가 거부합니다.

로드 실패 시 모델 식별 정보, 아키텍처, 차원, GGUF 버전을 확인하세요. 검증을 우회하려고 다른 모델의 표기를 바꾸면 안 됩니다. 메모리가 부족하면 컨텍스트, 배치 크기 또는 GPU 오프로딩 양을 줄이세요. 웹이 없으면 재빌드하거나 `--path`을 지정하세요.

## MCP와 웹 기능

`--local-shell-mcp-server` 또는 `DGS_ARG_LOCAL_SHELL_MCP_SERVER=1`로 내장 셸 MCP 서버를 활성화하세요. 웹이 자동으로 감지하고 도구 호출에 대한 권한을 요청합니다. 설치된 Bash, Zsh, sh, PowerShell을 제공하며, 명령과 stdout/stderr를 표시하고 호출마다 JSONL로 기록합니다.

웹에는 CPU/GPU 및 메모리 모니터, 상태 표시줄, 서버에 저장되는 공유 설정, 15개 언어 선택, 선택한 지역 언어로 답변하라는 선택적 지시, 웹 요청과 다른 API 클라이언트를 구별하는 대기열 알림이 있습니다. 모델 탐색과 다운로드, 대화 탭, 밀집 모델 정책도 유지됩니다.

웹은 `/`, API는 기본적으로 `/api` 아래에서 제공됩니다. OpenAI 호환 클라이언트의 기본 경로는 `/api/v1`여야 합니다. Swagger 문서는 `/api/doc/`에 있습니다. 새 API 경로에는 루트 별칭이 없습니다. `--api-prefix`로 `/api` 하위 경로를 고를 수 있고, `/api/config.js`이 웹에 이를 알립니다.

## 로컬 셸 MCP

```sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf --local-shell-mcp-server --api-key YOUR_KEY
```

웹을 열고 API 키를 입력하세요. 서버에서 활성화하면 `local shell mcp server` 항목이 자동으로 나타나며 전체 또는 대화별로 끌 수 있습니다. `bash.run`, `zsh.run`, `sh.run`, `psh.run` 도구는 설치된 인터프리터에 대해서만 제공됩니다. 엔드포인트는 `POST /api/mcp/shell`, 전송은 MCP Streamable HTTP, 응답은 JSON, 프로토콜 버전은 `2025-11-25`입니다.

각 도구는 `command`과 선택적인 `cwd`, `timeout`를 받습니다. 시간 제한은 초 단위로 기본 60, 최대 3600입니다. 호출은 대화형 터미널 없이 독립 프로세스에서 실행됩니다. 작업 디렉터리와 환경 변경은 다음 호출에 이어지지 않습니다. 취소나 시간 초과는 프로세스 그룹을 종료합니다. 명령은 서버 계정 권한으로 실행되므로 API 키로 접근을 보호하세요.

채팅은 stdin(명령), stdout, stderr 및 종료 코드를 따로 표시합니다. 표시 설정에서 자동으로 펼칠 수 있습니다. `~/.dgs/local-shell-mcp-server/` 로그에는 전체 명령과 출력이 호출별 타임스탬프 JSONL로 저장됩니다. 채팅의 긴 출력은 잘릴 수 있습니다. 파일은 서버 계정만 접근할 수 있으며 자동 순환 삭제되지 않습니다.

sudo가 이미 설치된 Linux에서는 암호 요청이 입력란에 초점이 맞춰진 승인/취소 대화상자로 나타납니다. 암호는 보조 프로세스에만 전달되고 모델, 대화, 로그에는 전달되지 않습니다. 사용자 응답을 기다리는 동안 명령의 시간 제한은 멈추며, 응답이 없으면 5분 뒤 만료됩니다. 이 프로젝트는 sudo를 설치하지 않으며 운영 환경에서도 필수로 요구하지 않습니다.

## 모니터링과 공유 설정

사이드바의 **모니터링** 또는 하단 상태 표시줄을 여세요. 페이지가 보일 때 CPU/GPU 사용률을 2초마다 갱신합니다. 전체 화면에는 드라이버가 제공하는 RAM, VRAM, 온도, 전력도 표시됩니다. 메모리는 밀집 가중치, MTP/초안 가중치, KV/순환 상태 캐시, 계산 버퍼, 저장된 프롬프트/체크포인트, 기타 사용량, 여유 공간으로 구분됩니다. NVIDIA는 NVML을 동적으로, AMD는 sysfs를 사용하며 얻지 못한 수치는 알 수 없음으로 남습니다. 라우터 모드는 로드된 자식 프로세스를 합산하며 모델을 새로 로드하거나 깨우지 않습니다.

설정은 서버 계정의 `~/.dgs/config/ui-settings.json`에 0600 권한으로 저장됩니다. 같은 서버를 쓰는 모든 브라우저가 구성, 테마, 언어, 즐겨찾기, 도구 기본 설정을 공유합니다. 저장 시 문서 전체를 교체하므로 동시 편집에서는 마지막으로 성공한 저장이 우선합니다. API 키, 대화, 임시 탐색 및 스트림 상태는 브라우저에 남습니다. 기존 브라우저 설정은 이전하지 않습니다.

일반 설정에서 언어를 고르세요. 변경하면 선택을 저장하고 화면을 다시 불러오므로 다른 수정부터 저장하세요. 지원 로캘은 `de-DE`, `en-GB`, `en-US`, `es-AR`, `es-ES`, `fr-FR`, `he-IL`, `hi-IN`, `it-IT`, `ja-JP`, `ko-KR`, `pt-BR`, `pt-PT`, `ru-RU`, `zh-CN`입니다. 미국 영어가 기본이며 번역이 없는 문구에도 사용됩니다. **인터페이스 언어로 답변** (Respond in the interface language)을 활성화하면 이후 채팅, 도구 처리 단계, 제목 생성, KV 사전 채우기에 지역 언어 지시를 추가합니다. 기존 메시지는 번역하지 않으며 지시 준수 여부는 모델에 달려 있습니다.

추론 요청이 슬롯을 기다릴 때 앞선 요청 수와 직접 API에서 온 요청의 존재 여부를 표시합니다. 일반 API 클라이언트는 기존 응답 형식을 유지합니다. `"dgs_client": "webui"`으로 표시된 스트리밍 요청만 `dgs_queue` 이벤트를 받습니다.

프론트엔드 개발 시 `npm run dev`은 `/api`을 `http://localhost:9931`로 프록시합니다. 다른 서버를 사용하려면 `VITE_PUBLIC_SERVER_ORIGIN`을 설정하세요. 운영용 빌드는 웹과 로컬 Swagger 뷰어를 포함하므로 외부 문서 CDN이 필요 없습니다.

## 소스 및 빌드 디렉터리

`build/`에는 CMake 진입점과 `build-local.sh`이 있습니다. 스크립트는 자신의 위치를 기준으로 프로젝트를 찾고 Release를 설정하며 추가 CMake 옵션을 받습니다. 의존성은 미리 설치되어야 합니다. `DGS_BUILD_JOBS`의 기본값은 4입니다. 수동 설정은 `cmake -S build -B _/build`을 사용하고, `backend/CMakeLists.txt`가 엔진 프로젝트를 정의합니다. 생성 파일과 `bin/dgs-server`는 `_/build/`에 저장되어 빌드 스크립트와 분리됩니다.

빌드 후 root로 운영 설치할 때는 `cmake --install _/build --prefix /usr/local`을 사용합니다. 웹 준비 규칙은 `deploy/scripts/ui-assets.cmake`, 입력 목록은 `frontend/sources.cmake`에 있으며, `frontend/embed.py`이 내장 C++ 자원을 만듭니다. 기본 README는 루트에, 지역별 버전은 `doc/`에 있습니다.


<a id="license"></a>

## 라이선스와 서드파티 고지

NiPeGun의 기여분에는 [PolyForm Noncommercial 1.0.0](../LICENSE) 라이선스가 적용됩니다. 포함된 서드파티 코드는 자체 라이선스와 저작권 고지를 유지하며, 관련 문서는 [build/licenses/](../build/licenses/README.md)에 모아 두었습니다.

기본 LICENSE는 원문 그대로 유지합니다. 특히 ggml 저작자의 MIT 고지는 `build/licenses/LICENSE-llama.cpp`에 보존되며 Swagger UI의 Apache 라이선스와 NOTICE도 유지됩니다. 목록에는 포함된 HTTP, JSON, 해시, 미디어, subprocess, base64, CPU 커널, 수학 구성 요소도 있습니다. 의존성 소스와 웹 번들에 있는 고지도 계속 적용됩니다. 모델 가중치와 시스템 라이브러리는 각자의 조건을 따릅니다.

기본 데이터 배치에서 `cmake --install _/build --prefix PREFIX`은 프로젝트 LICENSE와 `build/licenses/`을 `PREFIX/share/doc/dgs-server/licenses/`에 복사합니다. 기본 접두사 `/usr/local`을 쓰면 `/usr/local/share/doc/dgs-server/licenses/`에 설치되며 사용자 설치는 `~/.local`를 사용할 수 있습니다. `Licenses` 구성 요소는 따로 설치할 수도 있습니다. 실행 파일을 수동 배포할 때도 이 문서를 함께 제공하세요.
