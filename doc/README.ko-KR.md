# dense-gguf-server

이 README는 다음 언어로도 제공됩니다:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), **ko-KR**, [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## 프로젝트 소개

웹 채팅과 HTTP 추론 API를 통해 밀집 GGUF 모델을 로컬에서 실행합니다.

llama.cpp처럼, 하지만 가능한 최고의 추론 속도를 달성하기 위해 NiPeGun 특유의 메스 같은 정밀함으로 다듬었습니다.

## 스크린샷

아직 스크린샷이 추가되지 않았습니다.

## 배포

**Debian**에서 빌드한 뒤 프로젝트 루트 디렉터리에서 `root` 권한으로 다음 명령을 실행하세요.

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

`http://127.0.0.1:8080`을 여세요. API는 `/api`, OpenAI 호환 클라이언트는 `/api/v1`를 사용하고, Swagger는 `/api/doc/`에 있습니다. 모델 가중치는 별도로 준비해야 합니다. 다른 클라이언트에 접근을 허용할 때는 `--api-key`를 사용하세요.

## 빌드

먼저 Debian 13에서 `root` 권한으로 의존성을 설치하세요.

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

그다음 일반 사용자 계정으로 프로젝트 디렉터리에서 실행하세요.

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

스크립트는 로컬 소스를 `_/build/`에 빌드합니다. `DGS_BUILD_JOBS=8`은 병렬 작업 수를 지정합니다. 추가 인수는 CMake에 전달되며, 해당 SDK가 설치되어 있으면 `-DGGML_CUDA=ON` 또는 `-DGGML_VULKAN=ON` 등을 사용할 수 있습니다. [빌드 의존성과 옵션](MANUAL.ko-KR.md#local-build)은 미리 준비되어 있어야 합니다.

## 문서 읽기

다음 문서를 참고하세요.

- [doc/MANUAL.ko-KR.md](MANUAL.ko-KR.md): DGS 빌드, 설치, 사용, MCP와 웹 설정 및 문제 진단.
- [doc/CODE.ko-KR.md](CODE.ko-KR.md): 밀집 엔진, 소스 구조, 백엔드와 프론트엔드의 빌드 계약.

## 프로젝트 후원

- 새 기능 개발은 **[저에게 의뢰하세요](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)**.
- **GitHub Sponsors.** GitHub에서 금전적으로 후원할 수도 있습니다: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).
