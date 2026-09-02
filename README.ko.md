# genshin-quest-voice-over

> **언어 / Languages**: [中文](README.md) · [English](README.en.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

《원신》에서 더빙이 없는 퀘스트(주로 세계 임무)의 대화 텍스트를 실시간으로 읽어주는 서비스입니다.

이 도구는 화면 캡처로 게임 내 대화 자막을 인식하고, OCR로 텍스트를 추출한 뒤 TTS로 음성을 합성하여 재생합니다. 게임 클라이언트는 전혀 수정하지 않습니다.

## 기능 흐름

```
게임 실행 → 화면 캡처(2-4 FPS) → OCR 텍스트 인식 → 텍스트 중복 제거 / 변화 감지
    → TTS 스트리밍 합성 → 스트리밍 재생(miniaudio, 합성하며 재생)
    └─ 스트리밍 불가 시 폴백: 일괄 합성 → 재생(winsound / miniaudio)
```

> 스트리밍: TTS 엔진과 플레이어가 모두 스트리밍을 지원할 때(Edge TTS + miniaudio) 합성하며 재생하는 방식을 우선하여 엔드투엔드 체감 지연을 낮춥니다. 엔진이 스트리밍을 지원하지 않거나(오프라인 VITS 스켈레톤 등) `playback`(miniaudio) 의존성 그룹이 설치되지 않은 경우 일괄 합성 + 블로킹 재생으로 자동 폴백합니다. 참고: Edge TTS는 MP3를 출력하며, `winsound`는 기본적으로 WAV만 지원합니다. `playback` 의존성 그룹이 설치되지 않으면 MP3를 디코딩할 수 없으므로, 이 경우 비-WAV 오디오는 건너뛰어도 프로그램은 중단되지 않습니다.

## 환경 준비

Python 의존성 관리는 [uv](https://docs.astral.sh/uv/)를 사용합니다:

```bash
uv sync
```

핵심 실행에는 `numpy`만 필요하며, 각 백엔드 라이브러리는 옵션 의존성 그룹을 통해 필요에 따라 활성화합니다. 옵션 의존성은 `pyproject.toml`에 선언되어 있으며 `uv sync --extra <그룹명>`으로 활성화합니다:

| 모듈 | 옵션 의존성 그룹 | 활성화 명령 |
|------|-----------|---------|
| 화면 캡처(DXCam + MSS) | `capture` | `uv sync --extra capture` |
| OCR(RapidOCR 기본) | `ocr-rapid` | `uv sync --extra ocr-rapid` |
| OCR(PaddleOCR 대체) | `ocr` | `uv sync --extra ocr` |
| OCR GPU(RapidOCR 대체) | `ocr-rapid-gpu` | `uv sync --extra ocr-rapid-gpu` |
| TTS(Edge TTS 온라인) | `tts-online` | `uv sync --extra tts-online` |
| 재생(스트리밍 재생 + 비-WAV 디코딩) | `playback` | `uv sync --extra playback` |
| OCR 전처리(이미지 개선 + 자막 영역 포커스) | `ocr-preprocess` | `uv sync --extra ocr-preprocess` |

단일 그룹 활성화는 `uv sync --extra capture --extra ocr-rapid --extra tts-online --extra playback`, 또는 한 번에 모두 활성화하려면 `uv sync --all-extras`를 사용합니다.
> 참고: `ocr-rapid`(CPU)와 `ocr-rapid-gpu`(GPU)는 상호 배타적이며, uv는 둘을 충돌 그룹으로 선언합니다. `--all-extras`는 둘을 동시에 활성화하므로 오류가 발생합니다. GPU 환경에서는 `--all-extras`를 사용하지 말고 GPU 그룹을 명시적으로 지정하세요(아래 'GPU 가속' 참조).

백엔드 의존성이 활성화되지 않으면 앱이 해당 활성화 안내를 표시하고 대체 백엔드로 자동 폴백을 시도합니다.

### GPU 가속(선택)

OCR 인식은 기본적으로 CPU에서 실행되며, `--gpu` 스위치로 GPU 추론 가속을 활성화할 수 있습니다. GPU 의존성은 CPU 버전과 충돌하므로 백엔드별로 둘 중 하나만 선택해 설치해야 합니다:

- **RapidOCR(권장)**: `ocr-rapid` 그룹 대신 `ocr-rapid-gpu` 그룹(onnxruntime-gpu)을 활성화:

  ```bash
  uv sync --extra ocr-rapid-gpu --extra capture --extra tts-online --extra playback
  ```

  `onnxruntime-gpu 1.28.x`는 CUDA 13.x와 cuDNN 9.x 런타임 환경이 필요합니다. 일치하는 CUDA Toolkit / cuDNN이 설치되어 있고 라이브러리 검색 경로가 올바르게 구성되어 있는지 확인하세요(Windows는 `PATH`, Linux는 `LD_LIBRARY_PATH`). 그렇지 않으면 GPU 추론이 조용히 CPU로 폴백되거나 초기화에 실패합니다.

- **PaddleOCR**: PaddlePaddle 3.x의 GPU 버전(`paddlepaddle-gpu`)은 Paddle 공식 소스에만 배포되어 일반 PyPI 의존성으로는 설치할 수 없습니다. [Paddle 설치 가이드](https://www.paddlepaddle.org.cn/documentation/zh//install/index_cn.html)에서 CUDA 버전에 맞는 공식 소스로 GPU 버전 `paddlepaddle`(CPU 버전 대체)을 설치하고, 나머지 의존성은 `uv sync --extra ocr`으로 설치하세요.

GPU 의존성 설치 후 실행 시 `--gpu`를 추가하면 가속이 활성화됩니다(그렇지 않으면 GPU 의존성은 사용되지 않고 CPU로 동작합니다).
> 참고: `--gpu`는 **사용자가 요청한** GPU 상태를 기록합니다. 실행 시 CUDA/cuDNN 환경이 없으면 RapidOCR이 실제로 CPU로 실행될 수 있으며, 초기화 로그에 `gpu_requested` 필드로 요청 상태가 반영됩니다. GPU 초기화에 실패하면 앱이 오류를 던지고 대체 OCR 백엔드로 폴백을 시도합니다.
> 참고: Edge TTS는 MP3를 출력하므로 재생하려면 `playback` 그룹(miniaudio)을 활성화해야 합니다. 활성화하면 miniaudio 스트리밍 재생(합성하며 재생)을 사용하며, 활성화하지 않으면 비-WAV 오디오(Edge TTS MP3 등)는 건너뜁니다.

> 개인정보 안내: Edge TTS(온라인 TTS)를 사용하면 화면 캡처 후 OCR로 인식된 텍스트가 네트워크를 통해 Microsoft Edge TTS API로 전송되어 음성 합성이 수행됩니다. 개인정보가 민감한 경우 오프라인 TTS(VITS) 백엔드를 사용하세요.

> 자막 영역 포커스: `ocr-preprocess` 그룹(OpenCV)을 설치하면 OCR 전에 그레이스케일 / 대비 개선과 약간의 확대를 수행하고, 인식 결과에서 화면 하단 대화 밴드 텍스트에 포커스하여 오른쪽 옵션 메뉴, 우상단 성능 수치(FPS/GPU), 게임패드 버튼 힌트(예: `X 재생 중`), UID 같은 UI 노이즈를 자동으로 제거합니다. 실제로 플레이어가 보는 대화만 읽어줍니다. `「」` / `《》`로 감싼 NPC 이름 라벨도 필터링됩니다. 그룹이 없으면 전체 화면 텍스트로 자동 폴백하며 기존 동작은 변하지 않습니다.

## PyPI에서 설치(명령줄 버전)

PyPI 배포 패키지에는 명령줄 프로그램과 해당 엔진 백엔드만 포함되며, **데스크톱 GUI**(아래 패키징 섹션 참조)와 웹 서비스는 포함되지 않습니다.

```bash
# 프로그램만 설치
uv tool install genshin-quest-voice-over
# 또는
pipx install genshin-quest-voice-over

# 옵션 백엔드도 함께 설치
uv tool install "genshin-quest-voice-over[capture,ocr-rapid,ocr-preprocess,tts-online,playback]"
```

설치 후에는 `gqvo` 명령을 사용합니다(긴 형태 `genshin-quest-voice-over`도 동일). 인수는 아래의 `python main.py`와 완전히 동일합니다:

```bash
gqvo --help
gqvo --select-region --fps 3
```

| 시나리오 | 설치가 필요한 옵션 그룹 |
| --- | --- |
| 화면 캡처 | `capture`(DXCam은 Windows 전용. Linux / macOS는 자동으로 MSS로 폴백) |
| OCR 인식 | `ocr-rapid`(기본 백엔드. ONNX 모델은 패키지에 포함) |
| 자막 영역 포커스 | `ocr-preprocess` |
| 온라인 음성 합성 | `tts-online`(Edge TTS, 네트워크 필요) |
| 스트리밍 재생 / MP3 디코딩 | `playback`(없으면 winsound로 폴백하고 MP3는 건너뜀) |

> 참고:
>
> - 이 저장소의 웹 서비스는 Vercel에 배포되며(Vercel은 주 의존성만 설치하고 옵션 그룹은 설치하지 않음), 따라서 `fastapi`와 `python-multipart`가 주 의존성에 남아 있습니다. CLI 자체는 이를 사용하지 않지만 설치 시 함께 설치됩니다.
> - `ocr-rapid`(CPU)와 `ocr-rapid-gpu`(GPU)의 상호 배타는 **uv의 `[tool.uv].conflicts`**로 선언됩니다. `uv`는 이를 강제하지만 `pip`는 이 제약을 인식하지 못하므로 둘 중 하나만 직접 설치하세요.

## 실행

```bash
uv run python main.py
```

자주 쓰는 파라미터(전체 파라미터는 `uv run python main.py --help` 참조):

```bash
# 캡처 영역 지정(left,top,right,bottom) 및 프레임 레이트 낮추기
uv run python main.py --region 100,200,900,600 --fps 3

# 캡처 영역 대화형 선택(전체 화면 오버레이, 마우스 드래그로 선택, Esc 시 전체 화면으로 폴백)
# 확장 화면 지원: 오버레이가 모든 모니터를 덮고, 선택 후 모니터를 자동 감지하여 좌표를 변환
# 참고: --select-region과 --region은 상호 배타적이며 동시에 사용할 수 없습니다
uv run python main.py --select-region --fps 3

# 대체 백엔드 사용
uv run python main.py --capture mss --ocr paddle --tts edge

# OCR 언어와 TTS 음성 지정
uv run python main.py --language ch --voice zh-CN-XiaoxiaoNeural

# 오프라인 TTS 사용(VITS, 모델 경로 지정 필요. 현재는 스켈레톤 구현으로 실제 추론은 미연결)
uv run python main.py --tts vits --tts-model-path /path/to/model

# GPU 가속 OCR 사용(해당 GPU 의존성 그룹 설치 필요. 위 'GPU 가속(선택)' 참조)
# OCR 백엔드 명시: rapid(RapidOCR, ocr-rapid-gpu 그룹 필요) 또는 paddle(PaddleOCR, 공식 소스 GPU 버전 필요)
uv run python main.py --ocr rapid --gpu
uv run python main.py --ocr paddle --gpu

# 로컬 저장 설정을 무시하고 내장 기본값으로 시작(종료 시 기본값이 기존 설정을 덮어쓰므로 초기화와 동일)
uv run python main.py --reset-config
```

`Ctrl+C`를 눌러 안전하게 종료하고 리소스를 해제합니다.

### 설정 자동 저장 및 복원

실행이 **종료될 때**마다 실제 적용된 설정을 `~/.genshin-quest-voice-over/config.json`에 기록하고, 다음 실행 시 자동으로 불러와 적용합니다. 캡처 영역, 모니터, 음성, 프레임 레이트 등을 다시 설정할 필요가 없습니다.

- **우선순위**: 내장 기본값 ← 설정 파일의 이전 값 ← 명령줄 명시 인수. 명령줄은 명시적으로 전달한 항목만 덮어쓰고 나머지는 마지막 저장 값을 유지합니다(예: 한 번 영역을 선택했다면 다음에는 `uv run python main.py`만으로 그 영역이 재사용됩니다).
- **저장된 스위치 끄기**: 불리언 스위치는 명령줄에서 켤 수만 있고 끌 수 없으므로, 설정 파일에서 복원된 스위치를 개별적으로 끄는 `--no-verbose` / `--no-gpu` / `--no-full-frame` / `--no-text-direction`을 제공합니다.
- **GUI**: 시작 시 저장된 설정을 폼에 채우고, '시작' 클릭 시와 창을 닫을 때 각각 저장합니다.
- **기본값 복원**: `--reset-config`로 시작하거나 설정 파일을 삭제하세요.
- **예외 상황 폴백**: 파일이 없거나 손상되었거나 구조 버전이 호환되지 않으면 내장 기본값으로 폴백하고 시작을 중단하지 않은 채 로그를 남깁니다. 잘못된 필드는 개별적으로 버려지고 나머지 필드는 정상 적용됩니다(영역 좌표 4개 중 하나라도 잘못되면 영역 전체를 버리고 전체 화면으로 폴백하며, 0으로 채워 잘못된 캡처 범위를 만들지 않습니다).

## Windows 실행 파일로 패키징

GUI(`gui.py`)는 PyInstaller로 **Windows x64 one-dir 프로그램**으로 패키징되어 Python 설치 없이 더블클릭으로 실행할 수 있습니다. 패키징 설정은 저장소 루트의 `gui.spec`에 모여 있으며, 로컬 빌드와 CI가 동일한 설정을 공유해 재현성을 보장합니다.

### 로컬 빌드

```bash
# 패키징에 필요한 모든 옵션 그룹 + build 그룹(pyinstaller) 설치
uv sync --extra gui --extra capture --extra ocr-rapid --extra ocr-preprocess --extra tts-online --extra playback --group build

# 빌드(결과물은 dist/GenshinQuestVoiceOver/)
uv run pyinstaller gui.spec --noconfirm --distpath dist --workpath build/pyinstaller
```

결과물 구성: `dist/GenshinQuestVoiceOver/GenshinQuestVoiceOver.exe` + `_internal/`(의존성 및 모델).
배포 시에는 `GenshinQuestVoiceOver` 디렉터리 전체를 압축하세요. **exe만 따로 복사하지 마세요**.

단일 파일 exe가 아닌 one-dir을 사용하는 이유: 의존성에 onnxruntime / RapidOCR 모델 / OpenCV(약 275MB)가 포함되어 단일 파일 버전은 실행마다 전체를 임시 디렉터리에 압축 해제해야 하므로 시작이 느리고 백신 소프트웨어에 오탐될 가능성이 훨씬 높습니다.

### 패키징 범위

| 의존성 그룹 | 포함 | 설명 |
| --- | --- | --- |
| `gui`(CustomTkinter) | 예 | 테마 리소스 `assets/themes/*.json` 포함 |
| `capture`(DXCam / MSS) | 예 | DXCam은 Windows 전용이며 comtypes로 DXGI/D3D11 호출 |
| `ocr-rapid`(RapidOCR + onnxruntime) | 예 | ONNX 모델은 wheel에 포함되어 있으므로 데이터 파일로 수집해야 함 |
| `ocr-preprocess`(OpenCV headless) | 예 | 없으면 전체 화면 텍스트로 자동 폴백 |
| `tts-online`(Edge TTS) | 예 | 네트워크 연결 필요 |
| `playback`(miniaudio) | 예 | 없으면 winsound 일괄 재생으로 폴백 |
| `ocr`(PaddleOCR / PaddlePaddle) | 아니오 | 수백 MB 증가. 대체 백엔드일 뿐 |
| `ocr-rapid-gpu`(onnxruntime-gpu) | 아니오 | CPU 버전과 상호 배타 |
| 웹 의존성(fastapi / uvicorn 등) | 아니오 | GUI 경로에서 참조가 없어 명시적으로 제외 |

테마 파일 `src/genshin_voice_over/gui/assets/genshin_theme.json`은 `.py`가 아니므로 모듈로 수집되지 않으며, `gui.spec`의 `datas`로 명시 선언합니다. `gui.py`는 프리즈 실행 시 `sys._MEIPASS`를 기준 디렉터리로 이 경로를 해석합니다.

### 릴리스 워크플로

`.github/workflows/release-desktop.yml`은 **Release가 게시될 때(`release: published`) 자동으로 트리거**됩니다:
체크아웃 → `setup-uv`(캐시 키 `uv.lock`) → `uv sync --frozen`(모든 옵션 그룹 + `build` 그룹) → `ruff check`
→ `pyinstaller gui.spec` → 테마 / 모델 / DLL 포함 여부 확인 → exe 시작 스모크 테스트(20초 동안 프로세스가 살아 있으면 통과)
→ zip 압축 → Release 에셋으로 업로드.

- 결과물 이름: `genshin-quest-voice-over-<tag>-win-x64.zip`. 버전은 Release 태그에서 가져오며 동일 이름의 에셋은 덮어씁니다.
- Actions 페이지에서 **Run workflow**로 수동 트리거할 수도 있습니다(아티팩트만 업로드하고 Release는 건드리지 않음).
- 에셋 업로드에는 `contents: write`가 필요하며 워크플로에 선언되어 있습니다. 내장 `GITHUB_TOKEN`을 사용하므로 별도 시크릿 설정이 필요 없습니다.

### 사용 및 문제 해결

압축을 푼 뒤 `GenshinQuestVoiceOver.exe`를 더블클릭하세요. 로그는 GUI 로그 패널에 표시됩니다(콘솔 창 없음).
설정과 디버그 스크린샷은 여전히 `~/.genshin-quest-voice-over/`에 기록되며 exe 위치와 무관합니다.

| 증상 | 확인 사항 |
| --- | --- |
| 실행해도 반응이 없거나 즉시 종료됨 | exe가 `_internal/`와 같은 계층에 있는지 확인. 백신에 격리되지 않았는지 확인(one-dir이 단일 파일보다 오탐이 적지만 첫 실행은 허용이 필요할 수 있음) |
| OCR 초기화 실패 | `_internal/rapidocr/` 아래 `.onnx` 모델 파일이 온전한지 확인 |
| 소리가 나지 않음 / 로그에 `miniaudio is not installed` | `_internal/` 아래 `_miniaudio.pyd`와 `_cffi_backend*.pyd`가 모두 있는지 확인. 후자는 miniaudio의 cffi ABI 확장이 런타임에 동적으로 임포트하며, 없으면 winsound로 조용히 폴백합니다 |
| 캡처 실패 | DXCam은 Windows 10 이상 및 배타적 전체 화면이 아닌 모드가 필요합니다. 실패 시 자동으로 MSS로 폴백합니다 |

## PyPI에 게시

`.github/workflows/publish-pypi.yml`은 **Release가 게시될 때(`release: published`) 자동으로 트리거**됩니다:
`uv build` → `pyproject.toml` 버전과 Release 태그 일치 확인(앞의 `v`는 허용)
→ wheel 점검(`cli.py`와 각 엔진 서브패키지 포함, `gui/`와 `server.py` 미포함) → `twine check`
→ Trusted Publishing으로 업로드.

### 릴리스 절차

1. `pyproject.toml`의 `[project].version`을 수동으로 변경합니다(버전은 수동 관리이며 태그에서 파생하지 않음).
2. 커밋 후 동일한 이름의 태그를 만들고 Release를 게시합니다(예: `v0.2.0`). 둘이 일치하지 않으면 워크플로가 실패합니다.
3. 워크플로가 끝나면 `pip install genshin-quest-voice-over==0.2.0`으로 설치할 수 있습니다.

### 최초 1회 설정(PyPI 측)

게시에는 API 토큰이 필요 없는 **Trusted Publishing(OIDC)**을 사용합니다. PyPI 프로젝트의
*Publishing → Trusted Publishers*에 다음 항목을 추가하세요(모든 필드가 워크플로와 정확히 일치해야 합니다):

| 필드 | 값 |
| --- | --- |
| PyPI Project Name | `genshin-quest-voice-over` |
| Owner | `MorningK` |
| Repository name | `genshin-quest-voice-over` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

### 로컬 게시(대비책)

```bash
uv build
uv publish --token <pypi-token>   # 또는 export UV_PUBLISH_TOKEN=... 선행
```

### 수동 검증

Actions 페이지에서 **Run workflow**로 `Publish to PyPI`를 트리거하면 빌드·점검·아티팩트 업로드만 수행하고 게시하지 않습니다.

## 웹 서비스(FastAPI + SSE)

이 프로젝트는 FastAPI 기반 웹 서비스(`server.py`)도 제공합니다. 프런트엔드가 업로드한 이미지와 선택적 파라미터는 **multipart POST 요청**으로 받고, 이미지에 OCR을 수행한 뒤 인식 텍스트와 스트리밍 TTS 음성을 **SSE(Server-Sent Events) 응답**으로 동일한 스트림에서 반환합니다. 처리 흐름은 데스크톱의 `pipeline.py`와 동일하게 맞췄습니다.

### 인터페이스

| 인터페이스 | 메서드 | 설명 |
|------|------|------|
| `/` | GET | 프런트엔드 페이지(이미지 업로드 + 파라미터 설정 + 받으면서 재생) |
| `/api/voice` | POST | SSE 스트리밍 인터페이스. `image`를 multipart로 업로드하고 `language`/`voice`/`rate`/`ocr_backend`/`tts_backend`를 선택적 폼 필드로 지정 |
| `/api/voices` | GET | 현재 TTS 엔진이 지원하는 음성 목록 반환 |
| `/health` | GET | 헬스 체크 |

`/api/voice` 이벤트 스트림: `event: text`(인식 결과 JSON) → 여러 개의 `event: audio`(base64 인코딩된 MP3 청크) → `event: done`. 오류 시 `event: error`를 내려보냅니다.

### 로컬 실행

OCR/TTS 엔진 런타임 의존성은 이미 `[project].dependencies`에 있습니다. 로컬 개발에는 `uvicorn`(옵션 그룹 `web`)이 추가로 필요합니다:

```bash
uv sync --extra web
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

> `uvicorn`은 로컬 실행 전용이며, Vercel 함수에 포함되지 않도록 기본 의존성에서 옵션 그룹 `web`으로 옮겼습니다(Vercel은 자체 ASGI 런타임으로 `app`을 로드하므로 uvicorn이 필요하지 않습니다).

브라우저에서 `http://localhost:8000`을 열면 사용할 수 있습니다. OCR 후에는 대화 밴드 포커스 텍스트(`roi_text`, `ocr-preprocess` 필요)를 우선 사용하고 비어 있으면 전체 프레임 텍스트로 폴백합니다.

> 서버 측 엔진은 지연 초기화 + 싱글턴 캐시를 사용해 첫 요청에 초기화되고 이후 요청에서 재사용되므로 콜드 스타트 비용을 줄입니다.

### Vercel에 배포

저장소 루트의 `server.py`가 `app = FastAPI()`를 노출하며 Vercel이 이를 진입점으로 자동 인식합니다. 웹/OCR/TTS 런타임 의존성은 `pyproject.toml`의 `[project].dependencies`에 있고, 함께 쓰는 `vercel.json`(함수 설정)도 준비되어 있습니다.

```bash
# Vercel CLI 설치
npm i -g vercel

# 프로젝트 루트에서
vercel           # 로컬 미리보기
vercel deploy    # 프로덕션 배포
```

또는 [Vercel Dashboard](https://vercel.com)에서 이 저장소를 연결해 바로 가져올 수 있습니다.

주의 사항:

- **의존성 설치**: Vercel은 `pyproject.toml`을 우선 읽고 `[project].dependencies`만 설치하며 옵션 의존성 그룹은 설치하지 않습니다. 따라서 웹/OCR/TTS 런타임 의존성(fastapi/python-multipart/numpy/onnxruntime/rapidocr/edge-tts/opencv-python-headless)을 모두 `[project].dependencies`에 두어 Vercel이 기본 설치하고 함수 번들에 포함하도록 했습니다. `uvicorn`은 로컬 개발 전용으로 옵션 그룹 `web`에 남겨 Vercel 번들에 포함되지 않습니다. `vercel.json`에 `installCommand`는 더 이상 필요하지 않습니다. 의존성은 반드시 `[project].dependencies`에 있어야 하며, 그렇지 않으면 Vercel이 빌드에 성공해도 런타임에 임포트에 실패합니다(예: `ModuleNotFoundError: No module named 'rapidocr'`).

- **Large Functions 활성화(필수)**: 이 서비스는 `onnxruntime` / `rapidocr` / `opencv`에 의존하며 번들 크기가 600MB 이상입니다. Large Functions를 활성화하지 않으면 Vercel이 번들에 **"optimizing dependencies"**를 실행해 표준 한도에 맞추려고 `onnxruntime` / `rapidocr` 등 큰 네이티브 의존성을 **번들에서 제거**하므로, 배포는 성공하지만 런타임에 `ModuleNotFoundError: No module named 'rapidocr'`가 발생합니다. 따라서 Large Functions(한도 5GB)를 활성화해 번들이 대용량 함수 경로를 타고 잘리지 않게 해야 합니다. 활성화 방법(모두 Vercel 프로젝트 설정에서 수동 구성해야 하며 `vercel.json`으로는 불가능):
  1. 프로젝트 **Settings → General**에서 **Fluid Compute**가 켜져 있는지 확인(새 프로젝트는 기본 켜짐).
  2. 프로젝트 **Settings → Environment Variables**에 `VERCEL_SUPPORT_LARGE_FUNCTIONS = 1` 추가.
  설정 후 **재배포**가 필요합니다. 빌드 로그에 "optimizing dependencies"가 더 이상 나타나지 않거나(또는 번들이 500MB를 훨씬 넘고 정상 배포되면) 적용된 것입니다.

- **요청 본문 한도(4.5MB)**: Vercel 함수의 요청/응답 본문은 최대 4.5MB이며, 매우 큰 이미지를 업로드하면 `FUNCTION_PAYLOAD_TOO_LARGE`가 발생합니다. 프런트엔드는 `static/index.html`에서 **클라이언트 측 압축**(Canvas로 가장 긴 변을 1600px로 비율 축소 후 JPEG 변환, 약 3.5MB 이하까지 단계적으로 품질 저하)을 수행해 업로드 크기를 한도 아래로 유지합니다. 서버 측 OCR도 이미지를 가장 긴 변 1280px로 축소하므로 인식 품질에는 영향이 없습니다. 프런트엔드를 거치지 않고 API를 직접 호출한다면 이미지 크기를 직접 관리하세요.

- **함수 실행 시간 및 리소스 설정**: `vercel.json`은 `functions.server.py.maxDuration: 60`과 `excludeFiles`만 설정하며 **`memory`는 설정하지 않습니다**. Fluid Compute에서 Hobby의 실행 시간 상한은 300초지만 이 함수는 `maxDuration: 60`으로 명시적으로 60초로 제한됩니다. 더 긴 시간이 필요하면 Vercel 콘솔에서 조정하세요. memory와 CPU도 Vercel 콘솔의 **Functions** 설정에서 구성해야 합니다(Fluid Compute에서는 `vercel.json`으로 설정할 수 없음).

- Vercel serverless는 콜드 스타트가 느리고(첫 요청에서 OCR/TTS 의존성 로드와 음성 목록 네트워크 조회 발생), 무거운 OCR 모델과 온라인 TTS는 네트워크 제한 환경에서 제약을 받을 수 있습니다. 프로덕션에서는 로컬 `uvicorn`이나 상주 프로세스가 있는 플랫폼을 주로 사용하고, Vercel은 가벼운 데모 / 공유용 진입점으로 활용하는 것을 권장합니다.

### OCR 런타임 실패 진단

배포 후 이미지를 업로드했을 때 `Failed to import rapidocr/onnxruntime: ...`가 포함된 `event: error`가 반환되면 아래 표로 근본 원인을 판독하고 대응하세요(오류 문구에 원본 `ImportError` 이유가 포함되며 로그에 전체 traceback이 남습니다):

| 오류의 근본 원인 | 의미 | 대응 |
| --- | --- | --- |
| `No module named 'onnxruntime'` / `No module named 'rapidocr'` | 의존성이 Vercel의 "optimizing dependencies"에 의해 함수 번들에서 제거됨 | Large Functions가 완전히 적용되었는지(`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` + Fluid Compute + Active CPU) 확인 후 재배포 |
| `libgomp.so.1: cannot open shared object file` 등 | Vercel 런타임 이미지에 `onnxruntime`이 필요한 시스템 라이브러리가 없음 | onnxruntime은 추가 시스템 라이브러리를 필요로 하며 Vercel 이미지가 충족하지 못할 수 있습니다. 다른 상주 플랫폼 사용이나 의존성 조정 검토 |
| 기타 `cannot open shared object` / `undefined symbol` | 네이티브 라이브러리 ABI와 런타임 환경 불일치 | `onnxruntime` 버전 변경 또는 다른 배포 플랫폼으로 전환 |

> 팁: 오류 원인은 SSE `error` 이벤트의 `detail` 필드(`cause:` 체인 포함)로도 반환되므로 서버 로그에만 의존하지 않고 브라우저 페이지에서 바로 확인할 수 있습니다.

## 코드 구조

```text
main.py                              # 저장소 내 CLI 실행 셸. genshin_voice_over.cli:main으로 전달
gui.py                               # 데스크톱 GUI 진입점(CustomTkinter)
server.py                            # 웹 서비스 진입점(FastAPI + SSE)
gui.spec                             # PyInstaller 패키징 설정(GUI → Windows exe)
src/genshin_voice_over/              # 임포트 가능한 최상위 패키지(src-layout)
├── cli.py                           # CLI 구현. console script `gqvo`가 가리키는 곳
├── common.py                        # 공용 데이터 타입(Point/Region/SelectedRegion)
├── app/                             # 애플리케이션 오케스트레이션
│   ├── config.py                    # 런타임 설정과 CLI 파싱
│   ├── pipeline.py                  # VoiceOverApp 메인 파이프라인
│   ├── region_selector.py           # 대화형 화면 영역 선택(tkinter, 멀티 모니터 지원)
│   ├── monitor.py                   # 모니터 열거와 멀티 화면 좌표 변환
│   ├── textproc.py                  # 텍스트 정리 / 중복 제거 / 변화 감지
│   └── player.py                    # 오디오 재생(winsound / miniaudio)
├── capture/                         # 화면 캡처(DXCam/MSS)
├── recognition/                     # OCR 인식(PaddleOCR/RapidOCR)
├── tts/                             # TTS 합성(Edge TTS/VITS)
└── gui/                             # 데스크톱 GUI(exe에만 포함, PyPI 패키지에는 미포함)
```
