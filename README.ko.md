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
```

`Ctrl+C`를 눌러 안전하게 종료하고 리소스를 해제합니다.

## 코드 구조

```
main.py                      # 애플리케이션 진입점: CLI 인수 파싱 + VoiceOverApp 드라이버
src/
├── common.py                # 공용 데이터 타입(Point/Region/SelectedRegion)
├── app/                     # 애플리케이션 오케스트레이션
│   ├── config.py            # 런타임 설정과 CLI 파싱
│   ├── pipeline.py          # VoiceOverApp 메인 파이프라인
│   ├── region_selector.py   # 대화형 화면 영역 선택(tkinter, 멀티 모니터 지원)
│   ├── monitor.py           # 모니터 열거와 멀티 화면 좌표 변환
│   ├── textproc.py          # 텍스트 정리 / 중복 제거 / 변화 감지
│   └── player.py            # 오디오 재생(winsound)
├── capture/                 # 화면 캡처(DXCam/MSS)
├── recognition/             # OCR 인식(PaddleOCR/RapidOCR)
└── tts/                     # TTS 합성(Edge TTS/VITS)
```
