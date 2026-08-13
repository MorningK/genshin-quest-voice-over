# genshin-quest-voice-over

> **言語 / Languages**：[中文](README.md) · [English](README.en.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

《原神》のボイスがないクエスト（主に世界任務）の会話テキストをリアルタイムに読み上げるサービスです。

本ツールは画面キャプチャでゲーム内の会話字幕を認識し、OCR でテキストを抽出した後、TTS で音声を合成して再生します。ゲームクライアントは一切変更しません。

## 機能フロー

```
ゲーム起動 → 画面キャプチャ（2-4 FPS）→ OCR テキスト認識 → テキストの重複排除 / 変化検出
    → TTS ストリーミング合成 → ストリーミング再生（miniaudio、合成しながら再生）
    └─ ストリーミング不可時のフォールバック：一括合成 → 再生（winsound / miniaudio）
```

> ストリーミング：TTS エンジンとプレイヤーの両方がストリーミングに対応している場合（Edge TTS + miniaudio）は、合成しながら再生する方式を優先し、エンドツーエンドの体感遅延を低減します。エンジンがストリーミングに対応していない場合（オフラインの VITS スケルトンなど）や `playback`（miniaudio）依存グループをインストールしていない場合は、一括合成 + ブロッキング再生に自動的にフォールバックします。なお、Edge TTS は MP3 を出力し、`winsound` はネイティブに WAV のみをサポートしています。`playback` 依存グループをインストールしていない場合は MP3 をデコードできないため、非 WAV 音声はスキップされ、プログラムは中断されません。

## 環境準備

Python の依存関係管理には [uv](https://docs.astral.sh/uv/) を使用します：

```bash
uv sync
```

コア実行には `numpy` のみが必要で、各バックエンドライブラリはオプションの依存グループを通じて必要に応じて有効化します。オプションの依存は `pyproject.toml` に宣言されており、`uv sync --extra <グループ名>` で有効化します：

| モジュール | オプション依存グループ | 有効化コマンド |
|------|-----------|---------|
| 画面キャプチャ（DXCam + MSS） | `capture` | `uv sync --extra capture` |
| OCR（RapidOCR デフォルト） | `ocr-rapid` | `uv sync --extra ocr-rapid` |
| OCR（PaddleOCR 代替） | `ocr` | `uv sync --extra ocr` |
| OCR GPU（RapidOCR 代替） | `ocr-rapid-gpu` | `uv sync --extra ocr-rapid-gpu` |
| TTS（Edge TTS オンライン） | `tts-online` | `uv sync --extra tts-online` |
| 再生（ストリーミング再生 + 非 WAV デコード） | `playback` | `uv sync --extra playback` |

単一グループの有効化は `uv sync --extra capture --extra ocr-rapid --extra tts-online --extra playback`、または一括で全部有効化する場合は `uv sync --all-extras` を使用します。
> 注意：`ocr-rapid`（CPU）と `ocr-rapid-gpu`（GPU）は排他的で、uv はこれらを競合グループとして宣言しており、`--all-extras` は両方を同時に有効化するためエラーになります。GPU 環境では `--all-extras` を使用せず、明示的に GPU グループを指定してください（下記「GPU アクセラレーション」参照）。

バックエンド依存が有効化されていない場合、アプリは対応する有効化ヒントを表示し、自動的に代替バックエンドへのフォールバックを試みます。

### GPU アクセラレーション（オプション）

OCR 認識はデフォルトで CPU 上で実行され、`--gpu` スイッチで GPU 推論の高速化を有効にできます。GPU 依存は CPU 版と競合するため、バックエンドごとにどちらか一方を選択してインストールする必要があります：

- **RapidOCR（推奨）**：`ocr-rapid` グループの代わりに `ocr-rapid-gpu` グループ（onnxruntime-gpu）を有効化：

  ```bash
  uv sync --extra ocr-rapid-gpu --extra capture --extra tts-online --extra playback
  ```

  `onnxruntime-gpu 1.28.x` には CUDA 13.x と cuDNN 9.x のランタイム環境が必要です。対応する CUDA Toolkit / cuDNN がインストールされ、ライブラリ検索パスが正しく設定されていることを確認してください（Windows は `PATH`、Linux は `LD_LIBRARY_PATH`）。設定されていない場合、GPU 推論は CPU に静かにフォールバックするか、初期化に失敗します。

- **PaddleOCR**：PaddlePaddle 3.x の GPU 版（`paddlepaddle-gpu`）は Paddle 公式ソースのみで公開されており、通常の PyPI 依存としてはインストールできません。[Paddle インストールガイド](https://www.paddlepaddle.org.cn/documentation/zh//install/index_cn.html) から CUDA バージョンに合った公式ソースで GPU 版の `paddlepaddle`（CPU 版の代替）をインストールし、その他の依存は `uv sync --extra ocr` でインストールしてください。

GPU 依存をインストールした後、実行時に `--gpu` を追加すれば高速化が有効になります（それ以外は GPU 依存は使用されず、CPU のままです）。
> 注意：`--gpu` は**ユーザーが要求した** GPU 状態を記録します。実行時に CUDA/cuDNN 環境が不足している場合、RapidOCR は実際には CPU で実行される可能性があり、初期化ログに `gpu_requested` フィールドで要求状態が反映されます。GPU 初期化に失敗した場合、アプリはエラーをスローし、代替の OCR バックエンドへのフォールバックを試みます。
> 注意：Edge TTS は MP3 を出力するため、再生には `playback` グループ（miniaudio）の有効化が必要です。有効化後は miniaudio によるストリーミング再生（合成しながら再生）が使用され、有効化していない場合は非 WAV 音声（Edge TTS の MP3 など）がスキップされます。

> プライバシーについて：Edge TTS（オンライン TTS）を使用する場合、画面キャプチャと OCR で認識されたテキストがネットワーク経由で Microsoft Edge TTS API に送信され、音声合成が行われます。プライバシーが気になる場合は、オフライン TTS（VITS）バックエンドを使用してください。

## 実行

```bash
uv run python main.py
```

よく使うパラメータ（全パラメータは `uv run python main.py --help` を参照）：

```bash
# キャプチャ領域を指定（left,top,right,bottom）し、フレームレートを下げる
uv run python main.py --region 100,200,900,600 --fps 3

# キャプチャ領域を対話的に選択（フルスクリーンオーバーレイ表示、マウスドラッグで選択、Esc でフルスクリーンにフォールバック）
# 拡張ディスプレイに対応：オーバーレイはすべてのモニターを覆い、選択後にモニターを自動検出して座標を変換します
# 注意：--select-region と --region は排他的で、同時には使用できません
uv run python main.py --select-region --fps 3

# 代替バックエンドを使用
uv run python main.py --capture mss --ocr paddle --tts edge

# OCR 言語と TTS 音声を指定
uv run python main.py --language ch --voice zh-CN-XiaoxiaoNeural

# オフライン TTS を使用（VITS、モデルパスを指定する必要があります。現在はスケルトン実装で、実際の推論は未接続）
uv run python main.py --tts vits --tts-model-path /path/to/model

# GPU 高速化 OCR を使用（対応する GPU 依存グループのインストールが必要です。上記「GPU アクセラレーション（オプション）」参照）
# OCR バックエンドを明示指定：rapid（RapidOCR、ocr-rapid-gpu グループが必要）または paddle（PaddleOCR、公式ソースの GPU 版が必要）
uv run python main.py --ocr rapid --gpu
uv run python main.py --ocr paddle --gpu
```

`Ctrl+C` でグレースフルに停止し、リソースを解放します。

## コード構成

```
main.py                      # アプリのエントリポイント：CLI 引数解析 + VoiceOverApp ドライバ
src/
├── common.py                # 共有データ型（Point/Region/SelectedRegion）
├── app/                     # アプリケーションのオーケストレーション
│   ├── config.py            # 実行設定と CLI 解析
│   ├── pipeline.py          # VoiceOverApp メインパイプライン
│   ├── region_selector.py   # 対話式画面領域選択（tkinter、マルチモニター対応）
│   ├── monitor.py           # モニター列挙とマルチスクリーン座標変換
│   ├── textproc.py          # テキストのクレンジング / 重複排除 / 変化検出
│   └── player.py            # オーディオ再生（winsound）
├── capture/                 # 画面キャプチャ（DXCam/MSS）
├── recognition/             # OCR 認識（PaddleOCR/RapidOCR）
└── tts/                     # TTS 合成（Edge TTS/VITS）
```
