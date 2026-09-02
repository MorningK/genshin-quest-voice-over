# genshin-quest-voice-over

<p align="center">
  <img src="assets/logo/logo.svg" alt="genshin-quest-voice-over" width="180">
</p>

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
| OCR 前処理（画像強調 + 字幕領域フォーカス） | `ocr-preprocess` | `uv sync --extra ocr-preprocess` |

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

> 字幕領域フォーカス：`ocr-preprocess` グループ（OpenCV）をインストールすると、OCR 前にグレースケール / コントラスト強調と軽度の拡大を行い、認識結果から画面下部の対白帯テキストにフォーカスして、右側のオプションメニュー、右上のパフォーマンス数値（FPS/GPU）、ゲームパッドのボタンヒント（例：`X 再生中`）、UID などの UI ノイズを自動的に除外します。読み上げられるのはプレイヤーが実際に見ている会話内容のみです。`「」` / `《》` で囲まれた NPC 名ラベルもフィルタされます。グループ未インストール時は自動的に全画面テキストへフォールバックし、既存の動作は変わりません。

## PyPI からインストール（コマンドライン版）

PyPI 配布パッケージにはコマンドラインプログラムとそのエンジンバックエンドのみが含まれ、**デスクトップ GUI**（後述のパッケージング章を参照）と Web サービスは含まれません。

```bash
# プログラム本体のみをインストール
uv tool install genshin-quest-voice-over
# または
pipx install genshin-quest-voice-over

# オプションのバックエンドも同時にインストール
uv tool install "genshin-quest-voice-over[capture,ocr-rapid,ocr-preprocess,tts-online,playback]"
```

インストール後は `gqvo` コマンドを使用します（長い形式の `genshin-quest-voice-over` も同等です）。引数は後述の `python main.py` と完全に同じです：

```bash
gqvo --help
gqvo --select-region --fps 3
```

| シナリオ | インストールが必要なオプショングループ |
| --- | --- |
| 画面キャプチャ | `capture`（DXCam は Windows 専用。Linux / macOS は自動的に MSS へフォールバック） |
| OCR 認識 | `ocr-rapid`（デフォルトバックエンド。ONNX モデルはパッケージに同梱） |
| 字幕領域フォーカス | `ocr-preprocess` |
| オンライン音声合成 | `tts-online`（Edge TTS、ネットワーク接続が必要） |
| ストリーミング再生 / MP3 デコード | `playback`（未インストール時は winsound にフォールバックし、MP3 はスキップ） |

> 説明：
>
> - 本リポジトリの Web サービスは Vercel にデプロイされており（Vercel は主依存のみをインストールし、オプショングループはインストールしません）、そのため `fastapi` と `python-multipart` が主依存に残っています。CLI 自体はこれらを使用しませんが、インストール時には一緒に導入されます。
> - `ocr-rapid`（CPU）と `ocr-rapid-gpu`（GPU）の排他関係は **uv の `[tool.uv].conflicts`** で宣言されています。`uv` では強制されますが、`pip` はこの制約を認識しないため、どちらか一方を手動でインストールしてください。

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

# ローカル保存済みの設定を無視し、組み込みの既定値で起動（終了時に既定値が元の設定を上書きするため、リセットと同等）
uv run python main.py --reset-config
```

`Ctrl+C` でグレースフルに停止し、リソースを解放します。

### 設定の自動保存と復元

各実行の**終了時**に、実際に有効だった設定を `~/.genshin-quest-voice-over/config.json` に書き込み、次回起動時に自動的に読み込んで適用します。キャプチャ領域、モニター、音声、フレームレートなどを再設定する必要はありません。

- **優先度**：組み込み既定値 ← 設定ファイルの履歴値 ← コマンドラインの明示引数。コマンドラインは明示的に渡した項目のみを上書きし、それ以外は前回保存された値を引き継ぎます（例：一度領域を選択していれば、次回は `uv run python main.py` だけでその領域が再利用されます）。
- **保存済みスイッチの無効化**：ブールスイッチはコマンドラインで有効にすることしかできず、無効化できないため、設定ファイルから復元されたスイッチを個別に無効化する `--no-verbose` / `--no-gpu` / `--no-full-frame` / `--no-text-direction` を用意しています。
- **GUI**：起動時に保存済み設定をフォームへ反映し、「開始」クリック時とウィンドウ終了時にそれぞれ保存します。
- **既定値へ戻す**：`--reset-config` を付けて起動するか、設定ファイルを削除してください。
- **異常時のフォールバック**：ファイルの欠落、内容の破損、構造バージョンの非互換がある場合は組み込み既定値へフォールバックし、起動を中断せずにログ出力します。不正なフィールドは個別に破棄され、残りのフィールドは通常どおり適用されます（領域の 4 座標のいずれかが不正な場合は領域全体を破棄してフルスクリーンにフォールバックし、0 で埋めて誤ったキャプチャ範囲を作ることはありません）。

## Windows 実行ファイルのパッケージング

GUI（`gui.py`）は PyInstaller で **Windows x64 の one-dir プログラム**としてパッケージングされ、Python をインストールせずにダブルクリックで実行できます。パッケージング設定はリポジトリルートの `gui.spec` に集約されており、ローカルビルドと CI が同じ設定を共有して再現性を確保しています。

### ローカルビルド

```bash
# パッケージングに必要な全オプショングループ + build グループ（pyinstaller）をインストール
uv sync --extra gui --extra capture --extra ocr-rapid --extra ocr-preprocess --extra tts-online --extra playback --group build

# ビルド（成果物は dist/GenshinQuestVoiceOver/）
uv run pyinstaller gui.spec --noconfirm --distpath dist --workpath build/pyinstaller
```

成果物の構成：`dist/GenshinQuestVoiceOver/GenshinQuestVoiceOver.exe` + `_internal/`（依存関係とモデル）。
配布時は `GenshinQuestVoiceOver` ディレクトリごと圧縮してください。**exe 単体をコピーしてはいけません**。

単一ファイル exe ではなく one-dir を採用している理由：依存には onnxruntime / RapidOCR モデル / OpenCV（約 275MB）が含まれ、単一ファイル版は起動のたびに全体を一時ディレクトリへ展開するため起動が遅く、ウイルス対策ソフトに誤検知される可能性も明確に高くなります。

### パッケージング範囲

| 依存グループ | 同梱 | 説明 |
| --- | --- | --- |
| `gui`（CustomTkinter） | はい | テーマリソース `assets/themes/*.json` を含む |
| `capture`（DXCam / MSS） | はい | DXCam は Windows 専用で、comtypes 経由で DXGI/D3D11 を呼び出す |
| `ocr-rapid`（RapidOCR + onnxruntime） | はい | ONNX モデルは wheel に同梱されており、データファイルとして収集する必要がある |
| `ocr-preprocess`（OpenCV headless） | はい | 未インストール時は全画面テキストへ自動フォールバック |
| `tts-online`（Edge TTS） | はい | ネットワーク接続が必要 |
| `playback`（miniaudio） | はい | 未インストール時は winsound の一括再生へフォールバック |
| `ocr`（PaddleOCR / PaddlePaddle） | いいえ | 数百 MB 増加するため、代替バックエンドとしてのみ |
| `ocr-rapid-gpu`（onnxruntime-gpu） | いいえ | CPU 版と排他 |
| Web 依存（fastapi / uvicorn など） | いいえ | GUI 経路から参照がないため明示的に除外 |

テーマファイル `src/genshin_voice_over/gui/assets/genshin_theme.json` は `.py` ではないためモジュールとして収集されず、`gui.spec` の `datas` で明示的に宣言しています。`gui.py` は凍結実行時に `sys._MEIPASS` を基準ディレクトリとしてこのパスを解決します。

### リリースフロー

`.github/workflows/release-desktop.yml` は **Release の公開時（`release: published`）に自動的にトリガー**されます：
チェックアウト → `setup-uv`（キャッシュキーは `uv.lock`）→ `uv sync --frozen`（全オプショングループ + `build` グループ）→ `ruff check`
→ `pyinstaller gui.spec` → テーマ / モデル / DLL の同梱確認 → exe 起動スモークテスト（20 秒間プロセスが生存していれば合格）
→ zip 圧縮 → Release アセットとしてアップロード。

- 成果物名：`genshin-quest-voice-over-<tag>-win-x64.zip`。バージョンは Release タグから取得し、同名アセットは上書きされます。
- Actions ページの **Run workflow** から手動トリガーも可能です（artifact のアップロードのみで、Release は変更しません）。
- アセットのアップロードには `contents: write` が必要で、ワークフローで宣言済みです。組み込みの `GITHUB_TOKEN` を使用するため追加のシークレット設定は不要です。

### 使用とトラブルシューティング

解凍後 `GenshinQuestVoiceOver.exe` をダブルクリックしてください。ログは GUI のログパネルに表示されます（コンソールウィンドウはありません）。
設定とデバッグ用スクリーンショットは従来どおり `~/.genshin-quest-voice-over/` に書き込まれ、exe の配置場所には依存しません。

| 現象 | 確認事項 |
| --- | --- |
| 起動しても反応がない、すぐ終了する | exe が `_internal/` と同じ階層にあるか確認。ウイルス対策ソフトで隔離されていないか確認（one-dir は単一ファイルより誤検知されにくいですが、初回実行時は許可が必要な場合があります） |
| OCR の初期化に失敗する | `_internal/rapidocr/` 配下の `.onnx` モデルファイルが揃っているか確認 |
| 音が出ない / ログに `miniaudio is not installed` | `_internal/` 配下に `_miniaudio.pyd` と `_cffi_backend*.pyd` が両方あるか確認。後者は miniaudio の cffi ABI 拡張が実行時に動的インポートするもので、欠落すると winsound へ静かにフォールバックします |
| キャプチャに失敗する | DXCam は Windows 10 以降かつ排他フルスクリーン以外のモードが必要です。失敗時は自動的に MSS へフォールバックします |

## PyPI への公開

`.github/workflows/publish-pypi.yml` は **Release の公開時（`release: published`）に自動的にトリガー**されます：
`uv build` → `pyproject.toml` のバージョンと Release タグの一致確認（先頭の `v` は許容）
→ wheel チェック（`cli.py` と各エンジンサブパッケージを含むこと、`gui/` と `server.py` を含まないこと）→ `twine check`
→ Trusted Publishing でアップロード。

### リリース手順

1. `pyproject.toml` の `[project].version` を手動で更新します（バージョンは手動管理で、タグからは導出しません）。
2. コミット後、同名タグを作成して Release を公開します（例：`v0.2.0`）。二者が一致しない場合ワークフローは失敗します。
3. ワークフロー完了後、`pip install genshin-quest-voice-over==0.2.0` でインストールできます。

### 初回設定（PyPI 側）

公開には API トークンが不要な **Trusted Publishing（OIDC）** を使用します。PyPI プロジェクトの
*Publishing → Trusted Publishers* に次の内容を追加してください（各フィールドはワークフローと完全に一致させる必要があります）：

| フィールド | 値 |
| --- | --- |
| PyPI Project Name | `genshin-quest-voice-over` |
| Owner | `MorningK` |
| Repository name | `genshin-quest-voice-over` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

### ローカルからの公開（フォールバック）

```bash
uv build
uv publish --token <pypi-token>   # 事前に export UV_PUBLISH_TOKEN=... しても可
```

### 手動検証

Actions ページから **Run workflow** で `Publish to PyPI` をトリガーすると、ビルド・検証・artifact アップロードのみを行い、公開はしません。

## Web サービス（FastAPI + SSE）

本プロジェクトは FastAPI ベースの Web サービス（`server.py`）も提供しています。フロントエンドからの画像アップロードとオプションのパラメータは **multipart POST リクエスト**として受け取り、画像に OCR を実行したうえで、認識テキストとストリーミング TTS 音声を **SSE（Server-Sent Events）レスポンス**として同じストリームで返します。処理フローはデスクトップ版の `pipeline.py` と揃えています。

### インターフェース

| インターフェース | メソッド | 説明 |
|------|------|------|
| `/` | GET | フロントエンドページ（画像アップロード + パラメータ設定 + 受信しながら再生） |
| `/api/voice` | POST | SSE ストリーミングインターフェース。`image` を multipart でアップロードし、`language`/`voice`/`rate`/`ocr_backend`/`tts_backend` を任意のフォーム項目として指定 |
| `/api/voices` | GET | 現在の TTS エンジンが対応する音声の一覧を返す |
| `/health` | GET | ヘルスチェック |

`/api/voice` のイベントストリーム：`event: text`（認識結果 JSON）→ 複数の `event: audio`（base64 エンコードされた MP3 チャンク）→ `event: done`。エラー時は `event: error` を配信します。

### ローカル実行

OCR/TTS エンジンの実行時依存はすでに `[project].dependencies` に含まれています。ローカル開発ではさらに `uvicorn`（オプショングループ `web`）が必要です：

```bash
uv sync --extra web
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

> `uvicorn` はローカル実行専用で、Vercel 関数に同梱しないよう基本依存からオプショングループ `web` へ移しています（Vercel は独自の ASGI ランタイムで `app` を読み込むため uvicorn は不要です）。

ブラウザで `http://localhost:8000` を開くと利用できます。OCR 後は対白帯フォーカステキスト（`roi_text`、`ocr-preprocess` が必要）を優先し、空の場合は全フレームテキストへフォールバックします。

> サーバー側エンジンは遅延初期化 + シングルトンキャッシュを採用しており、初回リクエスト時に初期化され、以降のリクエストで再利用されるためコールドスタートコストを抑えられます。

### Vercel へのデプロイ

リポジトリルートの `server.py` が `app = FastAPI()` を公開しており、Vercel が自動的にエントリポイントとして認識します。Web/OCR/TTS の実行時依存は `pyproject.toml` の `[project].dependencies` に配置済みで、付随する `vercel.json`（関数設定）も準備されています。

```bash
# Vercel CLI をインストール
npm i -g vercel

# プロジェクトルートで
vercel           # ローカルプレビュー
vercel deploy    # 本番デプロイ
```

または [Vercel Dashboard](https://vercel.com) から本リポジトリを接続して直接インポートします。

注意事項：

- **依存のインストール**：Vercel は `pyproject.toml` を優先的に読み込み、`[project].dependencies` のみをインストールしてオプション依存グループはインストールしません。そのため Web/OCR/TTS の実行時依存（fastapi/python-multipart/numpy/onnxruntime/rapidocr/edge-tts/opencv-python-headless）はすべて `[project].dependencies` に置き、Vercel がネイティブにインストールして関数バンドルに同梱されるようにしています。`uvicorn` はローカル開発専用としてオプショングループ `web` に残し、Vercel には同梱しません。`vercel.json` に `installCommand` は不要になりました。なお、依存は必ず `[project].dependencies` に置く必要があります。そうしないと Vercel はビルドに成功しても実行時に読み込みに失敗します（例：`ModuleNotFoundError: No module named 'rapidocr'`）。

- **Large Functions の有効化（必須）**：本サービスは `onnxruntime` / `rapidocr` / `opencv` に依存し、バンドルサイズは 600MB 以上です。Large Functions を有効にしない場合、Vercel はバンドルに対して **"optimizing dependencies"** を実行し、標準上限に収めるために `onnxruntime` / `rapidocr` などの大容量ネイティブ依存を**バンドルから除外**するため、デプロイは成功しても実行時に `ModuleNotFoundError: No module named 'rapidocr'` になります。そのため Large Functions（上限 5GB）を有効にし、バンドルが大関数パスを通って削除されないようにする必要があります。有効化手順（いずれも Vercel のプロジェクト設定で手動構成が必要で、`vercel.json` では設定できません）：
  1. プロジェクトの **Settings → General** で **Fluid Compute** が有効であることを確認（新規プロジェクトは既定で有効）。
  2. プロジェクトの **Settings → Environment Variables** に `VERCEL_SUPPORT_LARGE_FUNCTIONS = 1` を追加。
  設定後は**再デプロイ**が必要です。ビルドログに "optimizing dependencies" が表示されなくなった場合（またはバンドルが明らかに 500MB を超えて正常にデプロイされた場合）、有効化されています。

- **リクエストボディの上限（4.5MB）**：Vercel 関数のリクエスト / レスポンスボディの上限は 4.5MB で、巨大な画像をアップロードすると `FUNCTION_PAYLOAD_TOO_LARGE` になります。フロントエンドは `static/index.html` で**クライアント側圧縮**（Canvas で最長辺 1600px に等比縮小して JPEG 化し、約 3.5MB 以内まで段階的に品質を下げる）を行い、アップロードサイズを上限未満に抑えています。サーバー側の OCR も画像を最長辺 1280px に縮小するため、認識精度には影響しません。フロントエンドを経由せず直接 API を呼ぶ場合は、画像サイズを自己管理してください。

- **関数の実行時間とリソース設定**：`vercel.json` は `functions.server.py.maxDuration: 60` と `excludeFiles` のみを設定し、**`memory` は設定していません**。Fluid Compute では Hobby の実行時間上限は 300 秒ですが、本関数は `maxDuration: 60` により明示的に 60 秒に制限されています。より長い時間が必要な場合は Vercel コンソールで調整してください。memory と CPU も Vercel コンソールの **Functions** 設定で構成する必要があります（Fluid Compute では `vercel.json` で設定できません）。

- Vercel の serverless はコールドスタートが遅く（初回に OCR/TTS 依存の読み込みと音声一覧のネットワーク取得が発生）、重量な OCR モデルとオンライン TTS はネットワーク制限環境では制約を受ける可能性があります。本番運用では ローカルの `uvicorn` や常駐プロセスを持つプラットフォームを主とし、Vercel は軽量なデモ / 共有用エントリとして位置づけることを推奨します。

### OCR 実行時エラーの診断

デプロイ後に画像をアップロードして `Failed to import rapidocr/onnxruntime: ...` を含む `event: error` が返る場合、次の表で根本原因を判別して対処してください（エラー文言には元の `ImportError` の理由が含まれ、ログには完全な traceback が出力されます）：

| エラー内の根本原因 | 意味 | 対処 |
| --- | --- | --- |
| `No module named 'onnxruntime'` / `No module named 'rapidocr'` | 依存が Vercel の "optimizing dependencies" によって関数バンドルから除外された | Large Functions が完全に有効（`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` + Fluid Compute + Active CPU）であることを確認して再デプロイ |
| `libgomp.so.1: cannot open shared object file` など | Vercel のランタイムイメージに `onnxruntime` が必要とするシステムライブラリがない | onnxruntime は追加のシステムライブラリを必要とし、Vercel イメージが満たさない可能性があります。他の常駐プラットフォームの利用や依存の調整を検討 |
| その他の `cannot open shared object` / `undefined symbol` | ネイティブライブラリの ABI と実行環境が不一致 | `onnxruntime` のバージョンを変更するか、別のデプロイプラットフォームに切り替える |

> ヒント：エラー原因は SSE `error` イベントの `detail` フィールド（`cause:` チェーンを含む）でも返されるため、サーバーログだけでなくブラウザページで直接確認できます。

## コード構成

```text
main.py                              # リポジトリ内 CLI 起動シェル。genshin_voice_over.cli:main へ転送
gui.py                               # デスクトップ GUI のエントリポイント（CustomTkinter）
server.py                            # Web サービスのエントリポイント（FastAPI + SSE）
gui.spec                             # PyInstaller パッケージング設定（GUI → Windows exe）
src/genshin_voice_over/              # インポート可能なトップレベルパッケージ（src-layout）
├── cli.py                           # CLI 実装。console script `gqvo` の参照先
├── common.py                        # 共有データ型（Point/Region/SelectedRegion）
├── app/                             # アプリケーションのオーケストレーション
│   ├── config.py                    # 実行設定と CLI 解析
│   ├── pipeline.py                  # VoiceOverApp メインパイプライン
│   ├── region_selector.py           # 対話式画面領域選択（tkinter、マルチモニター対応）
│   ├── monitor.py                   # モニター列挙とマルチスクリーン座標変換
│   ├── textproc.py                  # テキストのクレンジング / 重複排除 / 変化検出
│   └── player.py                    # オーディオ再生（winsound / miniaudio）
├── capture/                         # 画面キャプチャ（DXCam/MSS）
├── recognition/                     # OCR 認識（PaddleOCR/RapidOCR）
├── tts/                             # TTS 合成（Edge TTS/VITS）
└── gui/                             # デスクトップ GUI（exe にのみ同梱、PyPI パッケージには含まれない）
```
