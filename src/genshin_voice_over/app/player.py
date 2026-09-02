"""音频播放抽象与基于 winsound 的实现。

MVP 阶段使用 Windows 内置的 ``winsound`` 播放音频，零第三方依赖。
通过抽象接口 ``AudioPlayer`` 屏蔽播放细节，便于后续接入 PyAudio 等专业播放库。
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import queue
import tempfile
import threading
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from types import ModuleType

    from genshin_voice_over.tts import TTSResult

logger = logging.getLogger(__name__)


def _decode_to_wav(audio_data: bytes, sample_rate: int = 44100, channels: int = 2) -> bytes:
    """用 miniaudio 将非 WAV 音频解码并封装为 WAV 字节。

    使用 ``miniaudio.decode`` 自动识别输入容器，可解码 MP3、OGG、FLAC 等格式。

    Args:
        audio_data: 待解码的音频字节。
        sample_rate: 目标采样率，miniaudio 会重采样到此值。
        channels: 目标声道数。

    Returns:
        可被 winsound 播放的 WAV 字节。

    Raises:
        RuntimeError: 依赖库 miniaudio 未安装时抛出。
    """
    try:
        import miniaudio  # pyrefly: ignore=missing-import  # 惰性导入，未安装时优雅降级
    except ImportError as exc:
        raise RuntimeError(
            "miniaudio is not installed. Run `uv sync --extra playback` to enable non-WAV playback."
        ) from exc

    decoded = miniaudio.decode(
        audio_data,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=channels,
        sample_rate=sample_rate,
    )
    pcm = decoded.samples.tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        # 写入实际解码得到的声道数，确保 WAV 头与 PCM 负载一致
        wav_file.setnchannels(decoded.nchannels)
        wav_file.setsampwidth(decoded.sample_width)
        wav_file.setframerate(decoded.sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class AudioPlayer(ABC):
    """音频播放抽象基类。

    所有播放器实现必须继承此类并实现 play/release 方法。
    生命周期：initialize() → play()/play_stream() 循环 → release()

    提供两种播放模式：
    - play: 一次性播放一段完整音频，阻塞到播完（降级路径）。
    - play_stream: 流式播放，边接收 chunk 边播放，与 TTS 合成并行（低延迟路径）。
    默认实现假定不支持流式，子类可通过覆写 supports_streaming/play_stream 开启。
    """

    @abstractmethod
    def play(self, result: TTSResult) -> None:
        """播放一段合成音频。

        Args:
            result: TTS 合成结果，包含音频数据与格式信息。
        """
        ...

    def play_stream(self, chunks: Iterator[TTSResult]) -> None:
        """流式播放合成音频片段。

        逐块从 ``chunks`` 读取音频数据并立即播放，与上游合成并行，降低感知延迟。
        默认实现不支持流式，抛出 NotImplementedError，调用方应据此降级为 play()。

        Args:
            chunks: TTS 流式合成结果迭代器。

        Raises:
            NotImplementedError: 该播放器不支持流式播放时抛出。
        """
        raise NotImplementedError("This player does not support streaming playback.")

    @property
    def supports_streaming(self) -> bool:
        """当前是否支持流式播放（play_stream）。

        默认返回 False，子类（如基于 miniaudio 的流式播放器）可覆写为 True。
        """
        return False

    @abstractmethod
    def release(self) -> None:
        """释放播放器占用的资源。

        调用后如需再次使用，需要重新调用 initialize()。
        """
        ...

    @abstractmethod
    def initialize(self) -> bool:
        """初始化播放能力。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: 初始化失败时抛出。
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        ...


class WinsoundPlayer(AudioPlayer):
    """基于 winsound 的音频播放实现。

    生命周期：initialize() → play() 循环 → release()。

    ``winsound.PlaySound`` 原生仅支持 WAV：
    - 对 ``format == "wav"`` 的音频直接写入临时文件并播放；
    - 对 MP3 等其他格式，优先用 ``miniaudio`` 解码为 WAV 后再播放；
    - 若 miniaudio 未安装，记录提示日志并跳过，不阻塞主循环。

    依赖库 ``winsound`` 仅在 initialize() 时惰性导入，
    非 Windows 平台未安装时抛出带明确提示的 RuntimeError。
    """

    def __init__(self) -> None:
        """初始化实例，尚未建立播放能力。"""
        self._winsound: ModuleType | None = None
        self._initialized = False

    def initialize(self) -> bool:
        """初始化 winsound 播放能力。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: winsound 不可用（非 Windows 平台）时抛出。
        """
        try:
            import winsound  # 非 Windows 平台导入失败，winsound 为系统内置模块
        except ImportError as exc:
            raise RuntimeError("winsound is not available on this platform.") from exc

        self._winsound = winsound
        self._initialized = True
        logger.info("Winsound player initialized.")
        return True

    def play(self, result: TTSResult) -> None:
        """播放一段合成音频。

        Args:
            result: TTS 合成结果。

        Raises:
            RuntimeError: 播放器未初始化时抛出。
        """
        if not self._initialized or self._winsound is None:
            raise RuntimeError("Winsound player is not initialized.")
        if not result.audio_data:
            return

        audio_format = (result.format or "wav").lower()
        if audio_format == "wav":
            wav_data = result.audio_data
        else:
            # winsound 原生仅支持 WAV，先用 miniaudio 将音频解码为 WAV
            try:
                wav_data = _decode_to_wav(result.audio_data)
            except RuntimeError as exc:
                logger.info("Skip playback: %s", exc)
                return
            except Exception as exc:  # noqa: BLE001 - 解码失败不应中断主循环
                logger.warning("Failed to decode audio format '%s': %s", audio_format, exc)
                return

        # 用 mkstemp 独占创建临时文件，避免基于可预测路径的不安全写入
        fd, temp_path_str = tempfile.mkstemp(prefix="genshin_vo_", suffix=".wav")
        os.close(fd)
        temp_path = Path(temp_path_str)
        try:
            temp_path.write_bytes(wav_data)
            self._winsound.PlaySound(str(temp_path), self._winsound.SND_FILENAME)
        except Exception as exc:  # noqa: BLE001 - 播放失败不应中断主循环
            logger.warning("Failed to play audio: %s", exc)
        finally:
            temp_path.unlink(missing_ok=True)

    def release(self) -> None:
        """释放播放器占用的资源。"""
        self._winsound = None
        self._initialized = False
        logger.info("Winsound player released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized


class MiniAudioPlayer(AudioPlayer):
    """基于 miniaudio.PlaybackDevice 的流式播放实现。

    生命周期：initialize() → play()/play_stream() → release()。

    使用 miniaudio 的低延迟输出设备，通过回调生成器按需拉取 PCM 数据，
    与上游 TTS 流式合成并行，实现真正的边合成边播放，降低感知延迟。

    - play_stream(chunks): 流式播放，边解码 MP3/PCM chunk 边写入输出设备。
    - play(result): 兼容一次性播放，阻塞到播放结束（降级路径）。
    - supports_streaming: 恒为 True。

    依赖库 ``miniaudio`` 仅在 initialize() 时惰性导入，
    未安装时抛出带明确提示的 RuntimeError。
    """

    #: SIGNED16 采样格式下每样本字节数
    _SAMPLE_BYTES = 2
    #: PCM 缓冲队列容量（上限，防止原始 chunk 解码速度过快导致内存膨胀）
    _BUFFER_MAX_CHUNKS = 64
    #: 设备回调在缓冲暂时耗尽时，等待解码线程补充的最长时间（秒）。
    #: 解码通常能在数十毫秒内补充数据，优先等待而非立即插入静音，避免产生爆音。
    _BUFFER_WAIT_TIMEOUT = 0.1
    #: 缓冲不足且等待超时后回退输出的静音时长（秒），用于平滑网络/解码尖峰。
    _SILENCE_BRIDGE_SECONDS = 0.02
    #: done 等待的轮询间隔（秒），设备异常停止时用于超时退出，避免主循环挂死。
    _DONE_POLL_SECONDS = 0.5
    #: 播放结束后排空缓冲队列、等待解码线程退出的总截止时间（秒）。
    #: 超过该时限后放弃等待，防止解码线程异常时清理过程被无限阻塞。
    _CLEANUP_TIMEOUT_SECONDS = 5.0

    def __init__(self, nchannels: int = 2, sample_rate: int = 44100) -> None:
        """初始化实例，尚未建立播放能力。

        Args:
            nchannels: 输出声道数。
            sample_rate: 输出采样率（Hz）。
        """
        self._nchannels = nchannels
        self._sample_rate = sample_rate
        self._miniaudio: Any = None
        self._device: Any = None
        self._initialized = False

    def initialize(self) -> bool:
        """初始化 miniaudio 输出设备。

        Returns:
            True 表示初始化成功。

        Raises:
            RuntimeError: 依赖库 miniaudio 未安装或初始化失败时抛出。
        """
        try:
            import miniaudio  # pyrefly: ignore=missing-import  # 惰性导入，未安装时优雅降级
        except ImportError as exc:
            raise RuntimeError(
                "miniaudio is not installed. Run `uv sync --extra playback` to enable streaming playback."
            ) from exc

        try:
            device = miniaudio.PlaybackDevice(
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=self._nchannels,
                sample_rate=self._sample_rate,
            )
        except Exception as exc:  # noqa: BLE001 - 设备初始化失败统一转为 RuntimeError
            raise RuntimeError(f"Failed to initialize miniaudio playback device: {exc}") from exc

        self._miniaudio = miniaudio
        self._device = device
        self._initialized = True
        logger.info("MiniAudio player initialized.")
        return True

    def play(self, result: TTSResult) -> None:
        """一次性播放一段合成音频，阻塞到播放结束。

        Args:
            result: TTS 合成结果。

        Raises:
            RuntimeError: 播放器未初始化时抛出。
        """
        if not self._initialized or self._device is None:
            raise RuntimeError("MiniAudio player is not initialized.")
        self.play_stream(iter([result]))

    def play_stream(self, chunks: Iterator[TTSResult]) -> None:
        """流式播放合成音频片段，边接收边播放。

        通过设备回调生成器按需从 ``chunks`` 拉取音频并解码播放，
        与上游 TTS 合成并行。方法阻塞到整个流播放完成。

        Args:
            chunks: TTS 流式合成结果迭代器。

        Raises:
            RuntimeError: 播放器未初始化时抛出。
        """
        if not self._initialized or self._device is None or self._miniaudio is None:
            raise RuntimeError("MiniAudio player is not initialized.")

        miniaudio = self._miniaudio
        device = self._device
        nchannels = self._nchannels
        sample_rate = self._sample_rate
        sample_bytes = self._SAMPLE_BYTES
        framesize = nchannels * sample_bytes
        done = threading.Event()

        # 已解码的 PCM 缓冲队列：解码工作线程写入，设备回调线程消费。
        # 队列作为缓冲层吸收网络传输与 MP3 解码的抖动，保证设备回调始终有数据可播，
        # 避免因同步解码导致的播放停顿。每个元素为流式解码器产出的一段 PCM。
        pcm_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=self._BUFFER_MAX_CHUNKS)
        decode_error: list[BaseException | None] = []

        class _ChunkStreamSource(miniaudio.StreamableSource):
            """将 TTS 流式 chunk 字节暴露为 miniaudio 的连续字节流。

            miniaudio 的 MP3 解码器在逐块解码时会因块边界非帧对齐而丢失部分帧，
            导致音频变短、语速变快。通过本类把原始 chunk 字节连续地喂给同一个
            解码器，保持解码状态，从而避免丢帧。
            """

            def __init__(self) -> None:
                self._buffer = b""
                self._done = False
                self._iter = iter(chunks)

            def _refill(self) -> None:
                while not self._buffer and not self._done:
                    try:
                        chunk = next(self._iter)
                    except StopIteration:
                        self._done = True
                        break
                    if chunk.audio_data:
                        self._buffer += chunk.audio_data

            def read(self, num_bytes: int) -> bytes | memoryview:
                """返回最多 ``num_bytes`` 字节，字节不足时从 chunk 迭代器补充。"""
                self._refill()
                if not self._buffer:
                    return b""
                segment = self._buffer[:num_bytes]
                self._buffer = self._buffer[num_bytes:]
                return segment

            def seek(self, _offset: int, _origin: Any) -> bool:
                """流式源不可 seek，返回 False。"""
                return False

        def _decode_worker() -> None:
            """后台解码线程：以流式解码器保持状态地解码 chunk 流，PCM 整块入队。"""
            try:
                source = _ChunkStreamSource()
                # 用同一个流式解码器处理所有 chunk，保持解码器内部状态，
                # 避免逐块 miniaudio.decode 在块边界丢失帧导致音频变短。
                decoder = miniaudio.stream_any(
                    source,
                    source_format=miniaudio.FileFormat.MP3,
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=nchannels,
                    sample_rate=sample_rate,
                )
                for pcm_arr in decoder:
                    pcm = pcm_arr.tobytes()
                    if not pcm:
                        continue
                    pcm_queue.put(pcm)
            except BaseException as exc:  # noqa: BLE001 - 跨线程传递异常
                decode_error.append(exc)
            finally:
                pcm_queue.put(None)  # 哨兵：标记解码结束

        worker = threading.Thread(target=_decode_worker, daemon=True)
        worker.start()

        # 预取首个 PCM 块，让解码线程先跑起来、缓冲队列尽快填满，
        # 设备一启动就有数据可播，降低首帧延迟与初期停顿。
        # 若取到结束哨兵（空流或首个 chunk 解码失败），则无需播放直接返回。
        first = pcm_queue.get()
        pcm_queue.task_done()
        if first is None:
            worker.join(timeout=5.0)
            return

        def _feed(initial: bytes) -> Generator[bytes, int, None]:
            """设备回调生成器：从缓冲队列累积整块已解码 PCM，按请求帧数切片播放。

            回调线程对实时性要求较高，此处优先**非阻塞**消费缓冲队列：
            - 队列有数据 → 持续累积到满足请求帧数后一次性返回，保持播放连续；
            - 队列空但仍在解码 → 先短时阻塞等待解码线程补充，尽量复用真实音频避免爆音；
              等待超时仍未补充才输出少量静音桥接，避免回调无数据可返导致设备欠载；
            - 队列空且解码已结束 → 返回剩余数据后收尾。

            Args:
                initial: 预取的首个已解码 PCM 块，作为初始缓冲，避免句首被丢弃。
            """
            frames_required = yield b""  # 素数化占位
            need = max(frames_required, 1) * framesize
            pending = initial
            while True:
                # 连续非阻塞取队列累积，直到 pending 满足请求帧数，或队列暂时无数据。
                while len(pending) < need:
                    try:
                        pcm = pcm_queue.get_nowait()
                        pcm_queue.task_done()
                    except queue.Empty:
                        break
                    if pcm is None:
                        # 解码已结束：返回剩余数据后收尾
                        if pending:
                            frames_required = yield pending
                        done.set()
                        return
                    pending += pcm
                if len(pending) >= need:
                    # 累积足够后按请求帧数一次性切片返回，保证对齐
                    segment = pending[:need]
                    pending = pending[need:]
                    frames_required = yield segment
                    need = max(frames_required, 1) * framesize
                    continue
                # 队列空且未结束：先短时阻塞等待解码线程补充真实音频，
                # 减少全零静音块的插入，避免静音交界处产生爆音。
                try:
                    pcm = pcm_queue.get(timeout=self._BUFFER_WAIT_TIMEOUT)
                    pcm_queue.task_done()
                except queue.Empty:
                    # 阻塞等待超时仍未补充，输出静音桥接等待下次补充。
                    # 静音长度与设备请求帧数精确对齐，避免回调欠载。
                    frames_required = yield b"\x00" * need
                    need = max(frames_required, 1) * framesize
                    continue
                if pcm is None:
                    # 解码已结束：播完剩余数据后收尾，避免结尾静音吞掉末尾音频。
                    if pending:
                        frames_required = yield pending
                    done.set()
                    return
                if len(pcm) == 0:
                    continue
                # 等待到真实数据：回到累积循环继续补充
                pending += pcm

        generator = _feed(first)
        playback_failed = False  # 标记播放阶段是否已抛出异常，供 finally 决定是否传播解码错误
        try:
            next(generator)  # 素数化生成器，满足 PlaybackDevice.start 的契约
            device.start(generator)
            # 阻塞等待设备回调消费完整个流；设备异常停止时按轮询超时退出，避免主循环挂死。
            while not done.wait(timeout=self._DONE_POLL_SECONDS):
                # 设备不再调用回调（异常/释放）且解码线程已退出、队列已消费完时提前结束
                if not worker.is_alive() and pcm_queue.empty():
                    break
        except Exception as exc:  # noqa: BLE001 - 播放失败不吞掉异常，交由调用方降级
            logger.warning("Failed to stream playback: %s", exc)
            # 标记播放异常，避免 finally 中传播解码错误时替换此异常
            playback_failed = True
            # 统一包装为 RuntimeError，保证 pipeline 的降级分支能捕获并转入一次性播放
            raise RuntimeError(f"Failed to stream playback: {exc}") from exc
        finally:
            with contextlib.suppress(Exception):  # 停止设备失败可忽略
                device.stop()
            with contextlib.suppress(Exception):  # 关闭生成器异常可忽略，不替换原始异常
                generator.close()
            # 排空有界队列，解除解码线程在 put 上的阻塞，确保 worker 可退出、避免线程泄漏。
            # 用有限超时持续轮询：get_nowait 可能在 worker 进入下一次 put 前的空窗期遇到
            # queue.Empty 而提前退出，导致 worker 重新填满队列后再次阻塞，故需反复排空直到
            # worker 退出或到达清理截止时间。
            deadline = time.monotonic() + self._CLEANUP_TIMEOUT_SECONDS
            while worker.is_alive() and time.monotonic() < deadline:
                try:
                    pcm_queue.get(timeout=0.1)
                    pcm_queue.task_done()
                except queue.Empty:
                    continue
            worker.join(timeout=self._CLEANUP_TIMEOUT_SECONDS)
            # 仅当未发生播放异常时才传播解码线程错误，避免替换 except 中已抛出的异常；
            # 播放正常但解码失败时，包装后重新抛出，使 pipeline 能进入一次性降级分支。
            if decode_error and not playback_failed:
                logger.warning("Decode worker ended with error: %s", decode_error[0])
                raise RuntimeError(f"Decode worker ended with error: {decode_error[0]}") from decode_error[0]

    def release(self) -> None:
        """释放播放器占用的资源。"""
        if self._device is not None:
            try:
                self._device.stop()
                self._device.close()
            except Exception:  # noqa: BLE001 - 关闭失败可忽略
                pass
        self._device = None
        self._miniaudio = None
        self._initialized = False
        logger.info("MiniAudio player released.")

    @property
    def is_initialized(self) -> bool:
        """当前是否已初始化并处于可用状态。"""
        return self._initialized

    @property
    def supports_streaming(self) -> bool:
        """当前是否支持流式播放。MiniAudioPlayer 恒为 True。"""
        return True
