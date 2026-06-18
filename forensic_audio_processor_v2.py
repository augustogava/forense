#!/usr/bin/env python3
"""
Forensic Audio Processor v2

Pipeline unificado para recuperar fala em portugues muito baixa e ruidosa:
separacao de voz por deep learning (Demucs), realce de fala por IA
(DeepFilterNet/ClearVoice) e refinamento DSP/matematico com VAD, gerando
variantes ajustaveis e validacao opcional de qualidade (entrada vs saida).
"""

import argparse
import io
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "encoding") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.ndimage import median_filter, uniform_filter1d

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

for _noisy in ("numba", "librosa", "matplotlib", "pydub", "PIL", "torch", "torchaudio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

TARGET_SR = 44100
STFT_CHUNK_SECONDS = 300
N_FFT = 2048
HOP_LENGTH = 512
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}
DEFAULT_VARIANTS = ["clean", "speech_boost", "max"]

_ffmpeg_path = None
try:
    import imageio_ffmpeg

    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    logger.debug(f"Using ffmpeg from imageio-ffmpeg: {_ffmpeg_path}")
except ImportError:
    for _candidate in ("ffmpeg", "ffmpeg.exe"):
        try:
            subprocess.run([_candidate, "-version"], capture_output=True, check=True)
            _ffmpeg_path = _candidate
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

if _ffmpeg_path is None:
    logger.warning("ffmpeg not found; non-wav input/output and mp3 export may fail")


def detect_dependencies() -> Dict[str, bool]:
    import importlib.util as _u

    status: Dict[str, bool] = {}
    for key, mod in (("torch", "torch"), ("demucs", "demucs"), ("deepfilter", "df"), ("clearvoice", "clearvoice"), ("pyloudnorm", "pyloudnorm")):
        try:
            status[key] = _u.find_spec(mod) is not None
        except Exception:
            status[key] = False

    status["cuda"] = False
    if status.get("torch"):
        try:
            import torch

            status["cuda"] = bool(torch.cuda.is_available())
        except Exception:
            status["cuda"] = False
    return status


def print_dependency_status(status: Dict[str, bool]) -> None:
    print("\n  Modulos disponiveis:")
    labels = {
        "demucs": "Demucs (separacao de voz)",
        "deepfilter": "DeepFilterNet (realce de fala)",
        "clearvoice": "ClearVoice (realce de fala)",
        "pyloudnorm": "pyloudnorm (loudness EBU R128)",
        "cuda": "GPU CUDA",
    }
    for key, label in labels.items():
        tag = "OK" if status.get(key) else "indisponivel"
        print(f"    {label:34s} {tag}")


import threading

_print_lock = threading.Lock()


def _tprint(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _ffmpeg_to_wav(input_path: Path, output_wav: Path, sr: int, mono: bool = True) -> None:
    ffmpeg = _ffmpeg_path or "ffmpeg"
    channels = "1" if mono else "2"
    cmd = [ffmpeg, "-y", "-i", str(input_path), "-ar", str(sr), "-ac", channels, str(output_wav)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join([l for l in (result.stderr or "").splitlines() if l.strip()][-5:]) or "(no stderr)"
        raise RuntimeError(f"ffmpeg conversion failed (exit {result.returncode}): {tail}")


def _load_wav_mono(wav_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(data, dtype=np.float32), sr


class ForensicAudioProcessorV2:

    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(self, input_path: str, output_dir: str, deps: Dict[str, bool], config: Dict):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.deps = deps
        self.config = config
        self.sample_rate = TARGET_SR

        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized v2 processor - input: {input_path}, output: {output_dir}, config: {config}")

    # ========== I/O ==========

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        ext = audio_path.suffix.lower()
        if ext == ".wav":
            return _load_wav_mono(audio_path, self.sample_rate)

        temp_wav = self.output_dir / f"_tmp_load_{audio_path.stem}.wav"
        try:
            _ffmpeg_to_wav(audio_path, temp_wav, self.sample_rate, mono=True)
            y, sr = _load_wav_mono(temp_wav, self.sample_rate)
            return y, sr
        finally:
            temp_wav.unlink(missing_ok=True)

    def _save_audio(self, audio: np.ndarray, base_name: str, suffix: str, sr: Optional[int] = None) -> Path:
        sr = sr or self.sample_rate
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        temp_wav = self.output_dir / f"_tmp_{base_name}_{suffix}.wav"
        sf.write(str(temp_wav), audio, sr, subtype="PCM_16")

        if self.config.get("wav"):
            wav_out = self.output_dir / f"{base_name}_{suffix}.wav"
            temp_wav.replace(wav_out)
            logger.debug(f"Saved WAV output: {wav_out.name}")
            return wav_out

        output_path = self.output_dir / f"{base_name}_{suffix}.mp3"
        try:
            ffmpeg = _ffmpeg_path or "ffmpeg"
            cmd = [ffmpeg, "-y", "-i", str(temp_wav), "-c:a", "libmp3lame", "-b:a", "192k", str(output_path)]
            subprocess.run(cmd, capture_output=True, check=True)
        except Exception as e:
            logger.error(f"MP3 encoding failed, keeping WAV: {e}")
            output_path = self.output_dir / f"{base_name}_{suffix}.wav"
            temp_wav.replace(output_path)
            return output_path
        finally:
            temp_wav.unlink(missing_ok=True)
        return output_path

    # ========== ANALYSIS ==========

    def _analyze_metrics(self, y: np.ndarray, sr: int, speech_ratio: Optional[float] = None) -> Dict:
        def _dbfs(v: float) -> float:
            return float(20.0 * np.log10(v)) if v > 1e-12 else -120.0

        n = len(y)
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        peak = float(np.max(np.abs(y))) if n else 0.0

        frame = max(1, int(sr * 0.02))
        rms_frames = np.array([
            np.sqrt(np.mean(y[i:i + frame] ** 2) + 1e-12)
            for i in range(0, n, frame)
        ]) if n else np.array([1e-6])

        noise_floor = float(np.percentile(rms_frames, 10))
        speech_level = float(np.percentile(rms_frames, 90))
        snr_est = _dbfs(speech_level) - _dbfs(noise_floor)

        clipped = int(np.sum(np.abs(y) > 0.98))
        clip_pct = round(100.0 * clipped / n, 4) if n else 0.0

        spec = np.abs(np.fft.rfft(y[: min(n, 1 << 18)] * np.hanning(min(n, 1 << 18)))) if n else np.array([1e-6])
        freqs = np.fft.rfftfreq(min(n, 1 << 18), 1.0 / sr) if n else np.array([0.0])
        low = (freqs >= 80) & (freqs < 1000)
        high = (freqs >= 1000) & (freqs < min(8000, sr / 2 - 1))
        e_low = float(np.mean(spec[low] ** 2) + 1e-20) if np.any(low) else 1e-20
        e_high = float(np.mean(spec[high] ** 2) + 1e-20) if np.any(high) else 1e-20
        tilt = float(10 * np.log10(e_high / e_low))

        metrics = {
            "duration_seconds": round(n / sr, 2) if sr else 0.0,
            "sample_rate": int(sr),
            "rms_dbfs": round(_dbfs(rms), 2),
            "peak_dbfs": round(_dbfs(peak), 2),
            "noise_floor_dbfs": round(_dbfs(noise_floor), 2),
            "estimated_snr_db": round(snr_est, 2),
            "clipping_pct": clip_pct,
            "spectral_tilt_db": round(tilt, 2),
        }
        if speech_ratio is not None:
            metrics["speech_active_pct"] = round(100.0 * float(speech_ratio), 2)
        return metrics

    # ========== VAD ==========

    def _compute_vad_mask(self, y: np.ndarray, sr: int) -> np.ndarray:
        if not self.config.get("no_silero"):
            try:
                mask = self._silero_vad_mask(y, sr)
                if mask is not None:
                    logger.debug("VAD: using Silero model")
                    return mask
            except Exception as e:
                logger.warning(f"Silero VAD failed, falling back to energy VAD: {e}")
        logger.debug("VAD: using energy/ZCR/flatness fallback")
        return self._energy_vad_mask(y, sr)

    def _silero_vad_mask(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        import torch

        device = "cuda" if self.deps.get("cuda") else "cpu"
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, verbose=False)
        get_speech_timestamps = utils[0]
        model.to(device)

        vad_sr = 16000
        y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=vad_sr) if sr != vad_sr else y.astype(np.float32)
        tensor = torch.from_numpy(y16).to(device)

        threshold = float(self.config.get("vad_threshold", 0.35))
        with torch.no_grad():
            timestamps = get_speech_timestamps(tensor, model, sampling_rate=vad_sr, threshold=threshold)

        mask = np.zeros(len(y), dtype=np.float32)
        margin = int(0.1 * sr)
        for ts in timestamps:
            start = max(0, int(ts["start"] * sr / vad_sr) - margin)
            end = min(len(y), int(ts["end"] * sr / vad_sr) + margin)
            mask[start:end] = 1.0
        return mask

    def _energy_vad_mask(self, y: np.ndarray, sr: int) -> np.ndarray:
        frame_len = int(sr * 0.03)
        hop = max(1, frame_len // 2)
        n_frames = max(1, (len(y) - frame_len) // hop + 1)

        rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame_len] ** 2) + 1e-10) for i in range(n_frames)])
        zcr = np.array([
            np.sum(np.abs(np.diff(np.sign(y[i * hop:i * hop + frame_len])))) / (2.0 * frame_len)
            for i in range(n_frames)
        ])

        flatness = np.zeros(n_frames)
        for i in range(n_frames):
            frame = y[i * hop:i * hop + frame_len]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))[1:]
            geo = np.exp(np.mean(np.log(spectrum + 1e-10)))
            arith = np.mean(spectrum) + 1e-10
            flatness[i] = geo / arith

        silence_rms = np.percentile(rms, 15)
        rms_threshold = silence_rms * 3.0
        score = np.zeros(n_frames)
        score[rms > rms_threshold] += 1.0
        score[zcr < 0.3] += 0.5
        score[flatness < 0.4] += 0.5
        raw = (score >= 1.0).astype(float)

        margin_frames = max(1, int(0.1 * sr / hop))
        expanded = raw.copy()
        for i in range(n_frames):
            if raw[i] > 0:
                expanded[max(0, i - margin_frames):min(n_frames, i + margin_frames + 1)] = 1.0
        expanded = median_filter(expanded, size=max(3, margin_frames // 2) | 1)

        sample_mask = np.interp(np.arange(len(y)), np.arange(n_frames) * hop + hop // 2, expanded)
        return np.clip(sample_mask, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _speech_ratio(vad_mask: np.ndarray) -> float:
        return float(np.mean(vad_mask)) if len(vad_mask) else 0.0

    # ========== STAGE 1: DEMUCS VOICE SEPARATION ==========

    def _stage_demucs(self, input_path: Path, work_dir: Path) -> Optional[Path]:
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            from demucs.audio import convert_audio
        except Exception as e:
            logger.warning(f"Demucs unavailable, skipping separation: {e}")
            return None

        device = "cuda" if self.deps.get("cuda") else "cpu"
        _tprint(f"    [demucs] Carregando modelo htdemucs ({device})...")
        try:
            model = get_model("htdemucs")
        except Exception as e:
            logger.error(f"Failed to load demucs model: {e}")
            return None
        model.to(device)

        ext = input_path.suffix.lower()
        if ext in {".wav", ".flac", ".ogg"}:
            data, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
        else:
            temp_wav = work_dir / "_demucs_input.wav"
            _ffmpeg_to_wav(input_path, temp_wav, model.samplerate, mono=False)
            data, sr = sf.read(str(temp_wav), dtype="float32", always_2d=True)
            temp_wav.unlink(missing_ok=True)

        waveform = torch.from_numpy(data.T)
        wav = convert_audio(waveform, sr, model.samplerate, model.audio_channels)

        vocals_idx = model.sources.index("vocals")
        chunk_samples = 120 * model.samplerate
        overlap_samples = int(5 * model.samplerate)
        total_samples = wav.shape[-1]
        n_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
        _tprint(f"    [demucs] Separando voz em {n_chunks} chunk(s) ({total_samples / model.samplerate:.0f}s)...")

        vocals_parts: List = []
        for i in range(n_chunks):
            start = max(0, i * chunk_samples - overlap_samples) if i > 0 else 0
            end = min(total_samples, (i + 1) * chunk_samples + overlap_samples)
            chunk = wav[:, start:end]

            ref = chunk.mean(0)
            chunk_scaled = (chunk - ref.mean()) / (ref.std() + 1e-8)

            if device == "cuda":
                torch.cuda.empty_cache()
            try:
                sources = apply_model(model, chunk_scaled[None], device=device, progress=False, split=True, overlap=0.25, segment=6)[0]
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logger.warning(f"Demucs OOM/runtime error on chunk {i + 1}, retrying on CPU: {e}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                    model.to("cpu")
                sources = apply_model(model, chunk_scaled[None], device="cpu", progress=False, split=True, overlap=0.25, segment=6)[0]

            sources = sources * ref.std() + ref.mean()
            vocal_chunk = sources[vocals_idx].cpu()
            del sources

            trim_start = overlap_samples if i > 0 else 0
            trim_end = vocal_chunk.shape[-1] - (overlap_samples if end < total_samples else 0)
            vocals_parts.append(vocal_chunk[:, trim_start:trim_end])
            del vocal_chunk

        vocals = torch.cat(vocals_parts, dim=-1)
        del vocals_parts
        vocals_path = work_dir / "_stage1_vocals.wav"
        sf.write(str(vocals_path), vocals.numpy().T, model.samplerate, subtype="PCM_16")
        del vocals
        _tprint(f"    [demucs] Voz isolada ({model.samplerate} Hz)")
        return vocals_path

    # ========== STAGE 2: AI SPEECH ENHANCEMENT ==========

    def _stage_enhance(self, input_wav: Path, work_dir: Path) -> Optional[Path]:
        if self.deps.get("deepfilter"):
            result = self._enhance_deepfilter(input_wav, work_dir)
            if result:
                return result
        if self.deps.get("clearvoice"):
            result = self._enhance_clearvoice(input_wav, work_dir)
            if result:
                return result
        logger.warning("No AI enhancer available, skipping enhancement stage")
        return None

    def _enhance_deepfilter(self, input_wav: Path, work_dir: Path) -> Optional[Path]:
        try:
            from df import init_df
            from df.enhance import enhance, load_audio, save_audio
        except Exception as e:
            logger.warning(f"DeepFilterNet unavailable: {e}")
            return None

        _tprint("    [deepfilter] Carregando modelo...")
        model, df_state, _ = init_df(config_allow_defaults=True)
        target_sr = df_state.sr()
        audio, _ = load_audio(str(input_wav), sr=target_sr)
        enhanced = enhance(model, df_state, audio)
        output_path = work_dir / "_stage2_enhanced.wav"
        save_audio(str(output_path), enhanced, sr=target_sr)
        _tprint("    [deepfilter] Fala realcada")
        return output_path

    def _enhance_clearvoice(self, input_wav: Path, work_dir: Path) -> Optional[Path]:
        try:
            from clearvoice import ClearVoice
        except Exception as e:
            logger.warning(f"ClearVoice unavailable: {e}")
            return None

        _tprint("    [clearvoice] Carregando modelo MossFormer2_SE_48K...")
        cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        out_sr = 48000

        try:
            info = sf.info(str(input_wav))
        except Exception as e:
            logger.error(f"clearvoice: failed to read audio info: {e}")
            return None

        in_sr = info.samplerate
        total_frames = info.frames
        overlap_seconds = 2
        chunk_frames = 120 * in_sr
        overlap_frames = overlap_seconds * in_sr
        n_chunks = max(1, (total_frames + chunk_frames - 1) // chunk_frames)
        _tprint(f"    [clearvoice] Realcando fala em {n_chunks} chunk(s)...")

        tmp_chunk = work_dir / "_cv_chunk_in.wav"
        enhanced_parts: List = []
        try:
            for i in range(n_chunks):
                start = max(0, i * chunk_frames - overlap_frames) if i > 0 else 0
                end = min(total_frames, (i + 1) * chunk_frames + overlap_frames)
                chunk_data, _ = sf.read(str(input_wav), start=start, stop=end, dtype="float32")
                sf.write(str(tmp_chunk), chunk_data, in_sr, subtype="PCM_16")
                del chunk_data

                enhanced_audio = cv(str(tmp_chunk))
                if enhanced_audio is None:
                    logger.error(f"clearvoice: no result on chunk {i + 1}/{n_chunks}")
                    return None
                enhanced_np = enhanced_audio if isinstance(enhanced_audio, np.ndarray) else enhanced_audio.cpu().numpy()
                if enhanced_np.ndim == 1:
                    enhanced_np = enhanced_np[np.newaxis, :]
                enhanced_np = enhanced_np.astype(np.float32)
                seg = enhanced_np if enhanced_np.shape[0] <= 2 else enhanced_np.T
                del enhanced_audio, enhanced_np

                trim_start = int(overlap_seconds * out_sr) if i > 0 else 0
                trim_end = seg.shape[-1] - (int(overlap_seconds * out_sr) if end < total_frames else 0)
                enhanced_parts.append(seg[:, trim_start:trim_end].copy())
                del seg
        finally:
            tmp_chunk.unlink(missing_ok=True)

        if not enhanced_parts:
            logger.error("clearvoice: no chunk processed")
            return None

        full = np.concatenate(enhanced_parts, axis=-1)
        del enhanced_parts
        audio_out = full.T
        del full
        peak = np.max(np.abs(audio_out))
        if peak > 1.0:
            audio_out = audio_out / peak
        output_path = work_dir / "_stage2_enhanced.wav"
        sf.write(str(output_path), audio_out, out_sr, subtype="FLOAT")
        _tprint("    [clearvoice] Fala realcada")
        return output_path

    # ========== STFT HELPER ==========

    def _process_stft_chunked(self, y: np.ndarray, sr: int, process_fn) -> np.ndarray:
        chunk_samples = int(STFT_CHUNK_SECONDS * sr)
        overlap_samples = int(2 * sr)

        if len(y) <= chunk_samples + overlap_samples:
            S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
            return librosa.istft(process_fn(S, sr), hop_length=HOP_LENGTH, length=len(y))

        result = np.zeros_like(y)
        weight = np.zeros_like(y)
        pos = 0
        while pos < len(y):
            end = min(pos + chunk_samples, len(y))
            chunk = y[pos:end]
            S = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
            chunk_out = librosa.istft(process_fn(S, sr), hop_length=HOP_LENGTH, length=len(chunk))

            fade_len = min(overlap_samples, len(chunk))
            w = np.ones(len(chunk))
            if pos > 0 and fade_len > 0:
                w[:fade_len] = np.linspace(0, 1, fade_len)
            if end < len(y) and fade_len > 0:
                w[-fade_len:] = np.linspace(1, 0, fade_len)
            result[pos:end] += chunk_out * w
            weight[pos:end] += w
            pos += chunk_samples - overlap_samples

        return result / np.where(weight > 0, weight, 1.0)

    # ========== DSP BUILDING BLOCKS ==========

    def _declip(self, y: np.ndarray) -> np.ndarray:
        clip_threshold = 0.98
        clipped = np.abs(y) > clip_threshold
        if not np.any(clipped):
            return y
        result = y.copy()
        clip_indices = np.where(clipped)[0]
        regions = []
        start = clip_indices[0]
        for i in range(1, len(clip_indices)):
            if clip_indices[i] - clip_indices[i - 1] > 1:
                regions.append((start, clip_indices[i - 1]))
                start = clip_indices[i]
        regions.append((start, clip_indices[-1]))
        for s, e in regions:
            pad = 10
            i_start = max(0, s - pad)
            i_end = min(len(y), e + pad + 1)
            x_good = [idx for idx in range(i_start, i_end) if not clipped[idx]]
            y_good = [y[idx] for idx in x_good]
            if len(x_good) >= 2:
                result[s:e + 1] = np.interp(range(s, e + 1), x_good, y_good)
        return result

    def _dc_remove(self, y: np.ndarray) -> np.ndarray:
        return (y - np.mean(y)).astype(np.float32) if len(y) else y

    def _highpass(self, y: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
        return sosfilt(sos, y).astype(np.float32)

    def _mild_wiener(self, y: np.ndarray, sr: int) -> np.ndarray:
        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            noise_frames = min(30, magnitude.shape[1])
            noise_est = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            snr = (magnitude ** 2) / (noise_est ** 2 + 1e-10)
            gain = np.clip(np.maximum(1.0 - (1.0 / (snr + 1e-10)), 0.0), 0.1, 1.0)
            return magnitude * gain * np.exp(1j * phase)

        return self._process_stft_chunked(y, sr, _process)

    def _whisper_spectral_boost_vad(self, y: np.ndarray, sr: int, vad_mask: np.ndarray, max_boost: float = 8.0) -> np.ndarray:
        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
            n_stft = magnitude.shape[1]
            centers = librosa.frames_to_samples(np.arange(n_stft), hop_length=HOP_LENGTH)
            frame_vad = np.array([
                np.mean(vad_mask[max(0, c - HOP_LENGTH // 2):min(len(vad_mask), c + HOP_LENGTH // 2)]) if len(vad_mask) else 0.0
                for c in centers
            ])
            speech_mask = (freqs >= 150) & (freqs <= 6000)
            speech_energy = np.mean(magnitude[speech_mask, :], axis=0)
            valid = speech_energy > 0
            median_energy = np.median(speech_energy[valid]) if np.any(valid) else 1e-10
            frame_gain = np.ones(n_stft)
            boost_frames = (speech_energy < median_energy * 0.3) & (frame_vad > 0.3)
            if np.any(boost_frames) and median_energy > 0:
                frame_gain[boost_frames] = np.clip(median_energy / (speech_energy[boost_frames] + 1e-10), 1.0, max_boost)
            freq_gain = np.ones_like(freqs)
            freq_gain[speech_mask] = 2.0
            freq_gain[~speech_mask] = 0.1
            return magnitude * freq_gain[:, np.newaxis] * frame_gain[np.newaxis, :] * np.exp(1j * phase)

        return self._peak_norm(self._process_stft_chunked(y, sr, _process))

    def _boost_quiet_segments_vad(self, y: np.ndarray, sr: int, vad_mask: np.ndarray, max_gain: float = 6.0) -> np.ndarray:
        frame_len = int(sr * 0.05)
        hop = max(1, frame_len // 2)
        n_frames = max(1, (len(y) - frame_len) // hop + 1)
        rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame_len] ** 2) + 1e-10) for i in range(n_frames)])
        frame_vad = np.array([
            np.mean(vad_mask[i * hop:min(len(vad_mask), i * hop + frame_len)]) if len(vad_mask) else 0.0
            for i in range(n_frames)
        ])
        silence_thresh = np.percentile(rms, 10)
        speech_rms = rms[rms > silence_thresh * 2]
        if len(speech_rms) == 0:
            return y
        target = np.percentile(speech_rms, 60)
        frame_gain = np.ones(n_frames)
        active = (rms > silence_thresh * 1.5) & (frame_vad > 0.3)
        frame_gain[active] = np.clip(target / (rms[active] + 1e-10), 1.0, max_gain)
        frame_gain = uniform_filter1d(frame_gain.astype(np.float64), size=7)
        sample_gain = np.interp(np.arange(len(y)), np.arange(n_frames) * hop + hop // 2, frame_gain)
        return (y * sample_gain).astype(np.float32)

    def _formant_boost(self, y: np.ndarray, sr: int) -> np.ndarray:
        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
            gain = np.ones_like(freqs)
            gain[(freqs >= 300) & (freqs <= 900)] = 1.35
            gain[(freqs >= 900) & (freqs <= 2500)] = 1.5
            gain[(freqs >= 2500) & (freqs <= 3500)] = 1.25
            return magnitude * gain[:, np.newaxis] * np.exp(1j * phase)

        return self._peak_norm(self._process_stft_chunked(y, sr, _process))

    def _dynamic_compress(self, y: np.ndarray, sr: int, threshold_db: float = -20.0, ratio: float = 3.0) -> np.ndarray:
        frame_len = int(sr * 0.03)
        hop = max(1, frame_len // 2)
        n_frames = max(1, (len(y) - frame_len) // hop + 1)
        rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame_len] ** 2) + 1e-10) for i in range(n_frames)])
        peak_env = np.max(rms) if len(rms) else 1.0
        rms_norm = rms / peak_env if peak_env > 0 else rms
        threshold = 10 ** (threshold_db / 20)
        frame_gain = np.ones(n_frames)
        above = rms_norm > threshold
        if np.any(above):
            excess = rms_norm[above] / threshold
            frame_gain[above] = (threshold * (excess ** (1.0 / ratio))) / (rms_norm[above] + 1e-10)
        frame_gain = uniform_filter1d(frame_gain.astype(np.float64), size=7)
        sample_gain = np.interp(np.arange(len(y)), np.arange(n_frames) * hop + hop // 2, frame_gain)
        return self._peak_norm((y * sample_gain).astype(np.float32))

    def _loudness_norm(self, y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        if self.deps.get("pyloudnorm"):
            try:
                import pyloudnorm as pyln

                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(y.astype(np.float64))
                if np.isfinite(loudness):
                    normalized = pyln.normalize.loudness(y.astype(np.float64), loudness, target_lufs)
                    logger.debug(f"Loudness norm (LUFS): {loudness:.1f} -> {target_lufs:.1f}")
                    return normalized.astype(np.float32)
            except Exception as e:
                logger.warning(f"pyloudnorm failed, using RMS fallback: {e}")
        target_rms = 10 ** ((target_lufs + 8.0) / 20.0)
        current_rms = np.sqrt(np.mean(y ** 2) + 1e-10)
        if current_rms > 0:
            gain = min(target_rms / current_rms, 40.0)
            return (y * gain).astype(np.float32)
        return y

    def _peak_limit(self, y: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
        peak = np.max(np.abs(y)) if len(y) else 0.0
        if peak > ceiling:
            y = np.tanh(y / peak * 1.5) * ceiling
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def _peak_norm(self, y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y)) if len(y) else 0.0
        if peak > 0:
            return (y * (0.99 / peak)).astype(np.float32)
        return y

    # ========== VARIANT BUILDERS ==========

    def _build_variant(self, name: str, base: np.ndarray, sr: int, vad_mask: np.ndarray) -> np.ndarray:
        max_gain = float(self.config.get("max_gain", 10.0))
        target_lufs = float(self.config.get("target_lufs", -16.0))
        y = base.copy()

        if name == "clean":
            y = self._dynamic_compress(y, sr, threshold_db=-20.0, ratio=3.0)
            y = self._loudness_norm(y, sr, target_lufs)
            return self._peak_limit(y)

        if name == "speech_boost":
            y = self._mild_wiener(y, sr)
            y = self._whisper_spectral_boost_vad(y, sr, vad_mask, max_boost=8.0)
            y = self._boost_quiet_segments_vad(y, sr, vad_mask, max_gain=max_gain)
            y = self._formant_boost(y, sr)
            y = self._dynamic_compress(y, sr, threshold_db=-20.0, ratio=3.5)
            y = self._loudness_norm(y, sr, target_lufs)
            return self._peak_limit(y)

        if name == "max":
            y = self._mild_wiener(y, sr)
            y = self._whisper_spectral_boost_vad(y, sr, vad_mask, max_boost=12.0)
            y = self._boost_quiet_segments_vad(y, sr, vad_mask, max_gain=max_gain * 1.5)
            y = self._formant_boost(y, sr)
            y = self._dynamic_compress(y, sr, threshold_db=-22.0, ratio=4.0)
            y = self._loudness_norm(y, sr, min(target_lufs + 2.0, -9.0))
            return self._peak_limit(y)

        raise ValueError(f"Unknown variant: {name}")

    @staticmethod
    def _resize_mask(mask: np.ndarray, new_len: int) -> np.ndarray:
        if len(mask) == new_len or len(mask) == 0:
            return mask if len(mask) == new_len else np.ones(new_len, dtype=np.float32)
        return np.interp(np.linspace(0, len(mask) - 1, new_len), np.arange(len(mask)), mask).astype(np.float32)

    # ========== ORCHESTRATION ==========

    def _ext(self) -> str:
        return "wav" if self.config.get("wav") else "mp3"

    def _expected_outputs(self, audio_path: Path) -> List[str]:
        return [f"{audio_path.stem}_v2_{v}.{self._ext()}" for v in self.config.get("variants", DEFAULT_VARIANTS)]

    def _is_already_processed(self, audio_path: Path) -> bool:
        for fname in self._expected_outputs(audio_path):
            out = self.output_dir / fname
            if not out.exists() or out.stat().st_size == 0:
                return False
        return True

    def _get_audio_files(self) -> List[Path]:
        if not self.input_path.is_dir():
            return []
        files: List[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_path.glob(f"*{ext}"))
            files.extend(self.input_path.glob(f"*{ext.upper()}"))
        return sorted(set(files), key=lambda p: p.name.lower())

    def _prepare_base(self, audio_path: Path, work_dir: Path) -> Tuple[np.ndarray, int]:
        current_wav: Optional[Path] = None

        if self.deps.get("demucs") and not self.config.get("no_demucs"):
            _tprint("    Etapa 1: Separacao de voz (Demucs)")
            try:
                vocals = self._stage_demucs(audio_path, work_dir)
                if vocals and vocals.exists():
                    current_wav = vocals
            except Exception as e:
                logger.error(f"Demucs stage failed: {e}", exc_info=True)
        else:
            _tprint("    Etapa 1: Demucs desativado/indisponivel, pulando")

        if current_wav is None:
            current_wav = work_dir / "_raw.wav"
            if audio_path.suffix.lower() == ".wav":
                import shutil

                shutil.copyfile(audio_path, current_wav)
            else:
                _ffmpeg_to_wav(audio_path, current_wav, self.sample_rate, mono=True)

        has_enhancer = (self.deps.get("deepfilter") or self.deps.get("clearvoice")) and not self.config.get("no_ai_enhance")
        if has_enhancer:
            _tprint("    Etapa 2: Realce de fala (IA)")
            try:
                enhanced = self._stage_enhance(current_wav, work_dir)
                if enhanced and enhanced.exists():
                    current_wav = enhanced
            except Exception as e:
                logger.error(f"Enhancement stage failed: {e}", exc_info=True)
        else:
            _tprint("    Etapa 2: Realce IA desativado/indisponivel, pulando")

        base, sr = _load_wav_mono(current_wav, self.sample_rate)
        base = self._declip(base)
        base = self._dc_remove(base)
        base = self._highpass(base, sr, int(self.config.get("hp_hz", 70)))
        return base, sr

    def process_single_audio(self, audio_path: Path) -> Dict:
        logger.debug(f"Loading audio: {audio_path.name}")
        y, sr = self._load_audio(audio_path)
        base_name = audio_path.stem
        duration = len(y) / sr if sr else 0.0
        _tprint(f"\n  Audio carregado: {base_name} ({duration:.1f}s / {duration / 60:.1f}min)")

        work_dir = self.output_dir / f"_work_{base_name}"
        work_dir.mkdir(parents=True, exist_ok=True)

        generated: List[str] = []
        try:
            base, base_sr = self._prepare_base(audio_path, work_dir)

            base_mask = self._compute_vad_mask(base, base_sr)
            speech_ratio = self._speech_ratio(base_mask)
            input_metrics = self._analyze_metrics(y, sr, speech_ratio)
            logger.debug(f"Input metrics: {input_metrics}")
            _tprint(f"  Voz detectada (pos-isolamento): {input_metrics.get('speech_active_pct', 0)}% | RMS {input_metrics['rms_dbfs']}dB | SNR ~{input_metrics['estimated_snr_db']}dB")

            variants = self.config.get("variants", DEFAULT_VARIANTS)
            _tprint(f"    Etapa 3-4: Gerando {len(variants)} variante(s): {', '.join(variants)}")

            workers = max(1, int(self.config.get("workers", 1)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(self._run_variant, name, base, base_sr, base_mask, base_name): name
                    for name in variants
                }
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        path = future.result()
                        if path:
                            generated.append(path.name)
                    except Exception as e:
                        logger.error(f"Variant {name} failed: {e}", exc_info=True)
        finally:
            self._cleanup_work_dir(work_dir)

        result = {
            "source": str(audio_path),
            "input_metrics": input_metrics,
            "variants": list(self.config.get("variants", DEFAULT_VARIANTS)),
            "files_generated": generated,
        }
        report_path = self.output_dir / f"{base_name}_v2_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write report: {e}")

        if self.config.get("validate") and generated:
            self._run_validation(audio_path, generated)

        return result

    def _run_variant(self, name: str, base: np.ndarray, sr: int, mask: np.ndarray, base_name: str) -> Optional[Path]:
        t0 = time.time()
        _tprint(f"    [{name}] Iniciando")
        try:
            processed = self._build_variant(name, base, sr, mask)
            if processed is None or len(processed) == 0:
                _tprint(f"    [{name}] Sem resultado")
                return None
            path = self._save_audio(processed, base_name, f"v2_{name}", sr)
            size_mb = path.stat().st_size / (1024 * 1024)
            _tprint(f"    [{name}] Concluido em {time.time() - t0:.0f}s -> {path.name} ({size_mb:.1f}MB)")
            return path
        except Exception as e:
            _tprint(f"    [{name}] ERRO: {e}")
            logger.error(f"Variant {name} error: {e}", exc_info=True)
            return None

    def _cleanup_work_dir(self, work_dir: Path) -> None:
        try:
            for f in work_dir.iterdir():
                f.unlink(missing_ok=True)
            work_dir.rmdir()
        except Exception as e:
            logger.debug(f"Work dir cleanup skipped: {e}")

    def _run_validation(self, audio_path: Path, generated: List[str]) -> None:
        try:
            import forensic_audio_validate as fav

            outputs = [self.output_dir / g for g in generated]
            _tprint("\n    Etapa 5: Validacao (entrada vs saida)")
            fav.validate_files(
                input_path=audio_path,
                output_paths=outputs,
                use_whisper=bool(self.config.get("validate_whisper")),
                report_path=self.output_dir / f"{audio_path.stem}_v2_validation.json",
            )
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)

    def process_all_audio(self) -> Dict:
        if self.input_path.is_file():
            audio_files = [self.input_path]
        else:
            audio_files = self._get_audio_files()

        if not audio_files:
            logger.warning("No audio files found")
            _tprint("  Nenhum arquivo de audio encontrado.")
            return {}

        total = len(audio_files)
        skipped = 0
        results: Dict = {}
        for idx, audio_path in enumerate(audio_files, 1):
            if self._is_already_processed(audio_path):
                _tprint(f"  [{idx}/{total}] {audio_path.name} -- ja processado, pulando")
                skipped += 1
                continue
            _tprint(f"\n{'-' * 60}\n  [{idx}/{total}] {audio_path.name}\n{'-' * 60}")
            try:
                results[audio_path.name] = self.process_single_audio(audio_path)
            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}", exc_info=True)
                results[audio_path.name] = {"error": str(e)}

        if skipped:
            _tprint(f"\n  Pulados (ja processados): {skipped}/{total}")
        return results


def _parse_variants(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_VARIANTS)
    requested = [v.strip() for v in value.split(",") if v.strip()]
    valid = [v for v in requested if v in DEFAULT_VARIANTS]
    invalid = [v for v in requested if v not in DEFAULT_VARIANTS]
    if invalid:
        logger.warning(f"Ignoring unknown variants: {invalid}; valid options: {DEFAULT_VARIANTS}")
    return valid or list(DEFAULT_VARIANTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic Audio Processor v2 (IA + DSP)")
    parser.add_argument("--input", "-i", type=str, help="Arquivo ou pasta de audio")
    parser.add_argument("--output", "-o", type=str, help="Diretorio de saida")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Threads para variantes (1 recomendado)")
    parser.add_argument("--variants", type=str, default=None, help=f"Variantes separadas por virgula {DEFAULT_VARIANTS}")
    parser.add_argument("--target-lufs", type=float, default=-16.0, help="Loudness alvo (LUFS)")
    parser.add_argument("--max-gain", type=float, default=10.0, help="Ganho maximo para realce de trechos baixos")
    parser.add_argument("--hp-hz", type=int, default=70, help="High-pass (Hz)")
    parser.add_argument("--wav", action="store_true", help="Salvar saida em WAV (lossless) em vez de MP3")
    parser.add_argument("--no-demucs", action="store_true", help="Desativar separacao Demucs")
    parser.add_argument("--no-ai-enhance", action="store_true", help="Desativar realce de fala por IA")
    parser.add_argument("--no-silero", action="store_true", help="Desativar Silero VAD (usar VAD por energia)")
    parser.add_argument("--vad-threshold", type=float, default=0.35, help="Sensibilidade do Silero VAD (menor = detecta mais fala fraca)")
    parser.add_argument("--validate", action="store_true", help="Validar entrada vs saida apos processar")
    parser.add_argument("--validate-whisper", action="store_true", help="Incluir intelig. via Whisper na validacao")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = args.input or str(base_dir / "to_analyse")
    output_dir = args.output or str(base_dir / "audio_processed")

    inp = Path(input_path)
    if not inp.exists():
        print(f"  Caminho nao encontrado: {input_path}")
        return 1

    print("\n" + "=" * 60)
    print("  Forensic Audio Processor v2")
    print("=" * 60)

    deps = detect_dependencies()
    print_dependency_status(deps)

    config = {
        "workers": args.workers,
        "variants": _parse_variants(args.variants),
        "target_lufs": args.target_lufs,
        "max_gain": args.max_gain,
        "hp_hz": args.hp_hz,
        "wav": args.wav,
        "no_demucs": args.no_demucs,
        "no_ai_enhance": args.no_ai_enhance,
        "no_silero": args.no_silero,
        "vad_threshold": args.vad_threshold,
        "validate": args.validate,
        "validate_whisper": args.validate_whisper,
    }

    print(f"\n  Entrada: {input_path}")
    print(f"  Saida: {output_dir}")
    print(f"  Variantes: {', '.join(config['variants'])}\n")

    t_global = time.time()
    try:
        processor = ForensicAudioProcessorV2(input_path, output_dir, deps, config)
        results = processor.process_all_audio()
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n  ERRO: {e}")
        return 1

    elapsed = time.time() - t_global
    logger.debug("Processing complete")
    print(f"\n  Processamento concluido em {elapsed:.0f}s!")
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: ERRO - {result['error']}")
        else:
            print(f"  {name}: {len(result.get('files_generated', []))} arquivo(s)")
            for f in result.get("files_generated", []):
                print(f"    -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
