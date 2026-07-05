#!/usr/bin/env python3
"""
Forensic Audio Processor v3

Combina o melhor do v1 e v2:
- Nucleo de realce = pipeline whisper_vad do v1 (declip, highpass, noisereduce
  duplo, boost espectral de sussurros com VAD por energia) - o que gera os
  melhores resultados.
- Estagio final de loudness corrigido: AGC por nivel de fala, normalizacao
  baseada no RMS da fala ativa e limitador soft (tanh) em vez de hard clip +
  peak norm, para saida consistentemente alta sem distorcao.
- Sem Demucs, sem realce por IA, sem Silero (estagios que cortavam demais ou
  geravam audio vazio no v2).
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time
import json
import threading

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfilt
from scipy.ndimage import median_filter, uniform_filter1d

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("pydub").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

_ffmpeg_path = None
try:
    import imageio_ffmpeg
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    logger.debug(f"Using ffmpeg from imageio-ffmpeg: {_ffmpeg_path}")
except ImportError:
    for candidate in ["ffmpeg", "ffmpeg.exe"]:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            _ffmpeg_path = candidate
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

MAX_WORKERS = 2
TARGET_SR = 44100
STFT_CHUNK_SECONDS = 300
DEFAULT_TARGET_DB = -12.0
DEFAULT_MAX_GAIN = 12.0
_print_lock = threading.Lock()


def _tprint(msg: str):
    with _print_lock:
        print(msg, flush=True)


class ForensicAudioProcessorV3:

    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.opus'}

    def __init__(self, input_path: str, output_dir: str, sample_rate: int = TARGET_SR,
                 target_db: float = DEFAULT_TARGET_DB, max_gain: float = DEFAULT_MAX_GAIN,
                 save_wav: bool = False):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.target_db = float(target_db)
        self.max_gain = float(max_gain)
        self.save_wav = bool(save_wav)

        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized v3 processor - Input: {input_path}, Output: {output_dir}, "
                     f"target_db: {self.target_db}, max_gain: {self.max_gain}, wav: {self.save_wav}")

    # ========== I/O ==========

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        ext = audio_path.suffix.lower()

        if ext in {'.m4a', '.aac', '.wma', '.opus'}:
            logger.debug(f"Converting {ext} to wav via ffmpeg: {audio_path.name}")
            temp_wav = self.output_dir / f"_temp_{audio_path.stem}.wav"
            try:
                ffmpeg = _ffmpeg_path or "ffmpeg"
                cmd = [
                    ffmpeg, "-y", "-i", str(audio_path),
                    "-ar", str(self.sample_rate), "-ac", "1",
                    "-sample_fmt", "s16",
                    str(temp_wav)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")

                y, sr = librosa.load(str(temp_wav), sr=self.sample_rate, mono=True)
                return y, sr
            except Exception as e:
                logger.error(f"Conversion failed for {audio_path.name}: {e}")
                raise
            finally:
                if temp_wav.exists():
                    temp_wav.unlink()

        y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        return y, sr

    def _save_audio(self, audio: np.ndarray, base_name: str, suffix: str, sr: Optional[int] = None) -> Path:
        sr = sr or self.sample_rate
        audio = np.clip(audio, -1.0, 1.0)
        temp_wav = self.output_dir / f"_tmp_{base_name}_{suffix}.wav"
        sf.write(str(temp_wav), audio, sr, subtype='PCM_16')

        if self.save_wav:
            output_path = self.output_dir / f"{base_name}_{suffix}.wav"
            temp_wav.replace(output_path)
            return output_path

        output_path = self.output_dir / f"{base_name}_{suffix}.mp3"
        try:
            ffmpeg = _ffmpeg_path or "ffmpeg"
            cmd = [ffmpeg, "-y", "-i", str(temp_wav), "-c:a", "libmp3lame", "-b:a", "192k", str(output_path)]
            subprocess.run(cmd, capture_output=True, check=True)
        except Exception as e:
            logger.error(f"MP3 encoding failed, keeping WAV: {e}")
            output_path = self.output_dir / f"{base_name}_{suffix}.wav"
            temp_wav.rename(output_path)
            return output_path
        finally:
            if temp_wav.exists():
                temp_wav.unlink()
        return output_path

    # ========== MAIN PROCESSING ==========

    def _ext(self) -> str:
        return "wav" if self.save_wav else "mp3"

    def _get_expected_outputs(self, audio_path: Path) -> List[str]:
        pipelines = self._get_active_pipelines()
        return [f"{audio_path.stem}_{suffix}.{self._ext()}" for suffix, _, _ in pipelines]

    def _is_already_processed(self, audio_path: Path) -> bool:
        expected = self._get_expected_outputs(audio_path)
        for fname in expected:
            out = self.output_dir / fname
            if not out.exists() or out.stat().st_size == 0:
                return False
        return True

    def process_all_audio(self, max_workers: int = MAX_WORKERS) -> Dict:
        if self.input_path.is_file():
            audio_files = [self.input_path]
        else:
            audio_files = self._get_audio_files()

        if not audio_files:
            logger.warning("No audio files found")
            return {}

        total = len(audio_files)
        skipped = 0
        results = {}

        for idx, audio_path in enumerate(audio_files, 1):
            if self._is_already_processed(audio_path):
                _tprint(f"  [{idx}/{total}] {audio_path.name} — já processado, pulando")
                skipped += 1
                continue

            _tprint(f"\n  [{idx}/{total}] {audio_path.name}")
            try:
                result = self.process_single_audio(audio_path, max_workers)
                results[audio_path.name] = result
            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                results[audio_path.name] = {"error": str(e)}

        if skipped > 0:
            _tprint(f"\n  Pulados (já processados): {skipped}/{total}")

        return results

    def _get_audio_files(self) -> List[Path]:
        if not self.input_path.is_dir():
            return []
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_path.glob(f"*{ext}"))
            files.extend(self.input_path.glob(f"*{ext.upper()}"))
        return sorted(set(files))

    def _get_active_pipelines(self):
        return [
            ("v3", "Sussurros com VAD + loudness consistente", self._pipeline_whisper_vad_v3),
        ]

    def process_single_audio(self, audio_path: Path, max_workers: int = MAX_WORKERS) -> Dict:
        logger.debug(f"Loading audio: {audio_path.name}")
        y, sr = self._load_audio(audio_path)
        base_name = audio_path.stem

        duration = len(y) / sr
        logger.debug(f"Audio loaded: {len(y)} samples, {sr}Hz, duration: {duration:.1f}s")
        _tprint(f"\n  Áudio carregado: {base_name} ({duration:.1f}s / {duration/60:.1f}min)")

        pipelines = self._get_active_pipelines()

        generated_files = []
        total = len(pipelines)

        _tprint(f"  Executando {total} pipelines em paralelo ({max_workers} threads)...\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for suffix, label, pipeline_fn in pipelines:
                future = executor.submit(self._run_pipeline, y, sr, base_name, suffix, label, pipeline_fn)
                future_map[future] = (suffix, label)

            for future in as_completed(future_map):
                suffix, label = future_map[future]
                try:
                    path = future.result()
                    if path:
                        generated_files.append(str(path.name))
                except Exception as e:
                    logger.error(f"Pipeline {suffix} failed: {e}")

        result = {
            "source": str(audio_path),
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "pipelines": total,
            "target_db": self.target_db,
            "max_gain": self.max_gain,
            "files_generated": generated_files,
        }

        report_path = self.output_dir / f"{base_name}_v3_processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    def _run_pipeline(self, y: np.ndarray, sr: int, base_name: str, suffix: str, label: str, pipeline_fn) -> Optional[Path]:
        t_start = time.time()
        _tprint(f"    [{suffix}] Iniciando: {label}")
        try:
            processed = pipeline_fn(y.copy(), sr)
            if processed is None or len(processed) == 0:
                _tprint(f"    [{suffix}] Sem resultado")
                return None
            path = self._save_audio(processed, base_name, suffix, sr)
            elapsed = time.time() - t_start
            size_mb = path.stat().st_size / (1024 * 1024)
            _tprint(f"    [{suffix}] Concluído em {elapsed:.0f}s -> {path.name} ({size_mb:.1f}MB)")
            return path
        except Exception as e:
            elapsed = time.time() - t_start
            _tprint(f"    [{suffix}] ERRO após {elapsed:.0f}s: {e}")
            logger.error(f"Pipeline {suffix} error: {e}", exc_info=True)
            return None

    # ========== V3 PIPELINE ==========

    def _pipeline_whisper_vad_v3(self, y: np.ndarray, sr: int) -> np.ndarray:
        """v1 whisper_vad core + new loudness stage (speech AGC, active-speech norm, soft limiter)."""
        vad_mask = self._compute_vad_mask(y, sr)

        y = self._declip(y)
        y = self._highpass(y, sr, 80)
        y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.90, thresh_n_mult_nonstationary=1.5, sigmoid_slope_nonstationary=15, n_fft=2048)
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.7, n_std_thresh_stationary=1.2, n_fft=2048)
        y = self._whisper_spectral_boost_vad(y, sr, vad_mask)
        y = self._boost_quiet_segments_vad(y, sr, vad_mask, max_gain=10.0)

        y = self._dynamic_compress(y, sr)
        y = self._speech_agc(y, sr, vad_mask, target_db=-18.0, max_gain=self.max_gain)
        y = self._speech_loudness_norm(y, sr, vad_mask, target_db=self.target_db)
        y = self._peak_limit(y)
        return y

    def _compute_vad_mask(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute voice activity mask on the ORIGINAL signal before any processing."""
        frame_len = int(sr * 0.03)
        hop = frame_len // 2
        n_frames = max(1, (len(y) - frame_len) // hop + 1)

        rms = np.array([
            np.sqrt(np.mean(y[i*hop:i*hop+frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])

        zcr = np.array([
            np.sum(np.abs(np.diff(np.sign(y[i*hop:i*hop+frame_len])))) / (2.0 * frame_len)
            for i in range(n_frames)
        ])

        spectral_flatness = np.zeros(n_frames)
        for i in range(n_frames):
            frame = y[i*hop:i*hop+frame_len]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            spectrum = spectrum[1:]
            geo_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
            arith_mean = np.mean(spectrum) + 1e-10
            spectral_flatness[i] = geo_mean / arith_mean

        silence_rms = np.percentile(rms, 15)
        rms_threshold = silence_rms * 3.0

        voice_score = np.zeros(n_frames)
        voice_score[rms > rms_threshold] += 1.0
        voice_score[zcr < 0.3] += 0.5
        voice_score[spectral_flatness < 0.4] += 0.5

        raw_mask = (voice_score >= 1.0).astype(float)

        margin_frames = int(0.1 * sr / hop)
        expanded_mask = np.copy(raw_mask)
        for i in range(n_frames):
            if raw_mask[i] > 0:
                start = max(0, i - margin_frames)
                end = min(n_frames, i + margin_frames + 1)
                expanded_mask[start:end] = 1.0

        kernel_size = max(3, margin_frames // 2) | 1
        expanded_mask = median_filter(expanded_mask, size=kernel_size)

        sample_mask = np.interp(
            np.arange(len(y)),
            np.arange(n_frames) * hop + hop // 2,
            expanded_mask
        )
        return np.clip(sample_mask, 0.0, 1.0)

    def _whisper_spectral_boost_vad(self, y: np.ndarray, sr: int, vad_mask: np.ndarray) -> np.ndarray:
        """Spectral boost for whispers, gated by VAD mask."""
        n_fft = 2048
        hop_length = 512

        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            n_stft_frames = magnitude.shape[1]
            frame_centers = librosa.frames_to_samples(np.arange(n_stft_frames), hop_length=hop_length)
            frame_vad = np.array([
                np.mean(vad_mask[max(0, c - hop_length//2):min(len(vad_mask), c + hop_length//2)])
                for c in frame_centers
            ])

            speech_mask = (freqs >= 150) & (freqs <= 6000)
            speech_energy = np.mean(magnitude[speech_mask, :], axis=0)
            valid = speech_energy > 0
            median_energy = np.median(speech_energy[valid]) if np.any(valid) else 1e-10

            frame_gain = np.ones(n_stft_frames)
            quiet = speech_energy < (median_energy * 0.3)
            has_voice = frame_vad > 0.3
            boost_frames = quiet & has_voice
            if np.any(boost_frames) and median_energy > 0:
                frame_gain[boost_frames] = np.clip(median_energy / (speech_energy[boost_frames] + 1e-10), 1.0, 8.0)

            freq_gain = np.ones_like(freqs)
            freq_gain[speech_mask] = 2.5
            freq_gain[~speech_mask] = 0.05

            return magnitude * freq_gain[:, np.newaxis] * frame_gain[np.newaxis, :] * np.exp(1j * phase)

        return self._peak_norm(self._process_stft_chunked(y, sr, _process))

    def _boost_quiet_segments_vad(self, y: np.ndarray, sr: int, vad_mask: np.ndarray, max_gain: float = 6.0) -> np.ndarray:
        """Boost quiet segments only where VAD detected voice activity, with smoothed gain."""
        frame_len = int(sr * 0.05)
        hop = frame_len // 2
        n_frames = max(1, (len(y) - frame_len) // hop + 1)

        rms = np.array([
            np.sqrt(np.mean(y[i*hop:i*hop+frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])

        frame_vad = np.array([
            np.mean(vad_mask[i*hop:min(len(vad_mask), i*hop+frame_len)])
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

        sample_gain = np.interp(
            np.arange(len(y)),
            np.arange(n_frames) * hop + hop // 2,
            frame_gain
        )

        return y * sample_gain

    # ========== CHUNKED STFT HELPER ==========

    def _process_stft_chunked(self, y: np.ndarray, sr: int, process_fn) -> np.ndarray:
        """Process audio through STFT in chunks to avoid memory errors on long files."""
        chunk_samples = int(STFT_CHUNK_SECONDS * sr)
        overlap_samples = int(2 * sr)
        n_fft = 2048
        hop_length = 512

        if len(y) <= chunk_samples + overlap_samples:
            S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S_processed = process_fn(S, sr)
            return librosa.istft(S_processed, hop_length=hop_length, length=len(y))

        result = np.zeros_like(y)
        weight = np.zeros_like(y)
        pos = 0

        while pos < len(y):
            end = min(pos + chunk_samples, len(y))
            chunk = y[pos:end]

            S = librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length)
            S_processed = process_fn(S, sr)
            chunk_out = librosa.istft(S_processed, hop_length=hop_length, length=len(chunk))

            fade_len = min(overlap_samples, len(chunk))
            w = np.ones(len(chunk))
            if pos > 0 and fade_len > 0:
                w[:fade_len] = np.linspace(0, 1, fade_len)
            if end < len(y) and fade_len > 0:
                w[-fade_len:] = np.linspace(1, 0, fade_len)

            result[pos:end] += chunk_out * w
            weight[pos:end] += w

            pos += chunk_samples - overlap_samples

        safe_weight = np.where(weight > 0, weight, 1.0)
        return result / safe_weight

    # ========== BUILDING BLOCK TECHNIQUES ==========

    def _declip(self, y: np.ndarray) -> np.ndarray:
        clip_threshold = 0.98
        clipped = np.abs(y) > clip_threshold
        if not np.any(clipped):
            return y

        result = y.copy()
        clip_indices = np.where(clipped)[0]
        if len(clip_indices) == 0:
            return y

        regions = []
        start = clip_indices[0]
        for i in range(1, len(clip_indices)):
            if clip_indices[i] - clip_indices[i-1] > 1:
                regions.append((start, clip_indices[i-1]))
                start = clip_indices[i]
        regions.append((start, clip_indices[-1]))

        for s, e in regions:
            pad = 10
            i_start = max(0, s - pad)
            i_end = min(len(y), e + pad + 1)
            x_good = [idx for idx in range(i_start, i_end) if not clipped[idx]]
            y_good = [y[idx] for idx in x_good]
            if len(x_good) >= 2:
                result[s:e+1] = np.interp(range(s, e + 1), x_good, y_good)

        return result

    def _highpass(self, y: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        sos = butter(4, cutoff, btype='highpass', fs=sr, output='sos')
        return sosfilt(sos, y)

    def _dynamic_compress(self, y: np.ndarray, sr: int) -> np.ndarray:
        threshold_db = -18
        ratio = 4.0

        frame_len = int(sr * 0.02)
        hop = frame_len // 2
        n_frames = max(1, (len(y) - frame_len) // hop + 1)

        rms = np.array([
            np.sqrt(np.mean(y[i*hop:i*hop+frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])

        peak_env = np.max(rms) if len(rms) > 0 else 1.0
        if peak_env > 0:
            rms = rms / peak_env

        threshold = 10 ** (threshold_db / 20)
        frame_gain = np.where(
            rms > threshold,
            threshold * (rms / threshold) ** (1.0 / ratio) / (rms + 1e-10),
            1.0
        )

        frame_gain = uniform_filter1d(frame_gain.astype(np.float64), size=7)

        sample_gain = np.interp(
            np.arange(len(y)),
            np.arange(n_frames) * hop + hop // 2,
            frame_gain
        )

        return self._peak_norm(y * sample_gain)

    # ========== NEW LOUDNESS STAGE (v3) ==========

    def _speech_agc(self, y: np.ndarray, sr: int, vad_mask: np.ndarray, target_db: float = -18.0, max_gain: float = DEFAULT_MAX_GAIN) -> np.ndarray:
        """Sliding-window AGC: levels each speech passage to a consistent target,
        measuring RMS only on VAD-active samples so silence/noise is not amplified."""
        frame_len = int(sr * 0.4)
        hop = frame_len // 4
        n_frames = max(1, (len(y) - frame_len) // hop + 1)

        target = 10 ** (target_db / 20)

        frame_gain = np.ones(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = min(len(y), start + frame_len)
            seg = y[start:end]
            seg_vad = vad_mask[start:end]
            active = seg_vad > 0.3
            if not np.any(active):
                continue
            vad_ratio = float(np.mean(active))
            if vad_ratio < 0.2:
                continue
            seg_rms = np.sqrt(np.mean(seg[active] ** 2) + 1e-12)
            if seg_rms <= 1e-6:
                continue
            frame_gain[i] = np.clip(target / seg_rms, 1.0 / max_gain, max_gain)

        smooth_frames = max(3, int(0.8 * sr / hop)) | 1
        frame_gain = uniform_filter1d(frame_gain.astype(np.float64), size=smooth_frames)

        sample_gain = np.interp(
            np.arange(len(y)),
            np.arange(n_frames) * hop + frame_len // 2,
            frame_gain
        )

        return y * sample_gain

    def _speech_loudness_norm(self, y: np.ndarray, sr: int, vad_mask: np.ndarray, target_db: float = DEFAULT_TARGET_DB) -> np.ndarray:
        """Normalize based on the p90 frame RMS of ACTIVE-SPEECH frames only
        (robust speech level, not diluted by pauses inside the VAD mask nor
        skewed by sparse transients), so speech lands at the target level."""
        frame_len = int(sr * 0.02)
        hop = frame_len
        n_frames = max(1, (len(y) - frame_len) // hop + 1)

        rms = np.array([
            np.sqrt(np.mean(y[i*hop:i*hop+frame_len] ** 2) + 1e-12)
            for i in range(n_frames)
        ])
        frame_vad = np.array([
            np.mean(vad_mask[i*hop:min(len(vad_mask), i*hop+frame_len)])
            for i in range(n_frames)
        ])

        active_rms = rms[frame_vad > 0.3]
        if len(active_rms) == 0:
            active_rms = rms

        speech_level = float(np.percentile(active_rms, 90))
        target_rms = 10 ** (target_db / 20)
        if speech_level > 1e-9:
            gain = min(target_rms / speech_level, 40.0)
            return y * gain
        return y

    def _peak_limit(self, y: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
        """Per-sample soft limiter (tanh waveshaper): transparent below the
        ceiling, squashes only the loud transients themselves instead of
        scaling the whole file down by its single loudest peak."""
        peak = np.max(np.abs(y)) if len(y) else 0.0
        if peak > ceiling:
            y = ceiling * np.tanh(y / ceiling)
        return np.clip(y, -1.0, 1.0)

    def _peak_norm(self, y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y))
        if peak > 0:
            return y * (0.99 / peak)
        return y


def main():
    parser = argparse.ArgumentParser(description='Forensic Audio Processor v3')
    parser.add_argument('--input', '-i', type=str, help='Caminho do arquivo ou diretório de áudio')
    parser.add_argument('--output', '-o', type=str, help='Diretório de saída')
    parser.add_argument('--workers', '-w', type=int, default=MAX_WORKERS, help='Threads paralelas (2 recomendado para RAM)')
    parser.add_argument('--target-db', type=float, default=DEFAULT_TARGET_DB, help='Nível alvo da fala ativa em dBFS (padrão -12)')
    parser.add_argument('--max-gain', type=float, default=DEFAULT_MAX_GAIN, help='Ganho máximo do AGC de fala (padrão 12)')
    parser.add_argument('--wav', action='store_true', help='Salvar saída em WAV (lossless) em vez de MP3')

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_path = args.input or str(base_dir)
    output_dir = args.output or str(base_dir / "audio_processed")

    inp = Path(input_path)
    if not inp.exists():
        print(f"Caminho não encontrado: {input_path}")
        return

    print(f"\nProcessando: {input_path}")
    print(f"Saída: {output_dir}")
    print(f"Threads: {args.workers}")
    print(f"Alvo de fala: {args.target_db} dBFS | Ganho máx AGC: {args.max_gain}x\n")

    t_global = time.time()
    try:
        processor = ForensicAudioProcessorV3(input_path, output_dir,
                                             target_db=args.target_db,
                                             max_gain=args.max_gain,
                                             save_wav=args.wav)
        results = processor.process_all_audio(max_workers=args.workers)
        elapsed = time.time() - t_global
        logger.debug("Processing complete")
        print(f"\nProcessamento concluído em {elapsed:.0f}s!")

        for name, result in results.items():
            if "error" not in result:
                print(f"  {name}: {result['pipelines']} pipelines, {len(result['files_generated'])} arquivos")
                for f in result['files_generated']:
                    print(f"    -> {f}")
            else:
                print(f"  {name}: ERRO - {result['error']}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
