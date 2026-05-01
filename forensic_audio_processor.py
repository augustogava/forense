#!/usr/bin/env python3
"""
Forensic Audio Processor

Script Python para processar áudios forenses aplicando múltiplas técnicas
combinadas em pipelines, executadas em paralelo via threads.
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
from scipy.ndimage import median_filter

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
_print_lock = threading.Lock()


def _tprint(msg: str):
    with _print_lock:
        print(msg, flush=True)


class ForensicAudioProcessor:

    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.opus'}

    def __init__(self, input_path: str, output_dir: str, sample_rate: int = TARGET_SR):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate

        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized audio processor - Input: {input_path}, Output: {output_dir}")

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
        output_path = self.output_dir / f"{base_name}_{suffix}.mp3"
        sf.write(str(temp_wav), audio, sr, subtype='PCM_16')
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

    def process_all_audio(self, max_workers: int = MAX_WORKERS) -> Dict:
        if self.input_path.is_file():
            audio_files = [self.input_path]
        else:
            audio_files = self._get_audio_files()

        if not audio_files:
            logger.warning("No audio files found")
            return {}

        results = {}
        for audio_path in audio_files:
            try:
                result = self.process_single_audio(audio_path, max_workers)
                results[audio_path.name] = result
            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                results[audio_path.name] = {"error": str(e)}

        return results

    def _get_audio_files(self) -> List[Path]:
        if not self.input_path.is_dir():
            return []
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_path.glob(f"*{ext}"))
            files.extend(self.input_path.glob(f"*{ext.upper()}"))
        return sorted(set(files))

    def process_single_audio(self, audio_path: Path, max_workers: int = MAX_WORKERS) -> Dict:
        logger.debug(f"Loading audio: {audio_path.name}")
        y, sr = self._load_audio(audio_path)
        base_name = audio_path.stem

        duration = len(y) / sr
        logger.debug(f"Audio loaded: {len(y)} samples, {sr}Hz, duration: {duration:.1f}s")
        _tprint(f"\n  Áudio carregado: {base_name} ({duration:.1f}s / {duration/60:.1f}min)")

        pipelines = [
            ("clean", "Redução de ruído limpa", self._pipeline_clean),
            # ("vocal_enhanced", "Realce vocal + formantes", self._pipeline_vocal_enhanced),
            ("whisper_boost", "Realce de sussurros", self._pipeline_whisper_boost),
            # ("forensic_full", "Pipeline forense completa", self._pipeline_forensic_full),
        ]

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
            "files_generated": generated_files,
        }

        report_path = self.output_dir / f"{base_name}_processing_report.json"
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

    # ========== COMBINED PIPELINES ==========

    def _pipeline_clean(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Noise reduction + highpass + declip + normalize. Clean baseline."""
        y = self._declip(y)
        y = self._highpass(y, sr, 60)
        y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.80, time_constant_s=2.0, n_fft=2048)
        y = self._multiband_denoise(y, sr)
        y = self._loudness_norm(y, target_db=-8)
        y = self._peak_norm(y)
        return y

    def _pipeline_vocal_enhanced(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Clean + vocal isolation + formant boost + compression. Best for clear speech."""
        y = self._declip(y)
        y = self._highpass(y, sr, 60)
        y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.75, n_fft=2048)
        y = self._vocal_enhance(y, sr)
        y = self._formant_boost(y, sr)
        y = self._dynamic_compress(y, sr)
        y = self._loudness_norm(y, target_db=-8)
        y = self._peak_norm(y)
        return y

    def _pipeline_whisper_boost(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Aggressive noise reduction + whisper detection + heavy boost. For very quiet speech."""
        y = self._declip(y)
        y = self._highpass(y, sr, 80)
        y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.90, thresh_n_mult_nonstationary=1.5, sigmoid_slope_nonstationary=15, n_fft=2048)
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.7, n_std_thresh_stationary=1.2, n_fft=2048)
        y = self._whisper_spectral_boost(y, sr)
        y = self._boost_quiet_segments(y, sr, max_gain=10.0)
        y = self._dynamic_compress(y, sr)
        y = self._loudness_norm(y, target_db=-6)
        y = self._peak_norm(y)
        return y

    def _pipeline_forensic_full(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Complete forensic pipeline: all best techniques combined for maximum clarity."""
        y = self._declip(y)
        y = self._highpass(y, sr, 60)
        y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.80, time_constant_s=2.0, n_fft=2048)
        y = self._wiener_filter(y, sr)
        y = self._vocal_enhance(y, sr)
        y = self._formant_boost(y, sr)
        y = self._boost_quiet_segments(y, sr, max_gain=8.0)
        y = self._dynamic_compress(y, sr)
        y = self._loudness_norm(y, target_db=-8)
        y = self._peak_norm(y)
        return y

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

    def _multiband_denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        bands = [
            (20, 200, 0.95),
            (200, 800, 0.70),
            (800, 3000, 0.50),
            (3000, 6000, 0.65),
            (6000, min(16000, sr // 2 - 1), 0.90),
        ]

        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            noise_frames = min(30, magnitude.shape[1])
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            for low_f, high_f, reduction in bands:
                if low_f >= sr // 2:
                    continue
                mask = (freqs >= low_f) & (freqs <= min(high_f, sr // 2 - 1))
                subtracted = magnitude[mask] ** 2 - reduction * (noise_spectrum[mask] ** 2)
                magnitude[mask] = np.sqrt(np.maximum(subtracted, 0.02 * magnitude[mask] ** 2))
            return magnitude * np.exp(1j * phase)

        return self._process_stft_chunked(y, sr, _process)

    def _vocal_enhance(self, y: np.ndarray, sr: int) -> np.ndarray:
        sos_bp = butter(6, [80, 4000], btype='bandpass', fs=sr, output='sos')
        vocal_band = sosfilt(sos_bp, y)

        sos_presence = butter(4, [2000, 4000], btype='bandpass', fs=sr, output='sos')
        presence = sosfilt(sos_presence, y)

        return self._peak_norm(y + 0.3 * vocal_band + 0.15 * presence)

    def _formant_boost(self, y: np.ndarray, sr: int) -> np.ndarray:
        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            gain = np.ones_like(freqs)
            gain[(freqs >= 300) & (freqs <= 900)] = 1.4
            gain[(freqs >= 900) & (freqs <= 2500)] = 1.6
            gain[(freqs >= 2500) & (freqs <= 3500)] = 1.3
            return magnitude * gain[:, np.newaxis] * np.exp(1j * phase)

        return self._peak_norm(self._process_stft_chunked(y, sr, _process))

    def _wiener_filter(self, y: np.ndarray, sr: int) -> np.ndarray:
        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            noise_frames = min(30, magnitude.shape[1])
            noise_est = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            snr = (magnitude ** 2) / (noise_est ** 2 + 1e-10)
            gain = np.clip(np.maximum(1.0 - (1.0 / (snr + 1e-10)), 0.0), 0.05, 1.0)
            return magnitude * gain * np.exp(1j * phase)

        return self._process_stft_chunked(y, sr, _process)

    def _whisper_spectral_boost(self, y: np.ndarray, sr: int) -> np.ndarray:
        def _process(S, sr):
            magnitude = np.abs(S)
            phase = np.angle(S)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            speech_mask = (freqs >= 150) & (freqs <= 6000)
            speech_energy = np.mean(magnitude[speech_mask, :], axis=0)
            valid = speech_energy > 0
            median_energy = np.median(speech_energy[valid]) if np.any(valid) else 1e-10
            frame_gain = np.ones(magnitude.shape[1])
            quiet = speech_energy < (median_energy * 0.3)
            if np.any(quiet) and median_energy > 0:
                frame_gain[quiet] = np.clip(median_energy / (speech_energy[quiet] + 1e-10), 1.0, 8.0)
            freq_gain = np.ones_like(freqs)
            freq_gain[speech_mask] = 2.5
            freq_gain[~speech_mask] = 0.05
            return magnitude * freq_gain[:, np.newaxis] * frame_gain[np.newaxis, :] * np.exp(1j * phase)

        return self._peak_norm(self._process_stft_chunked(y, sr, _process))

    def _boost_quiet_segments(self, y: np.ndarray, sr: int, max_gain: float = 6.0) -> np.ndarray:
        frame_len = int(sr * 0.05)
        hop = frame_len // 2
        n_frames = max(1, (len(y) - frame_len) // hop + 1)

        rms = np.array([
            np.sqrt(np.mean(y[i*hop:i*hop+frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])

        silence_thresh = np.percentile(rms, 10)
        speech_rms = rms[rms > silence_thresh * 2]
        if len(speech_rms) == 0:
            return y

        target = np.percentile(speech_rms, 60)

        frame_gain = np.ones(n_frames)
        active = rms > silence_thresh * 1.5
        frame_gain[active] = np.clip(target / (rms[active] + 1e-10), 1.0, max_gain)

        sample_gain = np.interp(
            np.arange(len(y)),
            np.arange(n_frames) * hop + hop // 2,
            frame_gain
        )

        return y * sample_gain

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

        sample_gain = np.interp(
            np.arange(len(y)),
            np.arange(n_frames) * hop + hop // 2,
            frame_gain
        )

        return self._peak_norm(y * sample_gain)

    def _loudness_norm(self, y: np.ndarray, target_db: float = -8) -> np.ndarray:
        target_rms = 10 ** (target_db / 20)
        current_rms = np.sqrt(np.mean(y ** 2) + 1e-10)
        if current_rms > 0:
            gain = min(target_rms / current_rms, 40.0)
            return np.clip(y * gain, -1.0, 1.0)
        return y

    def _peak_norm(self, y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y))
        if peak > 0:
            return y * (0.99 / peak)
        return y


def main():
    parser = argparse.ArgumentParser(description='Forensic Audio Processor')
    parser.add_argument('--input', '-i', type=str, help='Caminho do arquivo ou diretório de áudio')
    parser.add_argument('--output', '-o', type=str, help='Diretório de saída')
    parser.add_argument('--workers', '-w', type=int, default=MAX_WORKERS, help='Threads paralelas (2 recomendado para RAM)')

    args = parser.parse_args()

    if args.input and args.output:
        input_path = args.input
        output_dir = args.output
    else:
        base_dir = Path(__file__).parent

        audio_files = []
        for ext in ForensicAudioProcessor.SUPPORTED_EXTENSIONS:
            audio_files.extend(base_dir.glob(f"*{ext}"))

        if not audio_files:
            print("Nenhum arquivo de áudio encontrado no diretório.")
            return

        print("\nArquivos de áudio disponíveis:")
        for idx, f in enumerate(sorted(audio_files), 1):
            print(f"  {idx}. {f.name}")

        while True:
            try:
                choice = input("\nEscolha o número do arquivo (ou 'q' para sair): ").strip()
                if choice.lower() == 'q':
                    print("Cancelado.")
                    return
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(audio_files):
                    input_path = str(sorted(audio_files)[choice_idx])
                    break
                else:
                    print("Número inválido. Tente novamente.")
            except ValueError:
                print("Entrada inválida. Digite um número.")

        output_dir = str(base_dir / "audio_processed")

    print(f"\nProcessando: {input_path}")
    print(f"Saída: {output_dir}")
    print(f"Threads: {args.workers}\n")

    t_global = time.time()
    try:
        processor = ForensicAudioProcessor(input_path, output_dir)
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
