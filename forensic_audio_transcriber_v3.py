#!/usr/bin/env python3
"""
Forensic Audio Transcriber V3 — whisper-timestamped

DTW-based word-level timestamps with per-word confidence scores.
Built-in VAD preprocessing. Best hallucination detection at word level.
Integrated audio preprocessing (noise reduction, VAD-gated spectral boost) from V2.

pip install whisper-timestamped librosa soundfile noisereduce scipy
"""

import argparse
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import torch
import whisper_timestamped as whisper
from scipy.signal import butter, sosfilt
from scipy.ndimage import median_filter

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}
_PREPROCESS_SR = 44100
_STFT_CHUNK_SECONDS = 300

_ffmpeg_path: Optional[str] = None
try:
    import imageio_ffmpeg
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = str(Path(_ffmpeg_path).parent) + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    for candidate in ["ffmpeg", "ffmpeg.exe"]:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            _ffmpeg_path = candidate
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _declip(y: np.ndarray) -> np.ndarray:
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


def _highpass(y: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
    sos = butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, y)


def _peak_norm(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        return y * (0.99 / peak)
    return y


def _loudness_norm(y: np.ndarray, target_db: float = -8) -> np.ndarray:
    target_rms = 10 ** (target_db / 20)
    current_rms = np.sqrt(np.mean(y ** 2) + 1e-10)
    if current_rms > 0:
        gain = min(target_rms / current_rms, 40.0)
        return np.clip(y * gain, -1.0, 1.0)
    return y


def _dynamic_compress(y: np.ndarray, sr: int) -> np.ndarray:
    threshold_db = -20
    ratio = 2.5
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
    return _peak_norm(y * sample_gain)


def _process_stft_chunked(y: np.ndarray, sr: int, process_fn) -> np.ndarray:
    chunk_samples = int(_STFT_CHUNK_SECONDS * sr)
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


def _compute_vad_mask(y: np.ndarray, sr: int) -> np.ndarray:
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
    rms_threshold = silence_rms * 2.0
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


def _whisper_spectral_boost_vad(y: np.ndarray, sr: int, vad_mask: np.ndarray) -> np.ndarray:
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
        speech_mask = (freqs >= 150) & (freqs <= 8000)
        speech_energy = np.mean(magnitude[speech_mask, :], axis=0)
        valid = speech_energy > 0
        median_energy = np.median(speech_energy[valid]) if np.any(valid) else 1e-10
        frame_gain = np.ones(n_stft_frames)
        quiet = speech_energy < (median_energy * 0.3)
        has_voice = frame_vad > 0.3
        boost_frames = quiet & has_voice
        if np.any(boost_frames) and median_energy > 0:
            frame_gain[boost_frames] = np.clip(median_energy / (speech_energy[boost_frames] + 1e-10), 1.0, 4.0)
        freq_gain = np.ones_like(freqs)
        freq_gain[speech_mask] = 1.5
        freq_gain[(freqs < 150)] = 0.3
        freq_gain[(freqs > 8000)] = 0.2
        return magnitude * freq_gain[:, np.newaxis] * frame_gain[np.newaxis, :] * np.exp(1j * phase)

    return _peak_norm(_process_stft_chunked(y, sr, _process))


def _boost_quiet_segments_vad(y: np.ndarray, sr: int, vad_mask: np.ndarray, max_gain: float = 4.0) -> np.ndarray:
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
    sample_gain = np.interp(
        np.arange(len(y)),
        np.arange(n_frames) * hop + hop // 2,
        frame_gain
    )
    return y * sample_gain


def _load_and_convert(audio_path: Path) -> Path:
    ext = audio_path.suffix.lower()
    if ext == ".wav":
        return audio_path
    temp_wav = audio_path.parent / f"_tmp_preprocess_{audio_path.stem}.wav"
    ffmpeg = _ffmpeg_path or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-i", str(audio_path),
        "-ar", str(_PREPROCESS_SR), "-ac", "1",
        "-sample_fmt", "s16", str(temp_wav)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")
    return temp_wav


def _preprocess_audio(audio_path: Path, output_dir: Path) -> Path:
    t0 = time.time()
    logger.debug(f"Preprocessing audio: {audio_path.name}")

    temp_input = None
    wav_path = audio_path
    if audio_path.suffix.lower() != ".wav":
        wav_path = _load_and_convert(audio_path)
        temp_input = wav_path

    try:
        y, sr = librosa.load(str(wav_path), sr=_PREPROCESS_SR, mono=True)
    finally:
        if temp_input and temp_input.exists():
            temp_input.unlink()

    vad_mask = _compute_vad_mask(y, sr)
    y = _declip(y)
    y = _highpass(y, sr, 80)
    y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.60,
                        n_std_thresh_stationary=2.0, n_fft=2048)
    y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.50,
                        thresh_n_mult_nonstationary=3.0, sigmoid_slope_nonstationary=10, n_fft=2048)
    y = _whisper_spectral_boost_vad(y, sr, vad_mask)
    y = _boost_quiet_segments_vad(y, sr, vad_mask, max_gain=4.0)
    y = _dynamic_compress(y, sr)
    y = _loudness_norm(y, target_db=-8)
    y = _peak_norm(y)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = output_dir / f"_tmp_preprocessed_{audio_path.stem}.wav"
    sf.write(str(temp_wav), y, sr, subtype='PCM_16')

    elapsed = time.time() - t0
    logger.debug(f"Preprocessing done in {elapsed:.0f}s: {audio_path.name}")
    print(f"    Pré-processamento concluído em {elapsed:.0f}s")
    return temp_wav


def _load_audio(input_path: Path, output_dir: Path) -> np.ndarray:
    import wave as _wave

    preprocessed_path = None
    try:
        preprocessed_path = _preprocess_audio(input_path, output_dir)
        wav_path = preprocessed_path
    except Exception as e:
        logger.error(f"Preprocessing failed for {input_path.name}, using raw conversion: {e}")
        wav_path = None

    if wav_path is None:
        if not _ffmpeg_path:
            raise RuntimeError("ffmpeg not found, cannot convert audio")
        import tempfile
        wav_path = Path(tempfile.gettempdir()) / f"whisper_{input_path.stem}.wav"
        cmd = [
            _ffmpeg_path, "-y",
            "-i", str(input_path),
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le",
            str(wav_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        preprocessed_path = wav_path

    try:
        with _wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            audio_np = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768.0
    finally:
        if preprocessed_path and preprocessed_path.exists():
            preprocessed_path.unlink(missing_ok=True)

    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

    return audio_np


def collect_audio_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        logger.error(f"Unsupported format: {input_path.suffix}")
        return []

    if input_path.is_dir():
        files = []
        for f in sorted(input_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(f)
        return files

    return []


_FORENSIC_PROMPT = (
    "Transcrição de conversa em português brasileiro. "
    "Possível conteúdo explícito: sexo, abuso, estupro, violência, assédio. "
    "Transcrever fielmente tudo que for dito, incluindo palavrões e termos sexuais."
)

_EXPLICIT_KEYWORDS = {
    "sexo", "sexual", "transar", "transou", "transando",
    "foder", "fuder", "fodeu", "fudeu", "fodendo", "fudendo",
    "buceta", "boceta", "xereca", "xoxota", "ppk",
    "pau", "pica", "rola", "cacete", "pinto",
    "chupar", "chupou", "chupando", "mamar", "mamou", "mamando", "boquete",
    "gozar", "gozou", "gozando", "goza", "enfia", "mete",
    "porra", "caralho", "merda", "puta", "putaria",
    "dedo", "dedos", "deda", "dedas", "dedou",
    "cu", "bunda", "rabo",
    "pelada", "pelado", "nua", "nu",
    "estupro", "estupra", "estuprar", "estuprou", "estuprando",
    "abuso", "abusar", "abusou", "abusando",
    "assédio", "assediar", "assediou",
    "molestar", "molestou", "molestando",
    "violência", "violentar", "violentou",
    "gemido", "gemendo", "gemer", "gemeu", "geme",
    "tesão", "excitado", "excitada",
    "masturba", "masturbar", "masturbou", "masturbando", "masturbação",
    "penetrar", "penetrou", "penetração",
    "ejacular", "ejaculou", "ejaculação",
    "oral", "vaginal", "anal",
}

_WORD_CONFIDENCE_THRESHOLD = 0.3


_YOUTUBE_PATTERNS = [
    re.compile(r"legenda(s|do|gem)?\s+(por|e)\s+", re.IGNORECASE),
    re.compile(r"transcri(ção|cão)\s+e\s+legenda", re.IGNORECASE),
    re.compile(r"obrigad[ao]\s+por\s+assistir", re.IGNORECASE),
    re.compile(r"inscreva-se\s+no\s+canal", re.IGNORECASE),
    re.compile(r"ative\s+o\s+sininho", re.IGNORECASE),
    re.compile(r"acesse\s+o\s+(nosso\s+)?site", re.IGNORECASE),
    re.compile(r"www\.\w+\.\w+", re.IGNORECASE),
]

_PROMPT_FRAGMENTS = [
    "transcrição de conversa em português",
    "conteúdo explícito",
    "transcrever fielmente",
    "incluindo palavrões",
    "acompanhe o processo de transcrição",
    "acompanhe a conversa em português",
    "português brasileiro",
]

_GENERIC_HALLUCINATION_PATTERNS = [
    re.compile(r"dispon[íi]vel em portugu[eê]s", re.IGNORECASE),
    re.compile(r"^a cidade no brasil\.?$", re.IGNORECASE),
]


def _has_ngram_repetition(words: list, ngram_sizes=(2, 3, 4), min_repeats: int = 4) -> bool:
    for n in ngram_sizes:
        if len(words) < n * min_repeats:
            continue
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        from collections import Counter
        counts = Counter(ngrams)
        most_common_ng, count = counts.most_common(1)[0]
        if count >= min_repeats and (count * n) / len(words) > 0.5:
            return True
    return False


def _has_sequential_numbers(text: str) -> bool:
    numbers = re.findall(r'\d+', text)
    if len(numbers) < 5:
        return False
    nums = [int(n) for n in numbers]
    consecutive = 0
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 4:
            return True
    return False


def _has_extended_char_repetition(text: str, min_repeat: int = 5) -> bool:
    for match in re.finditer(r'(.)\1{' + str(min_repeat - 1) + r',}', text):
        if match.group(1) not in (' ', '.', ','):
            return True
    return False


def _is_hallucination(seg: dict) -> bool:
    text = seg.get("text", "").strip()
    if not text:
        return True

    seg_duration = seg.get("end", 0) - seg.get("start", 0)
    if seg_duration > 120:
        return True

    no_speech = seg.get("no_speech_prob", 0.0)
    confidence = seg.get("confidence", 1.0)
    compression = seg.get("compression_ratio", 1.0)
    avg_logprob = seg.get("avg_logprob", 0.0)

    if no_speech > 0.85 and avg_logprob < -1.5:
        return True
    if compression > 3.0 and avg_logprob < -1.5:
        return True
    if confidence < _WORD_CONFIDENCE_THRESHOLD:
        return True

    seg_words = seg.get("words", [])
    if seg_words:
        low_conf_count = sum(1 for w in seg_words if w.get("confidence", 1.0) < _WORD_CONFIDENCE_THRESHOLD)
        if len(seg_words) > 0 and low_conf_count / len(seg_words) > 0.7:
            return True

    if _has_extended_char_repetition(text):
        return True

    if _has_sequential_numbers(text):
        return True

    text_lower = text.lower()
    for frag in _PROMPT_FRAGMENTS:
        if frag in text_lower:
            return True
    for pat in _YOUTUBE_PATTERNS:
        if pat.search(text):
            return True
    for pat in _GENERIC_HALLUCINATION_PATTERNS:
        if pat.search(text.strip()):
            return True

    words = [w for w in text_lower.replace(",", " ").replace(".", " ").split() if w]
    unique = set(words)
    if len(words) >= 4 and len(unique) <= 2 and unique.issubset(_EXPLICIT_KEYWORDS):
        return True
    if len(words) >= 6:
        from collections import Counter
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(words) > 0.7:
            return True

    if len(words) >= 8 and _has_ngram_repetition(words):
        return True

    return False


def _flag_explicit(text: str) -> bool:
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    return bool(words & _EXPLICIT_KEYWORDS)


def transcribe_file(
    model,
    audio_path: Path,
    output_dir: Path,
    word_timestamps: bool = False,
    model_name: str = "large-v3",
) -> Optional[Path]:
    t_start = time.time()
    print(f"\n    Transcrevendo: {audio_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{audio_path.stem}_transcription.json"

    try:
        audio = _load_audio(audio_path, output_dir)
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path.name}: {e}")
        return None

    try:
        result = whisper.transcribe(
            model,
            audio,
            language="pt",
            verbose=False,
            beam_size=10,
            condition_on_previous_text=False,
            compression_ratio_threshold=3.0,
            no_speech_threshold=0.8,
            temperature=(0.0, 0.2, 0.4),
            vad="silero",
            initial_prompt=_FORENSIC_PROMPT,
        )
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path.name}: {e}")
        return None

    segments_dict = {}
    explicit_segments = {}
    words_list = []
    prev_text = None
    repeat_count = 0
    hallu_count = 0

    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue

        if _is_hallucination(seg):
            hallu_count += 1
            logger.debug(
                f"Hallucination filtered (nsp={seg.get('no_speech_prob', 0):.2f} "
                f"conf={seg.get('confidence', 0):.2f} "
                f"cr={seg.get('compression_ratio', 0):.2f}): '{text}'"
            )
            continue

        if text == prev_text:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            repeat_count = 0
        prev_text = text

        key = f"{_fmt_time(seg['start'])},{_fmt_time(seg['end'])}"
        segments_dict[key] = text

        if _flag_explicit(text):
            explicit_segments[key] = text

        if word_timestamps and seg.get("words"):
            for w in seg["words"]:
                words_list.append({
                    "start": round(w["start"], 2),
                    "end": round(w["end"], 2),
                    "text": w["text"].strip(),
                    "confidence": round(w.get("confidence", 0.0), 3),
                })

    duration = round(len(audio) / 16000.0, 1)

    output_data = {
        "source_file": audio_path.name,
        "language": result.get("language", "pt"),
        "model": model_name,
        "duration": duration,
        "segments": segments_dict,
    }
    if hallu_count > 0:
        output_data["hallucinations_filtered"] = hallu_count
    if explicit_segments:
        output_data["explicit_segments"] = explicit_segments
        output_data["explicit_count"] = len(explicit_segments)
    if word_timestamps and words_list:
        output_data["words"] = words_list

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    seg_count = len(segments_dict)
    print(f"    Concluído em {elapsed:.0f}s — {seg_count} segmentos ({hallu_count} alucinações filtradas) -> {json_path.name}")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic Audio Transcriber V3 (whisper-timestamped)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Arquivo de áudio ou pasta com áudios")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Diretório de saída para JSONs (padrão: transcriptions/)")
    parser.add_argument("--model", "-m", type=str, default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Modelo Whisper (padrão: large-v3)")
    parser.add_argument("--word-timestamps", action="store_true",
                        help="Incluir timestamps e confidence por palavra")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Forensic Audio Transcriber V3 (whisper-timestamped)")
    print("=" * 60)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"  ERRO: Caminho não encontrado: {input_path}")
        return 1

    output_dir = Path(args.output).resolve() if args.output else Path.cwd() / "transcriptions"

    audio_files = collect_audio_files(input_path)
    if not audio_files:
        print("  Nenhum arquivo de áudio encontrado.")
        return 1

    random.shuffle(audio_files)
    print(f"\n  Arquivos encontrados: {len(audio_files)} (ordem aleatória)")
    for af in audio_files:
        print(f"    - {af.name}")

    if not _ffmpeg_path:
        print("  AVISO: ffmpeg não encontrado. Alguns formatos podem falhar.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Carregando modelo whisper-timestamped '{args.model}' ({device})...")
    t_load = time.time()
    model = whisper.load_model(args.model, device=device)
    print(f"  Modelo carregado em {time.time() - t_load:.0f}s")

    t_total = time.time()
    success = 0
    failed = 0
    skipped = 0

    for idx, af in enumerate(audio_files, 1):
        json_out = output_dir / f"{af.stem}_transcription.json"
        if json_out.exists() and json_out.stat().st_size > 0:
            print(f"  [{idx}/{len(audio_files)}] {af.name} — já transcrito, pulando")
            skipped += 1
            continue
        print(f"\n  [{idx}/{len(audio_files)}] {af.name}")
        try:
            result = transcribe_file(model, af, output_dir, args.word_timestamps, args.model)
            if result:
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Unexpected error on {af.name}: {e}", exc_info=True)
            print(f"    ERRO: {e}")
            failed += 1

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Transcrição concluída em {elapsed_total:.0f}s")
    print(f"  Sucesso: {success} | Falhas: {failed} | Pulados: {skipped}")
    print(f"  Saída: {output_dir}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
