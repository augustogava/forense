import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import noisereduce as nr
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

LOG = logging.getLogger(__name__)


def mp3_to_wav(ffmpeg_exe: str, mp3: Path, wav: Path) -> None:
    subprocess.run(
        [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp3),
            str(wav),
        ],
        check=True,
    )


def ffprobe_streams(ffmpeg_exe: str, media: Path) -> dict:
    out = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", str(media)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (out.stderr or "") + (out.stdout or "")
    info: dict = {"path": str(media)}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", text)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration_sec"] = hh * 3600 + mm * 60 + ss
    m = re.search(r"Audio:\s*[^,]+,\s*(\d+)\s*Hz", text)
    if m:
        info["sample_rate_hz"] = int(m.group(1))
    if re.search(r"stereo", text, re.I):
        info["channels"] = 2
    elif re.search(r"mono", text, re.I):
        info["channels"] = 1
    return info


def int16_to_float(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        return y / 32768.0
    return y / 32768.0


def float_to_int16(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


def dbfs(rms: float) -> float:
    if rms <= 1e-12:
        return -120.0
    return float(20.0 * np.log10(rms))


def analyze_waveform(y: np.ndarray, sr: int) -> dict:
    mono = y.mean(axis=1) if y.ndim == 2 else y
    n = len(mono)
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    frame = max(1, int(sr * 0.02))
    rms_frames = []
    for i in range(0, n, frame):
        seg = mono[i : i + frame]
        if seg.size:
            rms_frames.append(float(np.sqrt(np.mean(seg**2))))
    rms_frames = np.asarray(rms_frames, dtype=np.float64)
    dyn_db = float(dbfs(np.percentile(rms_frames, 95))) - float(dbfs(np.percentile(rms_frames, 5) + 1e-12))
    n_fft = min(65536, max(2048, 1 << int(np.ceil(np.log2(n)))))
    w = np.hanning(n_fft)
    seg = mono[:n_fft] * w[: min(n_fft, n)]
    spec = np.abs(np.fft.rfft(seg, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    low = (freqs >= 80) & (freqs < 1000)
    mid = (freqs >= 1000) & (freqs < 4000)
    high = (freqs >= 4000) & (freqs < min(12000, sr / 2 - 1))
    e_low = float(np.mean(spec[low] ** 2) + 1e-20)
    e_mid = float(np.mean(spec[mid] ** 2) + 1e-20)
    e_high = float(np.mean(spec[high] ** 2) + 1e-20)
    tilt = float(10 * np.log10((e_high + e_mid) / (e_low + 1e-20)))
    return {
        "samples": int(n),
        "channels": int(y.shape[1]) if y.ndim == 2 else 1,
        "sample_rate_hz": int(sr),
        "rms_dbfs": round(dbfs(rms), 2),
        "peak_dbfs": round(dbfs(peak), 2),
        "dynamic_range_db_approx": round(dyn_db, 2),
        "spectral_tilt_high_vs_low_db": round(tilt, 2),
    }


def highpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    nyq = sr * 0.5
    wn = min(hz / nyq, 0.99)
    b, a = butter(2, wn, btype="highpass")
    if y.ndim == 2:
        return np.stack([filtfilt(b, a, y[:, c]) for c in range(y.shape[1])], axis=1)
    return filtfilt(b, a, y)


def preemphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    if y.ndim == 2:
        return np.stack([preemphasis(y[:, c], coef) for c in range(y.shape[1])], axis=1)
    out = np.empty_like(y, dtype=np.float32)
    out[0] = y[0]
    out[1:] = y[1:] - coef * y[:-1]
    return out


def spectral_tilt_boost(y: np.ndarray, sr: int, f_break_hz: float, gain_db: float) -> np.ndarray:
    n = y.shape[0]
    mono = y.mean(axis=1) if y.ndim == 2 else y
    n_fft = 1 << int(np.ceil(np.log2(n)))
    pad = n_fft - n
    x = np.pad(mono.astype(np.float64), (0, pad), mode="constant")
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    g = np.ones_like(freqs, dtype=np.float64)
    g[freqs > f_break_hz] *= 10 ** (gain_db / 20.0)
    mono2 = np.fft.irfft(X * g, n=n_fft)[:n].astype(np.float32)
    if y.ndim == 2:
        ratio = mono2 / (np.abs(mono) + 1e-8)
        ratio = np.clip(ratio, 0.5, 2.5)
        return (y * ratio[:, None]).astype(np.float32)
    return mono2


def gentle_compress_normalize(
    y: np.ndarray,
    peak_dbfs: float = -0.12,
    makeup_db: float = 5.0,
) -> np.ndarray:
    y = np.tanh(1.12 * y) / 1.12
    peak = float(np.max(np.abs(y))) + 1e-12
    target = 10 ** (peak_dbfs / 20.0)
    y = (y / peak) * target
    y = y * (10 ** (makeup_db / 20.0))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def reduce_noise(
    y: np.ndarray,
    sr: int,
    stationary: bool,
    prop_decrease: float,
    noise_profile_sec: float,
) -> np.ndarray:
    def one(ch: np.ndarray) -> np.ndarray:
        kwargs = dict(y=ch, sr=sr, stationary=stationary, prop_decrease=prop_decrease)
        if noise_profile_sec > 0:
            n_noise = int(min(len(ch), noise_profile_sec * sr))
            if n_noise >= 512:
                kwargs["y_noise"] = ch[:n_noise]
        return nr.reduce_noise(**kwargs).astype(np.float32)

    if y.ndim == 2 and y.shape[1] >= 2:
        return np.stack([one(y[:, c]) for c in range(y.shape[1])], axis=1)
    return one(y.reshape(-1))


def process(
    y_i16: np.ndarray,
    sr: int,
    stationary: bool,
    prop_decrease: float,
    noise_profile_sec: float,
    hp_hz: float,
    pre_coef: float,
    tilt_break_hz: float,
    tilt_gain_db: float,
    peak_dbfs: float,
    makeup_db: float,
) -> np.ndarray:
    y = int16_to_float(y_i16)
    y = highpass(y, sr, hp_hz)
    y = reduce_noise(y, sr, stationary, prop_decrease, noise_profile_sec)
    y = preemphasis(y, pre_coef)
    y = spectral_tilt_boost(y, sr, tilt_break_hz, tilt_gain_db)
    y = gentle_compress_normalize(y, peak_dbfs=peak_dbfs, makeup_db=makeup_db)
    return float_to_int16(y)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Análise e limpeza de áudio (ruído + realce de voz fraca)",
        epilog="Exemplo: %(prog)s -i gravacao.mp3 -o limpo.wav   ou   %(prog)s gravacao.mp3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=None,
        metavar="AUDIO",
        help="Ficheiro de áudio de entrada (MP3, WAV, FLAC, …). Se omitido, usa -i/--input ou nata_140426.mp3 na pasta do script.",
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        dest="input_opt",
        help="Caminho do áudio de entrada (alternativa ao argumento posicional AUDIO)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="WAV de saída (predefinição: cleaned_audio.wav na pasta do script)",
    )
    p.add_argument("--analyze-only", action="store_true", help="Só analisar, não gravar")
    p.add_argument(
        "--non-stationary",
        action="store_true",
        help="Ruído variável no tempo (ex.: ambiente irregular; mais CPU)",
    )
    p.add_argument("--prop-decrease", type=float, default=0.88, help="Força da redução de ruído 0-1")
    p.add_argument("--noise-profile-sec", type=float, default=0.0, help="Segundos iniciais como perfil de ruído (0=auto)")
    p.add_argument("--hp-hz", type=float, default=100.0, help="High-pass (Hz)")
    p.add_argument("--tilt-db", type=float, default=4.5, help="Ganho em dB acima de --tilt-break-hz")
    p.add_argument("--tilt-break-hz", type=float, default=1400.0, help="Início do realce espectral (Hz)")
    p.add_argument(
        "--peak-dbfs",
        type=float,
        default=-0.12,
        help="Pico alvo após normalização (0 = máximo teórico; valores negativos = margem)",
    )
    p.add_argument(
        "--makeup-db",
        type=float,
        default=5.0,
        help="Ganho extra em dB após normalização (antes do clip final)",
    )
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    input_path = (args.audio or args.input_opt or (base / "nata_140426.mp3")).resolve()
    output_path = (args.output or (base / "cleaned_audio.wav")).resolve()
    wav_path = base / ".temp_forensic.wav"

    if not input_path.is_file():
        LOG.error("Input not found: %s", input_path)
        print(f"Ficheiro inexistente: {input_path}")
        return 1

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ff_info = ffprobe_streams(ffmpeg_exe, input_path)
    LOG.debug("ffmpeg parse: %s", ff_info)

    try:
        if input_path.suffix.lower() in (".mp3", ".m4a", ".aac", ".flac", ".ogg"):
            mp3_to_wav(ffmpeg_exe, input_path, wav_path)
            read_path = wav_path
        else:
            read_path = input_path

        sr, data = wavfile.read(str(read_path))
        if data.dtype != np.int16:
            data = np.asarray(data, dtype=np.int32)
            data = np.clip(data, -32768, 32767).astype(np.int16)

        y_f = int16_to_float(data)
        stats = analyze_waveform(y_f, int(sr))
        stats["container"] = input_path.suffix.lower()
        stats["ffmpeg_parse"] = {k: v for k, v in ff_info.items() if k != "path"}

        print(json.dumps({"analysis": stats}, indent=2, ensure_ascii=False))
        if stats["rms_dbfs"] < -45:
            LOG.info("Very low RMS; whisper-style content likely — spectral tilt applied")

        if args.analyze_only:
            print("Modo análise apenas; não foi gerado WAV de saída.")
            return 0

        cleaned = process(
            data,
            int(sr),
            stationary=not args.non_stationary,
            prop_decrease=float(np.clip(args.prop_decrease, 0.05, 1.0)),
            noise_profile_sec=max(0.0, args.noise_profile_sec),
            hp_hz=max(20.0, args.hp_hz),
            pre_coef=0.97,
            tilt_break_hz=max(200.0, args.tilt_break_hz),
            tilt_gain_db=max(0.0, args.tilt_db),
            peak_dbfs=float(np.clip(args.peak_dbfs, -6.0, 0.0)),
            makeup_db=float(np.clip(args.makeup_db, 0.0, 18.0)),
        )
        wavfile.write(str(output_path), int(sr), cleaned)
        LOG.info("Wrote %s", output_path)
        print(f"Concluído: {output_path}")
        return 0
    except subprocess.CalledProcessError as e:
        LOG.error("ffmpeg failed: %s", e)
        print("Falha no ffmpeg ao ler/converter o áudio.")
        return 1
    except Exception as e:
        LOG.error("Processing failed: %s", e)
        print(f"Erro no processamento: {e}")
        return 1
    finally:
        try:
            if wav_path.is_file():
                os.remove(wav_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
