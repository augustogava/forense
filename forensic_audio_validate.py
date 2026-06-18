#!/usr/bin/env python3
"""
Forensic Audio Validate

Compara metricas de qualidade entre o audio de entrada e as saidas geradas
para ajudar no ajuste (tuning) dos parametros do processador.

Metricas:
- DNSMOS P.835 (SIG/BAK/OVRL) -- nao-intrusiva, sem referencia limpa.
- Acusticas: LUFS, pico dBFS, clipping %, ruido de fundo, SNR estimado,
  inclinacao espectral, % de fala ativa.
- Opcional Whisper: contagem de segmentos/palavras, logprob medio,
  prob. media de ausencia de fala (intelig. para ASR).
"""

import argparse
import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

if hasattr(sys.stdout, "encoding") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import librosa
import soundfile as sf

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for _noisy in ("numba", "librosa", "matplotlib", "torch", "onnxruntime"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}
DNSMOS_SR = 16000
DNSMOS_INPUT_SECONDS = 9.01

_ffmpeg_path = None
try:
    import imageio_ffmpeg

    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    for _candidate in ("ffmpeg", "ffmpeg.exe"):
        try:
            subprocess.run([_candidate, "-version"], capture_output=True, check=True)
            _ffmpeg_path = _candidate
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass


def _load_mono(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    ext = path.suffix.lower()
    if ext == ".wav":
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
    else:
        ffmpeg = _ffmpeg_path or "ffmpeg"
        tmp = path.parent / f"_tmp_val_{path.stem}.wav"
        cmd = [ffmpeg, "-y", "-i", str(path), "-ar", str(target_sr), "-ac", "1", str(tmp)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join([l for l in (result.stderr or "").splitlines() if l.strip()][-5:])
            raise RuntimeError(f"ffmpeg failed: {tail}")
        data, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
        tmp.unlink(missing_ok=True)
        if data.ndim == 2:
            data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(data, dtype=np.float32), sr


def _dbfs(v: float) -> float:
    return float(20.0 * np.log10(v)) if v > 1e-12 else -120.0


def _speech_ratio(y: np.ndarray, sr: int) -> float:
    frame_len = int(sr * 0.03)
    hop = max(1, frame_len // 2)
    n_frames = max(1, (len(y) - frame_len) // hop + 1)
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame_len] ** 2) + 1e-10) for i in range(n_frames)])
    if not len(rms):
        return 0.0
    silence_rms = np.percentile(rms, 15)
    return float(np.mean(rms > silence_rms * 3.0))


def acoustic_metrics(y: np.ndarray, sr: int) -> Dict:
    n = len(y)
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    peak = float(np.max(np.abs(y))) if n else 0.0

    frame = max(1, int(sr * 0.02))
    rms_frames = np.array([np.sqrt(np.mean(y[i:i + frame] ** 2) + 1e-12) for i in range(0, n, frame)]) if n else np.array([1e-6])
    noise_floor = float(np.percentile(rms_frames, 10))
    speech_level = float(np.percentile(rms_frames, 90))
    snr_est = _dbfs(speech_level) - _dbfs(noise_floor)

    clipped = int(np.sum(np.abs(y) > 0.98))
    clip_pct = round(100.0 * clipped / n, 4) if n else 0.0

    win = min(n, 1 << 18) if n else 1
    spec = np.abs(np.fft.rfft(y[:win] * np.hanning(win))) if n else np.array([1e-6])
    freqs = np.fft.rfftfreq(win, 1.0 / sr) if n else np.array([0.0])
    low = (freqs >= 80) & (freqs < 1000)
    high = (freqs >= 1000) & (freqs < min(8000, sr / 2 - 1))
    e_low = float(np.mean(spec[low] ** 2) + 1e-20) if np.any(low) else 1e-20
    e_high = float(np.mean(spec[high] ** 2) + 1e-20) if np.any(high) else 1e-20
    tilt = float(10 * np.log10(e_high / e_low))

    lufs = _integrated_lufs(y, sr)

    return {
        "duration_seconds": round(n / sr, 2) if sr else 0.0,
        "lufs": round(lufs, 2) if lufs is not None else None,
        "rms_dbfs": round(_dbfs(rms), 2),
        "peak_dbfs": round(_dbfs(peak), 2),
        "noise_floor_dbfs": round(_dbfs(noise_floor), 2),
        "estimated_snr_db": round(snr_est, 2),
        "clipping_pct": clip_pct,
        "spectral_tilt_db": round(tilt, 2),
        "speech_active_pct": round(100.0 * _speech_ratio(y, sr), 2),
    }


def _integrated_lufs(y: np.ndarray, sr: int) -> Optional[float]:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        value = meter.integrated_loudness(y.astype(np.float64))
        return float(value) if np.isfinite(value) else None
    except Exception as e:
        logger.debug(f"pyloudnorm unavailable for LUFS: {e}")
        return None


# ========== DNSMOS P.835 ==========

_DNSMOS_MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
_DNSMOS_CACHE = Path(__file__).resolve().parent / "models" / "dnsmos"
_dnsmos_session = None
_dnsmos_backend = None


def _dnsmos_via_torchmetrics(y16: np.ndarray) -> Optional[Dict]:
    try:
        import torch
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
    except Exception as e:
        logger.debug(f"torchmetrics DNSMOS unavailable: {e}")
        return None
    try:
        metric = DeepNoiseSuppressionMeanOpinionScore(fs=DNSMOS_SR, personalized=False)
        scores = metric(torch.from_numpy(y16.astype(np.float32)))
        scores = scores.detach().cpu().numpy().reshape(-1)
        return {
            "P808": round(float(scores[0]), 3),
            "SIG": round(float(scores[1]), 3),
            "BAK": round(float(scores[2]), 3),
            "OVRL": round(float(scores[3]), 3),
            "backend": "torchmetrics",
        }
    except Exception as e:
        logger.warning(f"torchmetrics DNSMOS failed: {e}")
        return None


def _dnsmos_onnx_session():
    global _dnsmos_session
    if _dnsmos_session is not None:
        return _dnsmos_session
    try:
        import onnxruntime as ort
    except Exception as e:
        logger.debug(f"onnxruntime unavailable: {e}")
        return None
    model_path = _DNSMOS_CACHE / "sig_bak_ovr.onnx"
    if not model_path.exists():
        try:
            _DNSMOS_CACHE.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Downloading DNSMOS model to {model_path}")
            urlretrieve(_DNSMOS_MODEL_URL, str(model_path))
        except Exception as e:
            logger.warning(f"Failed to download DNSMOS model: {e}")
            return None
    try:
        _dnsmos_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return _dnsmos_session
    except Exception as e:
        logger.warning(f"Failed to load DNSMOS onnx session: {e}")
        return None


def _dnsmos_via_onnx(y16: np.ndarray) -> Optional[Dict]:
    session = _dnsmos_onnx_session()
    if session is None:
        return None
    try:
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

        input_len = int(DNSMOS_INPUT_SECONDS * DNSMOS_SR)
        hop = DNSMOS_SR
        audio = y16.astype(np.float32)
        if len(audio) < input_len:
            audio = np.tile(audio, int(np.ceil(input_len / max(1, len(audio)))))[:input_len]
        num_hops = max(1, int((len(audio) - input_len) / hop) + 1)

        sig_l, bak_l, ovr_l = [], [], []
        input_name = session.get_inputs()[0].name
        for idx in range(num_hops):
            seg = audio[int(idx * hop):int(idx * hop) + input_len]
            if len(seg) < input_len:
                break
            feats = np.array(seg).astype(np.float32)[np.newaxis, :]
            sig_raw, bak_raw, ovr_raw = session.run(None, {input_name: feats})[0][0]
            sig_l.append(p_sig(sig_raw))
            bak_l.append(p_bak(bak_raw))
            ovr_l.append(p_ovr(ovr_raw))
        if not sig_l:
            return None
        return {
            "SIG": round(float(np.mean(sig_l)), 3),
            "BAK": round(float(np.mean(bak_l)), 3),
            "OVRL": round(float(np.mean(ovr_l)), 3),
            "backend": "onnxruntime",
        }
    except Exception as e:
        logger.warning(f"onnx DNSMOS failed: {e}")
        return None


def dnsmos_score(y: np.ndarray, sr: int) -> Optional[Dict]:
    global _dnsmos_backend
    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=DNSMOS_SR) if sr != DNSMOS_SR else y.astype(np.float32)
    if _dnsmos_backend != "torchmetrics":
        result = _dnsmos_via_onnx(y16)
        if result is not None:
            _dnsmos_backend = "onnxruntime"
            return result
    result = _dnsmos_via_torchmetrics(y16)
    if result is not None:
        _dnsmos_backend = "torchmetrics"
    return result


# ========== WHISPER INTELLIGIBILITY ==========

_whisper_model = None


def whisper_metrics(path: Path, model_name: str = "base") -> Optional[Dict]:
    global _whisper_model
    try:
        import whisper
    except Exception as e:
        logger.warning(f"whisper unavailable for intelligibility: {e}")
        return None
    try:
        if _whisper_model is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Loading whisper model '{model_name}' on {device}")
            _whisper_model = whisper.load_model(model_name, device=device)
        audio, _ = _load_mono(Path(path), 16000)
        result = _whisper_model.transcribe(audio, language="pt", verbose=False)
        segments = result.get("segments", []) or []
        text = (result.get("text") or "").strip()
        word_count = len(text.split())
        logprobs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
        nospeech = [s.get("no_speech_prob") for s in segments if s.get("no_speech_prob") is not None]
        return {
            "segments": len(segments),
            "word_count": word_count,
            "avg_logprob": round(float(np.mean(logprobs)), 3) if logprobs else None,
            "mean_no_speech_prob": round(float(np.mean(nospeech)), 3) if nospeech else None,
        }
    except Exception as e:
        logger.error(f"whisper transcription failed for {path.name}: {e}", exc_info=True)
        return None


# ========== ORCHESTRATION ==========

def _quality_score(metrics: Dict, dnsmos: Optional[Dict], whisper: Optional[Dict]) -> float:
    score = 0.0
    if dnsmos:
        score += 10.0 * float(dnsmos.get("OVRL", 0.0))
        score += 5.0 * float(dnsmos.get("SIG", 0.0))
        score += 3.0 * float(dnsmos.get("BAK", 0.0))
    score += 0.5 * float(metrics.get("estimated_snr_db", 0.0))
    score -= 5.0 * float(metrics.get("clipping_pct", 0.0))
    if whisper:
        if whisper.get("avg_logprob") is not None:
            score += 15.0 * (float(whisper["avg_logprob"]) + 1.0)
        if whisper.get("mean_no_speech_prob") is not None:
            score -= 10.0 * float(whisper["mean_no_speech_prob"])
        score += 0.1 * float(whisper.get("word_count", 0))
    return round(score, 2)


def _evaluate_one(path: Path, use_whisper: bool, whisper_model: str) -> Dict:
    y, sr = _load_mono(path, 48000)
    metrics = acoustic_metrics(y, sr)
    dnsmos = dnsmos_score(y, sr)
    whisper = whisper_metrics(path, whisper_model) if use_whisper else None
    entry = {
        "file": path.name,
        "acoustic": metrics,
        "dnsmos": dnsmos,
        "whisper": whisper,
        "quality_score": _quality_score(metrics, dnsmos, whisper),
    }
    return entry


def _deltas(input_entry: Dict, output_entry: Dict) -> Dict:
    delta: Dict = {}
    in_ac, out_ac = input_entry["acoustic"], output_entry["acoustic"]
    for key in ("rms_dbfs", "estimated_snr_db", "spectral_tilt_db", "clipping_pct", "speech_active_pct"):
        if in_ac.get(key) is not None and out_ac.get(key) is not None:
            delta[key] = round(out_ac[key] - in_ac[key], 3)
    if input_entry.get("dnsmos") and output_entry.get("dnsmos"):
        for key in ("SIG", "BAK", "OVRL"):
            iv, ov = input_entry["dnsmos"].get(key), output_entry["dnsmos"].get(key)
            if iv is not None and ov is not None:
                delta[f"dnsmos_{key}"] = round(ov - iv, 3)
    if input_entry.get("whisper") and output_entry.get("whisper"):
        for key in ("word_count", "avg_logprob", "mean_no_speech_prob"):
            iv, ov = input_entry["whisper"].get(key), output_entry["whisper"].get(key)
            if iv is not None and ov is not None:
                delta[f"whisper_{key}"] = round(ov - iv, 3)
    delta["quality_score"] = round(output_entry["quality_score"] - input_entry["quality_score"], 2)
    return delta


def _print_table(input_entry: Dict, outputs: List[Dict]) -> None:
    def _fmt(entry: Dict) -> str:
        d = entry.get("dnsmos") or {}
        w = entry.get("whisper") or {}
        ac = entry["acoustic"]
        parts = [
            f"OVRL={d.get('OVRL', '-')}",
            f"SIG={d.get('SIG', '-')}",
            f"BAK={d.get('BAK', '-')}",
            f"SNR={ac.get('estimated_snr_db')}dB",
            f"LUFS={ac.get('lufs')}",
            f"clip={ac.get('clipping_pct')}%",
        ]
        if w:
            parts.append(f"words={w.get('word_count')}")
            parts.append(f"logprob={w.get('avg_logprob')}")
        return "  ".join(str(p) for p in parts)

    print("\n  ===== Validacao: entrada vs saidas =====")
    print(f"  [ENTRADA ] {input_entry['file']}  score={input_entry['quality_score']}")
    print(f"             {_fmt(input_entry)}")
    ranked = sorted(outputs, key=lambda e: e["quality_score"], reverse=True)
    for rank, entry in enumerate(ranked, 1):
        print(f"  [#{rank} SAIDA] {entry['file']}  score={entry['quality_score']}")
        print(f"             {_fmt(entry)}")
        delta = entry.get("delta", {})
        if delta:
            print(f"             delta: score={delta.get('quality_score')} SNR={delta.get('estimated_snr_db')} OVRL={delta.get('dnsmos_OVRL')}")
    if ranked:
        print(f"\n  Melhor variante: {ranked[0]['file']} (score={ranked[0]['quality_score']})")


def validate_files(input_path: Path, output_paths: List[Path], use_whisper: bool = False, whisper_model: str = "base", report_path: Optional[Path] = None) -> Dict:
    input_path = Path(input_path)
    t0 = time.time()
    logger.debug(f"Validating input={input_path.name} vs {len(output_paths)} output(s)")

    input_entry = _evaluate_one(input_path, use_whisper, whisper_model)
    output_entries: List[Dict] = []
    for out in output_paths:
        out = Path(out)
        if not out.exists():
            logger.warning(f"Output not found, skipping: {out}")
            continue
        try:
            entry = _evaluate_one(out, use_whisper, whisper_model)
            entry["delta"] = _deltas(input_entry, entry)
            output_entries.append(entry)
        except Exception as e:
            logger.error(f"Failed to evaluate {out.name}: {e}", exc_info=True)

    _print_table(input_entry, output_entries)

    ranked = sorted(output_entries, key=lambda e: e["quality_score"], reverse=True)
    report = {
        "input": input_entry,
        "outputs": output_entries,
        "ranking": [e["file"] for e in ranked],
        "best": ranked[0]["file"] if ranked else None,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if report_path:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.debug(f"Validation report written: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write validation report: {e}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic Audio Validate (entrada vs saida)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Audio de entrada (referencia)")
    parser.add_argument("outputs", nargs="*", help="Arquivos de saida a comparar")
    parser.add_argument("--outputs", "-O", dest="outputs_opt", type=str, default=None, help="Saidas separadas por virgula")
    parser.add_argument("--report", "-r", type=str, default=None, help="Caminho do JSON de relatorio")
    parser.add_argument("--whisper", action="store_true", help="Incluir intelig. via Whisper")
    parser.add_argument("--whisper-model", type=str, default="base", help="Modelo Whisper (base/small/medium/large-v3)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"  Entrada nao encontrada: {input_path}")
        return 1

    outputs: List[Path] = [Path(o).resolve() for o in args.outputs]
    if args.outputs_opt:
        outputs.extend(Path(o.strip()).resolve() for o in args.outputs_opt.split(",") if o.strip())
    if not outputs:
        print("  Nenhuma saida informada para comparar.")
        return 1

    report_path = Path(args.report).resolve() if args.report else input_path.with_name(f"{input_path.stem}_v2_validation.json")

    print("\n" + "=" * 60)
    print("  Forensic Audio Validate")
    print("=" * 60)
    try:
        validate_files(input_path, outputs, use_whisper=args.whisper, whisper_model=args.whisper_model, report_path=report_path)
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        print(f"\n  ERRO: {e}")
        return 1
    print(f"\n  Relatorio: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
