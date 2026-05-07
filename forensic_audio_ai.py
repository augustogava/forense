#!/usr/bin/env python3
"""
Forensic Audio AI Enhancement

Processamento de áudio forense usando IA (Demucs + DeepFilterNet).
Separa vozes do fundo e realça a fala sem introduzir eco ou artefatos.
"""

import argparse
import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.ndimage import uniform_filter1d


def _sf_load(path: str) -> tuple:
    import tempfile, imageio_ffmpeg
    p = Path(path)
    if p.suffix.lower() in (".m4a", ".aac", ".mp3", ".ogg", ".wma", ".flac"):
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_wav = Path(tempfile.gettempdir()) / f"_sf_load_{p.stem}.wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(p), "-ar", "44100", "-ac", "2", str(tmp_wav)],
            capture_output=True, check=True,
        )
        data, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=True)
        tmp_wav.unlink(missing_ok=True)
    else:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    return waveform, sr

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}

_ffmpeg_path = None
try:
    import imageio_ffmpeg
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    for candidate in ["ffmpeg", "ffmpeg.exe"]:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            _ffmpeg_path = candidate
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass


def check_dependencies() -> Dict[str, bool]:
    status = {}
    try:
        from demucs.pretrained import get_model  # noqa: F401
        from demucs.apply import apply_model  # noqa: F401
        status["demucs"] = True
    except ImportError:
        status["demucs"] = False

    try:
        from df import init_df  # noqa: F401
        status["deepfilter"] = True
    except ImportError:
        status["deepfilter"] = False

    try:
        from clearvoice import ClearVoice  # noqa: F401
        status["clearvoice"] = True
    except ImportError:
        status["clearvoice"] = False

    return status


def print_dependency_status(status: Dict[str, bool]) -> bool:
    print("\n  Módulos de IA disponíveis:")
    labels = {
        "demucs": "Demucs (separação vocal)",
        "deepfilter": "DeepFilterNet (realce fala)",
        "clearvoice": "ClearVoice (fallback)",
    }
    for key, label in labels.items():
        tag = "OK" if status[key] else "NÃO INSTALADO"
        print(f"    {label:38s} {tag}")

    missing = []
    if not status["demucs"]:
        missing.append("pip install demucs")
    if not status["deepfilter"] and not status["clearvoice"]:
        missing.append("pip install deepfilternet")

    if missing:
        print("\n  Para instalar dependências faltantes:")
        for m in missing:
            print(f"    {m}")

    has_any = status["demucs"] or status["deepfilter"] or status["clearvoice"]
    if not has_any:
        print("\n  ERRO: Nenhum módulo de IA disponível. Instale pelo menos um.")
    return has_any


def find_audio_files(search_dirs: List[Path]) -> List[Path]:
    files = []
    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(d.glob(f"*{ext}"))
            files.extend(d.glob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.name.lower())


def interactive_select_input(base_dir: Path) -> Optional[Path]:
    search_dirs = [base_dir / "to_analyse", base_dir]
    audio_files = find_audio_files(search_dirs)

    if not audio_files:
        print("  Nenhum arquivo de áudio encontrado.")
        return None

    print("\n  Arquivos de áudio disponíveis:\n")
    for idx, f in enumerate(audio_files, 1):
        try:
            rel = f.relative_to(base_dir)
        except ValueError:
            rel = f
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {idx:3d}. {rel}  ({size_mb:.1f} MB)")

    while True:
        try:
            choice = input("\n  Escolha o número do arquivo (ou 'q' para sair): ").strip()
            if choice.lower() == "q":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(audio_files):
                return audio_files[idx]
            print("  Número inválido.")
        except ValueError:
            print("  Entrada inválida.")


def interactive_select_output(base_dir: Path) -> Path:
    default = base_dir / "audio_processed"
    choice = input(f"\n  Diretório de saída [{default}]: ").strip()
    if choice:
        return Path(choice)
    return default


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def _to_wav(input_path: Path, output_wav: Path, sr: int = 44100, mono: bool = True):
    ffmpeg = _ffmpeg_path or "ffmpeg"
    ac = ["1"] if mono else ["2"]
    cmd = [ffmpeg, "-y", "-i", str(input_path), "-ar", str(sr), "-ac", *ac, str(output_wav)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")


def save_as_mp3(waveform: torch.Tensor, sr: int, output_path: Path) -> Path:
    output_path = output_path.with_suffix(".mp3")
    temp_wav = output_path.parent / f"_tmp_mp3_{output_path.stem}.wav"
    try:
        torchaudio.save(str(temp_wav), waveform.cpu(), sr, encoding="PCM_S", bits_per_sample=16)
        ffmpeg = _ffmpeg_path or "ffmpeg"
        cmd = [ffmpeg, "-y", "-i", str(temp_wav), "-c:a", "libmp3lame", "-b:a", "192k", str(output_path)]
        subprocess.run(cmd, capture_output=True, check=True)
    except Exception as e:
        logger.error(f"MP3 encoding failed, keeping WAV: {e}")
        output_path = output_path.with_suffix(".wav")
        if temp_wav.exists():
            temp_wav.rename(output_path)
        return output_path
    finally:
        if temp_wav.exists():
            temp_wav.unlink()
    return output_path


# ---------------------------------------------------------------------------
# Stage 1 – Demucs voice separation
# ---------------------------------------------------------------------------

def stage_demucs(input_path: Path, work_dir: Path) -> Optional[Path]:
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import convert_audio, save_audio
    except ImportError:
        print("    Demucs não disponível, pulando separação vocal...")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    [demucs] Carregando modelo htdemucs ({device})...")
    model = get_model("htdemucs")
    model.to(device)

    print("    [demucs] Carregando áudio...")
    waveform, sr = _sf_load(str(input_path))
    wav = convert_audio(waveform, sr, model.samplerate, model.audio_channels)

    print("    [demucs] Separando vozes...")
    ref = wav.mean(0)
    wav_centered = wav - ref.mean()
    wav_scaled = wav_centered / (ref.std() + 1e-8)

    sources = apply_model(model, wav_scaled[None], device=device, progress=True)[0]
    sources = sources * ref.std() + ref.mean()

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx]

    vocals_path = work_dir / "_stage1_vocals.wav"
    save_audio(vocals.cpu(), str(vocals_path), model.samplerate)
    print(f"    [demucs] Vocals isolados ({model.samplerate} Hz)")
    return vocals_path


# ---------------------------------------------------------------------------
# Stage 2 – AI speech enhancement
# ---------------------------------------------------------------------------

def stage_deepfilter(input_wav: Path, work_dir: Path) -> Optional[Path]:
    try:
        from df import init_df
        from df.enhance import enhance, load_audio, save_audio
    except ImportError:
        return None

    print("    [deepfilter] Carregando modelo DeepFilterNet...")
    model, df_state, _ = init_df(config_allow_defaults=True)
    target_sr = df_state.sr()

    print(f"    [deepfilter] Realçando fala ({target_sr} Hz)...")
    audio, _ = load_audio(str(input_wav), sr=target_sr)
    enhanced = enhance(model, df_state, audio)

    output_path = work_dir / "_stage2_enhanced.wav"
    save_audio(str(output_path), enhanced, sr=target_sr)
    print("    [deepfilter] Fala realçada")
    return output_path


def stage_clearvoice(input_wav: Path, work_dir: Path) -> Optional[Path]:
    try:
        from clearvoice import ClearVoice
        import soundfile as sf
    except ImportError:
        return None

    print("    [clearvoice] Carregando modelo MossFormer2_SE_48K...")
    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])

    print("    [clearvoice] Realçando fala...")
    enhanced_audio = cv(str(input_wav))

    output_path = work_dir / "_stage2_enhanced.wav"
    if enhanced_audio is not None:
        enhanced_np = enhanced_audio if isinstance(enhanced_audio, np.ndarray) else enhanced_audio.cpu().numpy()
        if enhanced_np.ndim == 1:
            enhanced_np = enhanced_np[np.newaxis, :]
        sr = 48000
        sf.write(str(output_path), enhanced_np.T if enhanced_np.shape[0] <= 2 else enhanced_np, sr)
        print("    [clearvoice] Fala realçada")
        return output_path

    print("    [clearvoice] Sem resultado")
    return None


def stage_enhance(input_wav: Path, work_dir: Path, deps: Dict[str, bool]) -> Optional[Path]:
    if deps["deepfilter"]:
        result = stage_deepfilter(input_wav, work_dir)
        if result:
            return result

    if deps["clearvoice"]:
        result = stage_clearvoice(input_wav, work_dir)
        if result:
            return result

    print("    Nenhum módulo de realce disponível, pulando etapa 2...")
    return None


# ---------------------------------------------------------------------------
# Stage 3 – Normalization (minimal traditional DSP)
# ---------------------------------------------------------------------------

def gentle_compress(audio: np.ndarray, sr: int) -> np.ndarray:
    threshold_db = -20
    ratio = 3.0
    frame_len = int(sr * 0.03)
    hop = frame_len // 2
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)

    rms = np.array([
        np.sqrt(np.mean(audio[i * hop : i * hop + frame_len] ** 2) + 1e-10)
        for i in range(n_frames)
    ])

    peak_env = np.max(rms) if len(rms) > 0 else 1.0
    rms_norm = rms / peak_env if peak_env > 0 else rms

    threshold = 10 ** (threshold_db / 20)
    frame_gain = np.ones(n_frames)
    above = rms_norm > threshold
    if np.any(above):
        excess = rms_norm[above] / threshold
        compressed = threshold * (excess ** (1.0 / ratio))
        frame_gain[above] = compressed / (rms_norm[above] + 1e-10)

    frame_gain = uniform_filter1d(frame_gain.astype(np.float64), size=7)

    sample_gain = np.interp(
        np.arange(len(audio)),
        np.arange(n_frames) * hop + hop // 2,
        frame_gain,
    )
    return audio * sample_gain


def loudness_normalize(audio: np.ndarray, target_db: float = -6.0) -> np.ndarray:
    target_rms = 10 ** (target_db / 20)
    current_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    if current_rms > 0:
        gain = min(target_rms / current_rms, 30.0)
        return audio * gain
    return audio


def peak_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        audio = np.tanh(audio / peak * 1.5) * ceiling
    return np.clip(audio, -1.0, 1.0)


def agc_boost(audio: np.ndarray, sr: int, max_gain: float = 6.0) -> np.ndarray:
    frame_len = int(sr * 0.05)
    hop = frame_len // 2
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)

    rms = np.array([
        np.sqrt(np.mean(audio[i * hop : i * hop + frame_len] ** 2) + 1e-10)
        for i in range(n_frames)
    ])

    silence_thresh = np.percentile(rms, 10)
    speech_rms = rms[rms > silence_thresh * 2]
    if len(speech_rms) == 0:
        return audio

    target = np.percentile(speech_rms, 70)
    frame_gain = np.ones(n_frames)
    active = rms > silence_thresh * 1.5
    frame_gain[active] = np.clip(target / (rms[active] + 1e-10), 1.0, max_gain)
    frame_gain = uniform_filter1d(frame_gain.astype(np.float64), size=7)

    sample_gain = np.interp(
        np.arange(len(audio)),
        np.arange(n_frames) * hop + hop // 2,
        frame_gain,
    )
    return audio * sample_gain


def _load_for_normalize(wav_path: Path) -> tuple:
    waveform, sr = _sf_load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.numpy().squeeze(), sr


def stage_normalize(input_wav: Path, output_dir: Path, base_name: str) -> Path:
    audio, sr = _load_for_normalize(input_wav)
    audio = gentle_compress(audio, sr)
    audio = loudness_normalize(audio, target_db=-6.0)
    audio = peak_limit(audio)
    out = torch.from_numpy(audio).unsqueeze(0).float()
    return save_as_mp3(out, sr, output_dir / f"{base_name}_ai_enhanced.mp3")


def stage_normalize_boosted(input_wav: Path, output_dir: Path, base_name: str) -> Path:
    audio, sr = _load_for_normalize(input_wav)
    audio = agc_boost(audio, sr, max_gain=6.0)
    audio = gentle_compress(audio, sr)
    audio = loudness_normalize(audio, target_db=-4.0)
    audio = peak_limit(audio)
    out = torch.from_numpy(audio).unsqueeze(0).float()
    return save_as_mp3(out, sr, output_dir / f"{base_name}_ai_enhanced_boosted.mp3")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_audio(input_path: Path, output_dir: Path, deps: Dict[str, bool]) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_path.stem
    work_dir = output_dir / f"_work_{base_name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    results: Dict = {"source": str(input_path), "files": []}

    print(f"\n  Processando: {input_path.name}")
    print(f"  Saída: {output_dir}\n")

    # Prepare a WAV copy for stages that need it
    if input_path.suffix.lower() != ".wav":
        raw_wav = work_dir / f"_raw.wav"
        _to_wav(input_path, raw_wav)
    else:
        raw_wav = input_path

    current_wav = raw_wav

    # --- Stage 1: Demucs ---
    if deps["demucs"]:
        print("  ── Etapa 1: Separação de vozes (Demucs) ──")
        t1 = time.time()
        vocals_wav = stage_demucs(input_path, work_dir)
        if vocals_wav and vocals_wav.exists():
            current_wav = vocals_wav
            wf, sr = _sf_load(str(vocals_wav))
            if wf.shape[0] > 1:
                wf = wf.mean(dim=0, keepdim=True)
            demucs_mp3 = save_as_mp3(wf, sr, output_dir / f"{base_name}_demucs_vocals.mp3")
            results["files"].append(demucs_mp3.name)
            print(f"    Concluído em {time.time() - t1:.0f}s\n")
        else:
            print("    Separação falhou, usando áudio original\n")
    else:
        print("  ── Etapa 1: Demucs não instalado, pulando ──\n")

    # --- Stage 2: AI Enhancement ---
    has_enhancer = deps["deepfilter"] or deps["clearvoice"]
    if has_enhancer:
        engine = "DeepFilterNet" if deps["deepfilter"] else "ClearVoice"
        print(f"  ── Etapa 2: Realce de fala ({engine}) ──")
        t2 = time.time()
        enhanced_wav = stage_enhance(current_wav, work_dir, deps)
        if enhanced_wav and enhanced_wav.exists():
            current_wav = enhanced_wav
            print(f"    Concluído em {time.time() - t2:.0f}s\n")
        else:
            print("    Realce falhou, continuando com áudio anterior\n")
    else:
        print("  ── Etapa 2: Nenhum realce IA disponível, pulando ──\n")

    # --- Stage 3: Normalization ---
    print("  ── Etapa 3: Normalização e compressão ──")
    t3 = time.time()

    final_path = stage_normalize(current_wav, output_dir, base_name)
    results["files"].append(final_path.name)
    print(f"    -> {final_path.name}")

    boosted_path = stage_normalize_boosted(current_wav, output_dir, base_name)
    results["files"].append(boosted_path.name)
    print(f"    -> {boosted_path.name}")

    print(f"    Concluído em {time.time() - t3:.0f}s")

    # Cleanup work directory
    try:
        for f in work_dir.iterdir():
            f.unlink(missing_ok=True)
        work_dir.rmdir()
    except Exception:
        pass

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = round(elapsed, 1)

    print(f"\n  Processamento concluído em {elapsed:.0f}s!")
    print("  Arquivos gerados:")
    for fname in results["files"]:
        fpath = output_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"    -> {fname} ({size_mb:.1f} MB)")

    report_path = output_dir / f"{base_name}_ai_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic Audio AI Enhancement")
    parser.add_argument("--input", "-i", type=str, help="Arquivo de áudio de entrada")
    parser.add_argument("--output", "-o", type=str, help="Diretório de saída")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 60)
    print("  Forensic Audio AI Enhancement")
    print("=" * 60)

    deps = check_dependencies()
    if not print_dependency_status(deps):
        return 1

    if args.input:
        input_path = Path(args.input).resolve()
        output_dir = Path(args.output).resolve() if args.output else base_dir / "audio_processed"
    else:
        input_path = interactive_select_input(base_dir)
        if not input_path:
            print("  Cancelado.")
            return 0
        output_dir = interactive_select_output(base_dir)

    if not input_path.exists():
        print(f"  ERRO: Arquivo não encontrado: {input_path}")
        return 1

    try:
        process_audio(input_path, output_dir, deps)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n  ERRO: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
