#!/usr/bin/env python3
"""
Forensic Audio Transcriber

Transcrição de áudio forense usando Whisper (OpenAI).
Detecta fala em português e gera JSON com timestamps por segmento.
"""

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import whisper

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}

NEEDS_CONVERSION = {".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}

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


def _convert_to_wav(input_path: Path) -> Path:
    if not _ffmpeg_path:
        logger.error("ffmpeg not found, cannot convert audio")
        sys.exit(1)

    tmp_wav = Path(tempfile.gettempdir()) / f"whisper_{input_path.stem}.wav"
    cmd = [
        _ffmpeg_path, "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(tmp_wav),
    ]
    logger.debug(f"Converting to WAV 16kHz mono: {input_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg conversion failed: {result.stderr.strip()}")
        sys.exit(1)
    return tmp_wav


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
    "sexo, buceta, pau, porra, caralho, foder, puta, cu, "
    "gozar, chupar, estupro, abuso, gemido, pelada"
)

_EXPLICIT_KEYWORDS = {
    "sexo", "sexual", "transar", "transou", "transando",
    "foder", "fuder", "fodeu", "fudeu", "fodendo", "fudendo",
    "buceta", "boceta", "xereca", "xoxota", "ppk", "gozando", "goza", "gozou", "enfia",
    "pau", "pica", "rola", "cacete", "pinto",
    "chupar", "chupou", "chupando", "mamar", "mamou", "mamando", "boquete",
    "gozar", "gozou", "gozando",
    "porra", "caralho", "merda", "puta", "putaria", "mete", "dedo", "dedos", "deda", "dedas", "dedou", "dedou", "dedo", "dedos", "deda", "dedas", "dedou", "dedou", 
    "cu", "bunda", "rabo",
    "pelada", "pelado", "nua", "nu",
    "estupro", "estuprar", "estuprou", "estuprando",
    "abuso", "abusar", "abusou", "abusando",
    "assédio", "assediar", "assediou",
    "molestar", "molestou", "molestando",
    "violência", "violentar", "violentou",
    "gemido", "gemendo", "gemer", "gemeu", "geme"
    "tesão", "excitado", "excitada",
    "masturbar", "masturbou", "masturbando", "masturbação",
    "penetrar", "penetrou", "penetração",
    "ejacular", "ejaculou", "ejaculação",
    "oral", "vaginal", "anal",
}


def _flag_explicit(text: str) -> bool:
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    return bool(words & _EXPLICIT_KEYWORDS)


def _extract_speech_segments(
    audio_np: np.ndarray,
    sr: int,
    vad_model,
    get_speech_ts,
) -> List[dict]:
    audio_tensor = torch.from_numpy(audio_np).float()
    timestamps = get_speech_ts(
        audio_tensor,
        vad_model,
        sampling_rate=sr,
        threshold=0.3,
        min_speech_duration_ms=500,
        min_silence_duration_ms=300,
    )
    vad_model.reset_states()

    if not timestamps:
        return []

    merged = [timestamps[0].copy()]
    merge_gap = int(sr * 1.5)
    for ts in timestamps[1:]:
        if ts["start"] - merged[-1]["end"] < merge_gap:
            merged[-1]["end"] = ts["end"]
        else:
            merged.append(ts.copy())

    return merged


def transcribe_file(
    model: whisper.Whisper,
    audio_path: Path,
    output_dir: Path,
    word_timestamps: bool = False,
    model_name: str = "large-v3",
    vad_model=None,
    get_speech_ts=None,
) -> Optional[Path]:
    t_start = time.time()
    sr = 16000
    print(f"\n    Transcrevendo: {audio_path.name}")

    wav_path = _convert_to_wav(audio_path)
    try:
        import wave as _wave
        with _wave.open(str(wav_path), "rb") as wf:
            audio_np = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path.name}: {e}")
        return None
    finally:
        wav_path.unlink(missing_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{audio_path.stem}_transcription.json"

    if vad_model is not None and get_speech_ts is not None:
        print("    [VAD] Detectando segmentos com fala...")
        speech_segments = _extract_speech_segments(audio_np, sr, vad_model, get_speech_ts)
        if not speech_segments:
            print("    [VAD] Nenhuma fala detectada")
            output_data = {
                "source_file": audio_path.name,
                "language": "pt",
                "model": model_name,
                "vad": "no_speech_detected",
                "segments": {},
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            elapsed = time.time() - t_start
            print(f"    Concluído em {elapsed:.0f}s — 0 segmentos (sem fala) -> {json_path.name}")
            return json_path

        total_speech = sum(s["end"] - s["start"] for s in speech_segments) / sr
        total_audio = len(audio_np) / sr
        print(f"    [VAD] {len(speech_segments)} segmentos com fala ({total_speech:.0f}s de {total_audio:.0f}s)")
    else:
        speech_segments = [{"start": 0, "end": len(audio_np)}]

    segments_dict = {}
    explicit_segments = {}
    words_list = []
    prev_text = None
    repeat_count = 0

    for chunk_idx, chunk in enumerate(speech_segments):
        chunk_audio = audio_np[chunk["start"]:chunk["end"]]
        offset_sec = chunk["start"] / sr

        try:
            result = model.transcribe(
                chunk_audio,
                language="pt",
                word_timestamps=word_timestamps,
                verbose=False,
                condition_on_previous_text=False,
                compression_ratio_threshold=1.8,
                no_speech_threshold=0.5,
                initial_prompt=_FORENSIC_PROMPT,
            )
        except Exception as e:
            logger.error(f"Transcription failed for chunk {chunk_idx} of {audio_path.name}: {e}")
            continue

        for seg in result.get("segments", []):
            real_start = seg["start"] + offset_sec
            real_end = seg["end"] + offset_sec
            key = f"{_fmt_time(real_start)},{_fmt_time(real_end)}"
            text = seg["text"].strip()
            if not text:
                continue
            if text == prev_text:
                repeat_count += 1
                if repeat_count >= 2:
                    continue
            else:
                repeat_count = 0
            prev_text = text
            segments_dict[key] = text

            if _flag_explicit(text):
                explicit_segments[key] = text

            if word_timestamps:
                for w in seg.get("words", []):
                    words_list.append({
                        "start": round(w["start"] + offset_sec, 2),
                        "end": round(w["end"] + offset_sec, 2),
                        "text": w["word"].strip(),
                    })

    output_data = {
        "source_file": audio_path.name,
        "language": "pt",
        "model": model_name,
        "segments": segments_dict,
    }
    if explicit_segments:
        output_data["explicit_segments"] = explicit_segments
        output_data["explicit_count"] = len(explicit_segments)
    if word_timestamps and words_list:
        output_data["words"] = words_list

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    seg_count = len(segments_dict)
    print(f"    Concluído em {elapsed:.0f}s — {seg_count} segmentos -> {json_path.name}")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic Audio Transcriber (Whisper)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Arquivo de áudio ou pasta com áudios")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Diretório de saída para JSONs (padrão: transcriptions/)")
    parser.add_argument("--model", "-m", type=str, default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Modelo Whisper (padrão: large-v3)")
    parser.add_argument("--word-timestamps", action="store_true",
                        help="Incluir timestamps por palavra")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Forensic Audio Transcriber")
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

    print(f"\n  Arquivos encontrados: {len(audio_files)}")
    for af in audio_files:
        print(f"    - {af.name}")

    if not _ffmpeg_path:
        print("  AVISO: ffmpeg não encontrado. Formatos m4a/aac/wma podem falhar.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Carregando Silero VAD...")
    vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_ts = vad_utils[0]
    print("  Silero VAD carregado")

    print(f"\n  Carregando modelo Whisper '{args.model}' ({device})...")
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
            result = transcribe_file(model, af, output_dir, args.word_timestamps, args.model, vad_model, get_speech_ts)
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
