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

import torch
import whisper

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}

NEEDS_CONVERSION = {".m4a", ".aac", ".wma", ".opus"}

_ffmpeg_path: Optional[str] = None
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


def transcribe_file(
    model: whisper.Whisper,
    audio_path: Path,
    output_dir: Path,
    word_timestamps: bool = False,
    model_name: str = "large-v3",
) -> Optional[Path]:
    t_start = time.time()
    print(f"\n    Transcrevendo: {audio_path.name}")

    needs_convert = audio_path.suffix.lower() in NEEDS_CONVERSION
    wav_path = _convert_to_wav(audio_path) if needs_convert else audio_path
    tmp_created = needs_convert

    try:
        logger.debug(f"Running whisper transcribe on {wav_path}")
        result = model.transcribe(
            str(wav_path),
            language="pt",
            word_timestamps=word_timestamps,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path.name}: {e}")
        if tmp_created:
            wav_path.unlink(missing_ok=True)
        return None
    finally:
        if tmp_created:
            wav_path.unlink(missing_ok=True)

    segments_dict = {}
    for seg in result.get("segments", []):
        key = f"{_fmt_time(seg['start'])},{_fmt_time(seg['end'])}"
        text = seg["text"].strip()
        if text:
            segments_dict[key] = text

    output_data = {
        "source_file": audio_path.name,
        "language": result.get("language", "pt"),
        "model": model_name,
        "segments": segments_dict,
    }

    if word_timestamps:
        words_list = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words_list.append({
                    "start": round(w["start"], 2),
                    "end": round(w["end"], 2),
                    "text": w["word"].strip(),
                })
        if words_list:
            output_data["words"] = words_list

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{audio_path.stem}_transcription.json"
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
    print(f"\n  Carregando modelo Whisper '{args.model}' ({device})...")
    t_load = time.time()
    model = whisper.load_model(args.model, device=device)
    print(f"  Modelo carregado em {time.time() - t_load:.0f}s")

    t_total = time.time()
    success = 0
    failed = 0

    for idx, af in enumerate(audio_files, 1):
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
    print(f"  Sucesso: {success} | Falhas: {failed}")
    print(f"  Saída: {output_dir}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
