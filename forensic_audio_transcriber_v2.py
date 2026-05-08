#!/usr/bin/env python3
"""
Forensic Audio Transcriber V2 — faster-whisper

4-8x faster than openai-whisper via CTranslate2.
Built-in Silero VAD and hallucination_silence_threshold.

pip install faster-whisper
"""

import argparse
import io
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}

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


def _is_hallucination(seg) -> bool:
    text = seg.text.strip() if hasattr(seg, "text") else ""
    if not text:
        return True

    no_speech = getattr(seg, "no_speech_prob", 0.0)
    logprob = getattr(seg, "avg_logprob", 0.0)
    compression = getattr(seg, "compression_ratio", 1.0)

    if no_speech > 0.6 and logprob < -1.0:
        return True
    if compression > 2.4 and logprob < -1.0:
        return True

    words = [w for w in text.lower().replace(",", " ").replace(".", " ").split() if w]
    unique = set(words)
    if len(words) >= 3 and len(unique) <= 2 and unique.issubset(_EXPLICIT_KEYWORDS):
        return True

    return False


def _flag_explicit(text: str) -> bool:
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    return bool(words & _EXPLICIT_KEYWORDS)


def transcribe_file(
    model: WhisperModel,
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
        segments_gen, info = model.transcribe(
            str(audio_path),
            language="pt",
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            temperature=0.0,
            hallucination_silence_threshold=2.0,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.3,
                min_speech_duration_ms=500,
                min_silence_duration_ms=300,
            ),
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

    for seg in segments_gen:
        text = seg.text.strip()
        if not text:
            continue

        if _is_hallucination(seg):
            logger.debug(
                f"Hallucination filtered (nsp={getattr(seg, 'no_speech_prob', 0):.2f} "
                f"logp={getattr(seg, 'avg_logprob', 0):.2f} "
                f"cr={getattr(seg, 'compression_ratio', 0):.2f}): '{text}'"
            )
            continue

        if text == prev_text:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            repeat_count = 0
        prev_text = text

        key = f"{_fmt_time(seg.start)},{_fmt_time(seg.end)}"
        segments_dict[key] = text

        if _flag_explicit(text):
            explicit_segments[key] = text

        if word_timestamps and seg.words:
            for w in seg.words:
                words_list.append({
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "text": w.word.strip(),
                    "probability": round(w.probability, 3),
                })

    output_data = {
        "source_file": audio_path.name,
        "language": info.language,
        "model": model_name,
        "duration": round(info.duration, 1),
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
    parser = argparse.ArgumentParser(description="Forensic Audio Transcriber V2 (faster-whisper)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Arquivo de áudio ou pasta com áudios")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Diretório de saída para JSONs (padrão: transcriptions/)")
    parser.add_argument("--model", "-m", type=str, default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"],
                        help="Modelo Whisper (padrão: large-v3)")
    parser.add_argument("--word-timestamps", action="store_true",
                        help="Incluir timestamps por palavra")
    parser.add_argument("--compute-type", type=str, default="float16",
                        choices=["float16", "float32", "int8", "int8_float16"],
                        help="Tipo de computação (padrão: float16)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Forensic Audio Transcriber V2 (faster-whisper)")
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
    compute_type = args.compute_type if device == "cuda" else "float32"

    print(f"\n  Carregando modelo faster-whisper '{args.model}' ({device}, {compute_type})...")
    t_load = time.time()
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
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
