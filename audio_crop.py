#!/usr/bin/env python3
"""
Audio Crop Tool - Recorta arquivos de áudio por tempo inicial e final.
"""

import argparse
import sys
import logging
from pathlib import Path

import librosa
import soundfile as sf

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crop_audio(input_path: str, output_path: str, start_sec: float, end_sec: float) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Arquivo não encontrado: {input_path}")
        sys.exit(1)

    logger.debug(f"Loading audio file: {input_path}")
    y, sr = librosa.load(input_path, sr=None, mono=False)

    total_duration = librosa.get_duration(y=y, sr=sr)
    logger.debug(f"Audio loaded - sample_rate={sr}, duration={total_duration:.2f}s")

    if start_sec < 0:
        logger.error("Tempo inicial não pode ser negativo.")
        sys.exit(1)

    if end_sec > total_duration:
        logger.warning(f"Tempo final ({end_sec}s) excede a duração ({total_duration:.2f}s). Usando duração total.")
        end_sec = total_duration

    if start_sec >= end_sec:
        logger.error(f"Tempo inicial ({start_sec}s) deve ser menor que o tempo final ({end_sec}s).")
        sys.exit(1)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    if y.ndim == 1:
        y_cropped = y[start_sample:end_sample]
    else:
        y_cropped = y[:, start_sample:end_sample]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Writing cropped audio: {output_path} ({start_sec}s - {end_sec}s)")
    sf.write(output_path, y_cropped.T if y_cropped.ndim > 1 else y_cropped, sr)

    cropped_duration = end_sec - start_sec
    logger.info(f"Áudio recortado com sucesso: {cropped_duration:.2f}s salvo em {output_path}")


def split_audio(input_path: str, interval_min: float, output_dir: str = None) -> None:
    from pydub import AudioSegment

    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Arquivo não encontrado: {input_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = str(input_file.parent / "audio_split")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ext = input_file.suffix.lower().lstrip(".")
    format_map = {"m4a": "m4a", "mp3": "mp3", "wav": "wav", "ogg": "ogg", "flac": "flac"}
    fmt = format_map.get(ext, ext)

    logger.debug(f"Loading audio file: {input_path} (format={fmt})")
    audio = AudioSegment.from_file(str(input_file), format=fmt)

    total_ms = len(audio)
    total_sec = total_ms / 1000.0
    interval_ms = int(interval_min * 60 * 1000)
    logger.info(f"Duração total: {total_sec:.2f}s | Intervalo: {interval_min} min ({interval_ms}ms)")

    part = 1
    start_ms = 0
    while start_ms < total_ms:
        end_ms = min(start_ms + interval_ms, total_ms)
        chunk = audio[start_ms:end_ms]

        out_name = f"{input_file.stem}_part{part:03d}.mp3"
        out_file = out_path / out_name

        chunk.export(str(out_file), format="mp3", bitrate="192k")
        logger.info(f"Parte {part}: {start_ms/1000:.0f}s - {end_ms/1000:.0f}s -> {out_file}")

        start_ms = end_ms
        part += 1

    logger.info(f"Split concluído: {part - 1} partes em {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Recorta arquivos de áudio por tempo.")
    subparsers = parser.add_subparsers(dest="command")

    crop_parser = subparsers.add_parser("crop", help="Recorta trecho específico")
    crop_parser.add_argument("-i", "--input", required=True, help="Caminho do arquivo de áudio de entrada")
    crop_parser.add_argument("-o", "--output", required=True, help="Caminho do arquivo de áudio de saída")
    crop_parser.add_argument("-s", "--start", type=float, required=True, help="Segundo inicial do recorte")
    crop_parser.add_argument("-e", "--end", type=float, required=True, help="Segundo final do recorte")

    split_parser = subparsers.add_parser("split", help="Divide áudio em partes iguais por intervalo")
    split_parser.add_argument("-i", "--input", required=True, help="Caminho do arquivo de áudio de entrada")
    split_parser.add_argument("-m", "--minutes", type=float, required=True, help="Intervalo em minutos para cada parte")
    split_parser.add_argument("-o", "--output-dir", default=None, help="Pasta de saída (padrão: audio_split/)")

    args = parser.parse_args()

    if args.command == "crop":
        crop_audio(args.input, args.output, args.start, args.end)
    elif args.command == "split":
        split_audio(args.input, args.minutes, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
