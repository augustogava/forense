#!/usr/bin/env python3
"""
Audio Crop Tool - Recorta arquivos de áudio por tempo inicial e final.
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path

import imageio_ffmpeg

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def _get_duration(input_path: str) -> float:
    import re
    cmd = [FFMPEG_EXE, "-hide_banner", "-i", input_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    info = (result.stderr or "") + (result.stdout or "")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", info)
    if m:
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return hh * 3600 + mm * 60 + ss
    logger.error(f"Could not determine duration: {info.strip()}")
    sys.exit(1)


def crop_audio(input_path: str, output_path: str, start_sec: float, end_sec: float) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Arquivo não encontrado: {input_path}")
        sys.exit(1)

    total_duration = _get_duration(input_path)
    logger.debug(f"Audio loaded - duration={total_duration:.2f}s")

    if start_sec < 0:
        logger.error("Tempo inicial não pode ser negativo.")
        sys.exit(1)

    if end_sec > total_duration:
        logger.warning(f"Tempo final ({end_sec}s) excede a duração ({total_duration:.2f}s). Usando duração total.")
        end_sec = total_duration

    if start_sec >= end_sec:
        logger.error(f"Tempo inicial ({start_sec}s) deve ser menor que o tempo final ({end_sec}s).")
        sys.exit(1)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    duration = end_sec - start_sec
    cmd = [
        FFMPEG_EXE, "-y",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    logger.debug(f"Running ffmpeg stream copy: {start_sec}s - {end_sec}s")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.info(f"Áudio recortado com sucesso: {duration:.2f}s salvo em {output_path}")


def split_audio(input_path: str, interval_min: float, output_dir: str = None) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Arquivo não encontrado: {input_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = str(input_file.parent / "audio_split")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ext = input_file.suffix.lower()
    total_duration = _get_duration(input_path)
    interval_sec = interval_min * 60
    logger.info(f"Duração total: {total_duration:.2f}s | Intervalo: {interval_min} min ({interval_sec:.0f}s)")

    part = 1
    start_sec = 0.0
    while start_sec < total_duration:
        seg_duration = min(interval_sec, total_duration - start_sec)
        out_name = f"{input_file.stem}_part{part:03d}{ext}"
        out_file = out_path / out_name

        cmd = [
            FFMPEG_EXE, "-y",
            "-ss", str(start_sec),
            "-i", input_path,
            "-t", str(seg_duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg failed on part {part}: {result.stderr.strip()}")
            sys.exit(1)

        logger.info(f"Parte {part}: {start_sec:.0f}s - {start_sec + seg_duration:.0f}s -> {out_file}")

        start_sec += interval_sec
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
