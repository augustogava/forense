#!/usr/bin/env python3
"""
Image Recompressor

Recompresses all images in processed_raw_* folders to reduce file size.
Uses JPEG quality 95 for optimal balance between quality and size.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import cv2

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
JPEG_QUALITY = 95
MAX_WORKERS = 8


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def recompress_image(image_path: Path) -> Tuple[Path, float, float, bool]:
    """
    Recompress a single image.
    
    Returns:
        Tuple of (path, original_size_mb, new_size_mb, success)
    """
    original_size = get_file_size_mb(image_path)
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to read: {image_path.name}")
            return (image_path, original_size, original_size, False)
        
        temp_path = image_path.with_suffix('.tmp.jpg')
        cv2.imwrite(str(temp_path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        
        new_size = get_file_size_mb(temp_path)
        
        if new_size < original_size:
            os.replace(str(temp_path), str(image_path))
            return (image_path, original_size, new_size, True)
        else:
            temp_path.unlink()
            return (image_path, original_size, original_size, False)
            
    except Exception as e:
        logger.error(f"Error recompressing {image_path.name}: {e}")
        temp_path = image_path.with_suffix('.tmp.jpg')
        if temp_path.exists():
            temp_path.unlink()
        return (image_path, original_size, original_size, False)


def process_folder(folder_path: Path, max_workers: int = MAX_WORKERS) -> Tuple[int, float, float]:
    """
    Process all images in a folder.
    
    Returns:
        Tuple of (files_processed, total_original_mb, total_new_mb)
    """
    image_files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        logger.warning(f"No images found in {folder_path.name}")
        return (0, 0.0, 0.0)
    
    total = len(image_files)
    print(f"\nProcessando {total} imagens em {folder_path.name}...")
    
    total_original = 0.0
    total_new = 0.0
    processed = 0
    reduced = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(recompress_image, img_path): img_path
            for img_path in image_files
        }
        
        for future in as_completed(futures):
            path, original_size, new_size, success = future.result()
            total_original += original_size
            total_new += new_size
            processed += 1
            
            if success and new_size < original_size:
                reduced += 1
                reduction = ((original_size - new_size) / original_size) * 100
                logger.debug(f"[OK] {path.name}: {original_size:.2f}MB -> {new_size:.2f}MB (-{reduction:.1f}%)")
            
            if processed % 50 == 0:
                print(f"  Progresso: {processed}/{total} ({(processed/total)*100:.1f}%)")
    
    print(f"  Concluído: {processed} arquivos, {reduced} reduzidos")
    
    return (processed, total_original, total_new)


def select_folders(base_images_dir: Path) -> Optional[List[Path]]:
    """List and select folders to process."""
    folders = sorted([
        folder for folder in base_images_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("processed")
    ])
    
    if not folders:
        print("\nNenhuma pasta 'processed*' encontrada em images/")
        return None
    
    print("\nPastas disponíveis para recompressão:")
    for idx, folder in enumerate(folders, 1):
        total_size = sum(
            f.stat().st_size for f in folder.iterdir() 
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ) / (1024 * 1024)
        file_count = len([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])
        print(f"  {idx}. {folder.name} ({file_count} arquivos, {total_size:.1f} MB)")
    
    print(f"  0. Todas as pastas")
    
    while True:
        try:
            choice = input("\nEscolha o número da pasta (ou 'q' para sair): ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice)
            if choice_idx == 0:
                return folders
            elif 1 <= choice_idx <= len(folders):
                return [folders[choice_idx - 1]]
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número.")


def main():
    """Main entry point."""
    base_dir = Path(r"D:\Development\forense")
    base_images_dir = base_dir / "images"
    
    print("\n=== Image Recompressor ===")
    print(f"Qualidade JPEG: {JPEG_QUALITY}")
    print(f"Workers: {MAX_WORKERS}")
    
    folders = select_folders(base_images_dir)
    if folders is None:
        print("Cancelado.")
        return
    
    print(f"\n--- Pastas selecionadas ---")
    for folder in folders:
        print(f"  - {folder.name}")
    
    confirm = input("\nIniciar recompressão? (s/n): ").strip().lower()
    if confirm != 's':
        print("Cancelado.")
        return
    
    grand_total_original = 0.0
    grand_total_new = 0.0
    grand_total_files = 0
    
    for folder in folders:
        files, original, new = process_folder(folder, MAX_WORKERS)
        grand_total_files += files
        grand_total_original += original
        grand_total_new += new
    
    saved = grand_total_original - grand_total_new
    if grand_total_original > 0:
        reduction_pct = (saved / grand_total_original) * 100
    else:
        reduction_pct = 0
    
    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50)
    print(f"  Arquivos processados: {grand_total_files}")
    print(f"  Tamanho original: {grand_total_original:.2f} MB")
    print(f"  Tamanho novo: {grand_total_new:.2f} MB")
    print(f"  Espaço economizado: {saved:.2f} MB ({reduction_pct:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
