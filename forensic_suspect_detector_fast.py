"""
Forensic Suspect Detector - Fast Parallel Version

Uses ThreadPoolExecutor to process multiple images concurrently.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
from PIL import Image


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_RAW_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heic'}
SUPPORTED_PROCESSED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

TECHNIQUES = {
    "clahe", "clahe_1.0", "clahe_4.0", "clahe_8.0", "clahe_16x16",
    "histogram_eq",
    "gamma_0.2", "gamma_0.3", "gamma_0.4", "gamma_0.5", "gamma_0.7",
    "gamma_1.5", "gamma_2.0", "gamma_2.5", "gamma_3.0", "gamma_4.0",
    "canny", "canny_sensitive", "canny_balanced", "canny_strict",
    "sobel_x", "sobel_y", "sobel_combined", "sobel_5x5",
    "laplacian", "laplacian_5", "gradient_magnitude",
    "dog", "morph_gradient",
    "red_channel", "green_channel", "blue_channel",
    "ir_simulation",
    "ms_480nm", "ms_620nm", "ms_850nm",
    "hemoglobin", "als_365nm", "als_415nm", "als_450nm",
    "hsv_h", "hsv_s", "hsv_v",
    "lab_l", "lab_a", "lab_b",
    "ycrcb_y", "ycrcb_cr", "ycrcb_cb",
    "unsharp", "highpass", "emboss", "median",
    "negative", "false_color",
    "skin_enhanced", "bilateral", "lbp",
    "dct_blocks", "wavelet",
    "retinex_ssr", "retinex_msr",
    "freq_filter", "freq_highpass",
    "cross_polarized",
    "ela", "noise_analysis", "jpeg_ghost",
}

TECHNIQUE_CATEGORIES = {
    "edge": {
        "canny", "canny_sensitive", "canny_balanced", "canny_strict",
        "sobel_x", "sobel_y", "sobel_combined", "sobel_5x5",
        "laplacian", "laplacian_5", "gradient_magnitude", "highpass", "emboss",
        "dog", "morph_gradient",
        "dct_blocks", "wavelet",
    },
    "enhancement": {
        "clahe", "clahe_1.0", "clahe_4.0", "clahe_8.0", "clahe_16x16",
        "histogram_eq",
        "gamma_0.2", "gamma_0.3", "gamma_0.4", "gamma_0.5", "gamma_0.7",
        "gamma_1.5", "gamma_2.0", "gamma_2.5", "gamma_3.0", "gamma_4.0",
        "unsharp", "negative", "false_color", "median",
        "retinex_ssr", "retinex_msr",
    },
    "channel": {
        "red_channel", "green_channel", "blue_channel",
        "hsv_h", "hsv_s", "hsv_v",
        "lab_l", "lab_a", "lab_b",
        "ycrcb_y", "ycrcb_cr", "ycrcb_cb",
    },
    "specialized": {
        "ir_simulation", "skin_enhanced", "bilateral", "lbp",
        "ms_480nm", "ms_620nm", "ms_850nm",
        "hemoglobin", "als_365nm", "als_415nm", "als_450nm",
        "freq_filter", "freq_highpass", "cross_polarized",
    },
    "forensic": {
        "ela", "noise_analysis", "jpeg_ghost",
    },
}

CATEGORY_WEIGHTS = {
    "edge": {"diff_ratio": 0.3, "diff_mean": 0.1, "ssim": 0.2, "edge_density": 0.4},
    "enhancement": {"diff_ratio": 0.4, "diff_mean": 0.3, "ssim": 0.3, "edge_density": 0.0},
    "channel": {"diff_ratio": 0.3, "diff_mean": 0.4, "ssim": 0.3, "edge_density": 0.0},
    "specialized": {"diff_ratio": 0.4, "diff_mean": 0.3, "ssim": 0.3, "edge_density": 0.0},
    "forensic": {"diff_ratio": 0.5, "diff_mean": 0.2, "ssim": 0.2, "edge_density": 0.1},
}

CATEGORY_THRESHOLDS = {
    "edge": {"iqr_mult": 1.8, "min_threshold": 60.0},
    "enhancement": {"iqr_mult": 1.3, "min_threshold": 35.0},
    "channel": {"iqr_mult": 1.3, "min_threshold": 30.0},
    "specialized": {"iqr_mult": 1.3, "min_threshold": 25.0},
    "forensic": {"iqr_mult": 1.0, "min_threshold": 20.0},
}

MAX_WORKERS = 8

_heif_registered = False
_heif_lock = threading.Lock()
_ssim_checked = False
_ssim_func = None
_ssim_lock = threading.Lock()
_cache_lock = threading.Lock()
_scores_lock = threading.Lock()


def _get_technique_category(technique: str) -> str:
    for category, techniques in TECHNIQUE_CATEGORIES.items():
        if technique in techniques:
            return category
    return "enhancement"


def _compute_ela(image: np.ndarray, quality: int = 90) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image, decoded)
    ela = cv2.multiply(ela, (255 // (100 - quality + 1),))
    return ela


def _compute_noise_analysis(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, denoised)
    noise_enhanced = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    return noise_enhanced


def _analyze_noise_consistency(gray: np.ndarray, block_size: int = 64) -> Tuple[float, float]:
    h, w = gray.shape
    block_stds = []
    
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            laplacian = cv2.Laplacian(block, cv2.CV_64F)
            block_stds.append(np.std(laplacian))
    
    if not block_stds:
        return 0.0, 0.0
    
    noise_mean = float(np.mean(block_stds))
    noise_std = float(np.std(block_stds))
    noise_inconsistency = noise_std / (noise_mean + 1e-6)
    
    return noise_mean, noise_inconsistency


def _compute_jpeg_ghost(image: np.ndarray, qualities: List[int] = None) -> Tuple[float, float, float]:
    """
    JPEG Ghost Analysis - detects regions with different compression history.
    Recompresses at multiple quality levels and analyzes variance in differences.
    Manipulated regions show different compression characteristics.
    """
    if qualities is None:
        qualities = [60, 70, 80, 90, 95]
    
    ghost_maps = []
    for q in qualities:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(image, decoded).astype(np.float32)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        ghost_maps.append(diff_gray)
    
    ghost_stack = np.stack(ghost_maps, axis=0)
    variance_map = np.var(ghost_stack, axis=0)
    
    ghost_mean = float(np.mean(variance_map))
    ghost_std = float(np.std(variance_map))
    ghost_max = float(np.max(variance_map))
    
    h, w = variance_map.shape
    block_size = 64
    block_vars = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = variance_map[y:y+block_size, x:x+block_size]
            block_vars.append(np.mean(block))
    
    if block_vars:
        block_inconsistency = float(np.std(block_vars) / (np.mean(block_vars) + 1e-6))
    else:
        block_inconsistency = 0.0
    
    return ghost_mean, ghost_std, block_inconsistency


def _register_heif() -> bool:
    global _heif_registered
    with _heif_lock:
        if _heif_registered:
            return True
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            _heif_registered = True
            return True
        except Exception as e:
            logger.warning("HEIC support is not available")
            return False


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _load_image(path: Path) -> Optional[np.ndarray]:
    if path.suffix.lower() == ".heic":
        if not _register_heif():
            return None
        try:
            with Image.open(path) as image:
                return _pil_to_bgr(image)
        except Exception as e:
            logger.error(f"Failed to read HEIC image: {path.name} - {e}")
            return None

    image = cv2.imread(str(path))
    if image is not None:
        return image

    try:
        with Image.open(path) as pil_image:
            return _pil_to_bgr(pil_image)
    except Exception as e:
        logger.error(f"Failed to read image: {path.name} - {e}")
        return None


def _extract_base_name(filename: str) -> str:
    stem = Path(filename).stem
    for technique in TECHNIQUES:
        suffix = f"_{technique}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _extract_technique(filename: str) -> str:
    stem = Path(filename).stem
    for technique in TECHNIQUES:
        suffix = f"_{technique}"
        if stem.endswith(suffix):
            return technique
    return ""


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _compute_ssim(raw_gray: np.ndarray, processed_gray: np.ndarray) -> Optional[float]:
    global _ssim_checked, _ssim_func
    with _ssim_lock:
        if not _ssim_checked:
            try:
                from skimage.metrics import structural_similarity as ssim
                _ssim_func = ssim
            except Exception:
                _ssim_func = None
            _ssim_checked = True

    if _ssim_func is None:
        return None

    try:
        return float(_ssim_func(raw_gray, processed_gray))
    except Exception:
        return None


def _score_processed(
    raw_gray: np.ndarray,
    processed_gray: np.ndarray,
    technique: str
) -> Tuple[float, float, float, float, float, Optional[float], float, str]:
    category = _get_technique_category(technique)
    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["enhancement"])
    
    diff = cv2.absdiff(raw_gray, processed_gray)
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))

    otsu_threshold, diff_mask = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    diff_ratio = float(np.mean(diff_mask > 0))

    ssim_value = _compute_ssim(raw_gray, processed_gray)
    ssim_score = 0.0
    if ssim_value is not None:
        ssim_score = max(0.0, (1.0 - ssim_value))

    edge_density = 0.0
    if category == "edge" or category == "forensic":
        edge_density = float(np.mean(processed_gray > 30))

    diff_ratio_norm = min(diff_ratio * 100.0, 100.0)
    diff_mean_norm = min(diff_mean / 2.0, 100.0)
    ssim_norm = ssim_score * 100.0
    edge_norm = edge_density * 100.0

    score = (
        weights["diff_ratio"] * diff_ratio_norm +
        weights["diff_mean"] * diff_mean_norm +
        weights["ssim"] * ssim_norm +
        weights["edge_density"] * edge_norm
    )

    return score, diff_ratio, diff_mean, diff_std, float(otsu_threshold), ssim_value, edge_density, category


def _process_raw_image(raw_name: str, raw_path: Path) -> Optional[Tuple[str, np.ndarray, List[Tuple[Path, str, float, str]]]]:
    """Process a single raw image - returns (raw_name, raw_gray, forensic_results)."""
    raw_img = _load_image(raw_path)
    if raw_img is None:
        return None
    
    logger.debug(f"[LOAD] Raw: {raw_path.name} ({raw_img.shape[1]}x{raw_img.shape[0]})")
    raw_gray = _to_gray(raw_img)
    
    results: List[Tuple[Path, str, float, str]] = []
    
    ela_img = _compute_ela(raw_img)
    ela_gray = _to_gray(ela_img)
    ela_mean = float(np.mean(ela_gray))
    ela_std = float(np.std(ela_gray))
    ela_max_region = float(np.percentile(ela_gray, 99))
    ela_score = (ela_mean / 2.0) + (ela_std / 1.5) + (ela_max_region / 3.0)
    
    logger.debug(
        f"[forensic/ela] {raw_name} | score={ela_score:.2f} | ela_mean={ela_mean:.2f} | ela_std={ela_std:.2f} | ela_max={ela_max_region:.2f}"
    )
    results.append((raw_path, "ela", ela_score, "forensic"))
    
    noise_mean, noise_inconsistency = _analyze_noise_consistency(raw_gray)
    noise_score = noise_inconsistency * 50.0 + (noise_mean / 2.0)
    
    logger.debug(
        f"[forensic/noise_analysis] {raw_name} | score={noise_score:.2f} | noise_mean={noise_mean:.2f} | noise_inconsistency={noise_inconsistency:.4f}"
    )
    results.append((raw_path, "noise_analysis", noise_score, "forensic"))
    
    ghost_mean, ghost_std, ghost_inconsistency = _compute_jpeg_ghost(raw_img)
    ghost_score = ghost_inconsistency * 40.0 + (ghost_mean * 2.0) + (ghost_std * 0.5)
    
    logger.debug(
        f"[forensic/jpeg_ghost] {raw_name} | score={ghost_score:.2f} | ghost_mean={ghost_mean:.2f} | ghost_std={ghost_std:.2f} | inconsistency={ghost_inconsistency:.4f}"
    )
    results.append((raw_path, "jpeg_ghost", ghost_score, "forensic"))
    
    return (raw_name, raw_gray, results)


def _process_single_processed(
    processed_path: Path,
    raw_gray: np.ndarray,
    base_name: str
) -> Optional[Tuple[str, Path, str, float, str]]:
    """Process a single processed image against its raw."""
    technique = _extract_technique(processed_path.name)
    
    processed_img = cv2.imread(str(processed_path))
    if processed_img is None:
        return None

    if processed_img.shape[:2] != raw_gray.shape[:2]:
        processed_img = cv2.resize(processed_img, (raw_gray.shape[1], raw_gray.shape[0]))
        logger.debug(f"[RESIZE] {processed_path.name} to {raw_gray.shape[1]}x{raw_gray.shape[0]}")

    processed_gray = _to_gray(processed_img)
    score, diff_ratio, diff_mean, diff_std, otsu_threshold, ssim_value, edge_density, category = _score_processed(
        raw_gray, processed_gray, technique
    )
    
    log_parts = [
        f"[{category}/{technique}] {processed_path.name}",
        f"score={score:.2f}",
        f"diff_ratio={diff_ratio:.4f}",
        f"diff_mean={diff_mean:.2f}",
        f"diff_std={diff_std:.2f}",
    ]
    if ssim_value is not None:
        log_parts.append(f"ssim={ssim_value:.4f}")
    if edge_density > 0:
        log_parts.append(f"edge_density={edge_density:.4f}")
    logger.debug(" | ".join(log_parts))
    
    return (base_name, processed_path, technique, score, category)


def detect_suspects(
    base_dir: Path,
    raw_folder: str,
    processed_folder: str,
    suspect_folder: str,
    max_workers: int = MAX_WORKERS
) -> None:
    raw_dir = base_dir / "images" / raw_folder
    processed_dir = base_dir / "images" / processed_folder
    suspect_dir = base_dir / "images" / suspect_folder

    logger.debug(f"[CONFIG] raw_dir={raw_dir}")
    logger.debug(f"[CONFIG] processed_dir={processed_dir}")
    logger.debug(f"[CONFIG] suspect_dir={suspect_dir}")
    logger.debug(f"[CONFIG] max_workers={max_workers}")

    if not raw_dir.exists():
        logger.warning(f"[ERROR] Raw directory not found: {raw_dir}")
        return
    if not processed_dir.exists():
        logger.warning(f"[ERROR] Processed directory not found: {processed_dir}")
        return

    suspect_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: Dict[str, Path] = {}
    for path in raw_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_RAW_EXTENSIONS:
            raw_paths[path.stem] = path

    logger.debug(f"[INIT] Found {len(raw_paths)} raw images")

    processed_paths = [
        path for path in processed_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_PROCESSED_EXTENSIONS
    ]
    logger.debug(f"[INIT] Found {len(processed_paths)} processed images")

    raw_gray_cache: Dict[str, np.ndarray] = {}
    scored_by_raw: Dict[str, List[Tuple[Path, str, float, str]]] = {}

    logger.debug(f"[PHASE1] Processing {len(raw_paths)} raw images in parallel...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_raw_image, raw_name, raw_path): raw_name
            for raw_name, raw_path in raw_paths.items()
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                raw_name, raw_gray, forensic_results = result
                with _cache_lock:
                    raw_gray_cache[raw_name] = raw_gray
                with _scores_lock:
                    scored_by_raw.setdefault(raw_name, []).extend(forensic_results)
            completed += 1
            if completed % 20 == 0:
                logger.debug(f"[PHASE1] Progress: {completed}/{len(raw_paths)} raw images")

    logger.debug(f"[PHASE1] Completed processing {len(raw_gray_cache)} raw images")

    processed_by_raw: Dict[str, List[Path]] = {}
    missing_raw = 0
    for processed_path in processed_paths:
        base_name = _extract_base_name(processed_path.name)
        if base_name in raw_gray_cache:
            processed_by_raw.setdefault(base_name, []).append(processed_path)
        else:
            missing_raw += 1

    if missing_raw > 0:
        logger.debug(f"[WARN] Processed images without raw match: {missing_raw}")

    logger.debug(f"[PHASE2] Processing {len(processed_paths) - missing_raw} processed images in parallel...")
    
    read_errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for base_name, paths in processed_by_raw.items():
            raw_gray = raw_gray_cache[base_name]
            for processed_path in paths:
                futures.append(
                    executor.submit(_process_single_processed, processed_path, raw_gray, base_name)
                )
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                base_name, processed_path, technique, score, category = result
                with _scores_lock:
                    scored_by_raw.setdefault(base_name, []).append((processed_path, technique, score, category))
            else:
                read_errors += 1
            completed += 1
            if completed % 100 == 0:
                logger.debug(f"[PHASE2] Progress: {completed}/{len(futures)} processed images")

    logger.debug(f"[PHASE2] Completed processing {completed - read_errors} processed images")
    if read_errors > 0:
        logger.debug(f"[WARN] Image read errors: {read_errors}")

    total_suspects = 0
    total_processed = 0

    global_forensic_items: List[Tuple[Path, str, float, str, str]] = []
    for base_name, items in scored_by_raw.items():
        for item in items:
            if item[3] == "forensic":
                global_forensic_items.append((item[0], item[1], item[2], item[3], base_name))

    if len(global_forensic_items) >= 4:
        ela_scores = [item[2] for item in global_forensic_items if item[1] == "ela"]
        noise_scores = [item[2] for item in global_forensic_items if item[1] == "noise_analysis"]
        
        if len(ela_scores) >= 2:
            ela_arr = np.array(ela_scores, dtype=np.float32)
            ela_q1 = float(np.percentile(ela_arr, 25))
            ela_q3 = float(np.percentile(ela_arr, 75))
            ela_iqr = ela_q3 - ela_q1
            ela_median = float(np.median(ela_arr))
            ela_threshold = max(
                ela_q3 + (CATEGORY_THRESHOLDS["forensic"]["iqr_mult"] * ela_iqr),
                ela_median * 1.3,
                CATEGORY_THRESHOLDS["forensic"]["min_threshold"]
            )
            logger.debug(
                f"[STATS] GLOBAL/ela: q1={ela_q1:.2f} median={ela_median:.2f} q3={ela_q3:.2f} iqr={ela_iqr:.2f} threshold={ela_threshold:.2f} items={len(ela_scores)}"
            )
            for item in global_forensic_items:
                if item[1] == "ela" and item[2] >= ela_threshold:
                    raw_name = item[4]
                    logger.debug(f"[SUSPECT] {raw_name}/ela: score={item[2]:.2f} >= threshold={ela_threshold:.2f}")
                    target_path = suspect_dir / f"{raw_name}_ela_suspect.png"
                    if not target_path.exists():
                        try:
                            raw_path = raw_paths.get(raw_name)
                            if raw_path:
                                shutil.copy2(raw_path, target_path)
                                total_suspects += 1
                                logger.debug(f"[COPIED] Suspect: {target_path.name}")
                        except Exception as e:
                            logger.error(f"[ERROR] Failed to copy ela suspect {raw_name}: {e}")
        
        if len(noise_scores) >= 2:
            noise_arr = np.array(noise_scores, dtype=np.float32)
            noise_q1 = float(np.percentile(noise_arr, 25))
            noise_q3 = float(np.percentile(noise_arr, 75))
            noise_iqr = noise_q3 - noise_q1
            noise_median = float(np.median(noise_arr))
            noise_threshold = max(
                noise_q3 + (CATEGORY_THRESHOLDS["forensic"]["iqr_mult"] * noise_iqr),
                noise_median * 1.3,
                CATEGORY_THRESHOLDS["forensic"]["min_threshold"]
            )
            logger.debug(
                f"[STATS] GLOBAL/noise: q1={noise_q1:.2f} median={noise_median:.2f} q3={noise_q3:.2f} iqr={noise_iqr:.2f} threshold={noise_threshold:.2f} items={len(noise_scores)}"
            )
            for item in global_forensic_items:
                if item[1] == "noise_analysis" and item[2] >= noise_threshold:
                    raw_name = item[4]
                    logger.debug(f"[SUSPECT] {raw_name}/noise: score={item[2]:.2f} >= threshold={noise_threshold:.2f}")
                    target_path = suspect_dir / f"{raw_name}_noise_suspect.png"
                    if not target_path.exists():
                        try:
                            raw_path = raw_paths.get(raw_name)
                            if raw_path:
                                shutil.copy2(raw_path, target_path)
                                total_suspects += 1
                                logger.debug(f"[COPIED] Suspect: {target_path.name}")
                        except Exception as e:
                            logger.error(f"[ERROR] Failed to copy noise suspect {raw_name}: {e}")

        ghost_scores = [item[2] for item in global_forensic_items if item[1] == "jpeg_ghost"]
        if ghost_scores:
            ghost_arr = np.array(ghost_scores, dtype=np.float32)
            ghost_q1 = float(np.percentile(ghost_arr, 25))
            ghost_q3 = float(np.percentile(ghost_arr, 75))
            ghost_iqr = ghost_q3 - ghost_q1
            ghost_median = float(np.median(ghost_arr))
            ghost_threshold = max(
                ghost_q3 + (CATEGORY_THRESHOLDS["forensic"]["iqr_mult"] * ghost_iqr),
                ghost_median * 1.3,
                CATEGORY_THRESHOLDS["forensic"]["min_threshold"]
            )
            logger.debug(
                f"[STATS] GLOBAL/jpeg_ghost: q1={ghost_q1:.2f} median={ghost_median:.2f} q3={ghost_q3:.2f} iqr={ghost_iqr:.2f} threshold={ghost_threshold:.2f} items={len(ghost_scores)}"
            )
            for item in global_forensic_items:
                if item[1] == "jpeg_ghost" and item[2] >= ghost_threshold:
                    raw_name = item[4]
                    logger.debug(f"[SUSPECT] {raw_name}/jpeg_ghost: score={item[2]:.2f} >= threshold={ghost_threshold:.2f}")
                    target_path = suspect_dir / f"{raw_name}_jpeg_ghost_suspect.png"
                    if not target_path.exists():
                        try:
                            raw_path = raw_paths.get(raw_name)
                            if raw_path:
                                shutil.copy2(raw_path, target_path)
                                total_suspects += 1
                                logger.debug(f"[COPIED] Suspect: {target_path.name}")
                        except Exception as e:
                            logger.error(f"[ERROR] Failed to copy jpeg_ghost suspect {raw_name}: {e}")

    for base_name, items in scored_by_raw.items():
        total_processed += len(items)
        
        items_by_category: Dict[str, List[Tuple[Path, str, float, str]]] = {}
        for item in items:
            cat = item[3]
            if cat == "forensic":
                continue
            items_by_category.setdefault(cat, []).append(item)
        
        for category, cat_items in items_by_category.items():
            if len(cat_items) < 2:
                continue
                
            scores = np.array([item[2] for item in cat_items], dtype=np.float32)
            cat_config = CATEGORY_THRESHOLDS.get(category, CATEGORY_THRESHOLDS["enhancement"])
            
            q1 = float(np.percentile(scores, 25))
            q3 = float(np.percentile(scores, 75))
            iqr = q3 - q1
            median = float(np.median(scores))
            
            threshold = max(
                q3 + (cat_config["iqr_mult"] * iqr),
                median * 1.2,
                cat_config["min_threshold"]
            )
            
            logger.debug(
                f"[STATS] {base_name}/{category}: q1={q1:.2f} median={median:.2f} q3={q3:.2f} iqr={iqr:.2f} threshold={threshold:.2f} items={len(cat_items)}"
            )

            suspects = [item for item in cat_items if item[2] >= threshold]

            if suspects:
                logger.debug(f"[SUSPECT] {base_name}/{category}: {len(suspects)} suspects detected (threshold={threshold:.2f})")

            for processed_path, technique, score, _ in suspects:
                logger.debug(
                    f"[SUSPECT] Marking: {processed_path.name} | category={category} | technique={technique} | score={score:.2f} | threshold={threshold:.2f}"
                )
                target_path = suspect_dir / processed_path.name
                if target_path.exists():
                    logger.debug(f"[SKIP] Suspect already exists: {target_path.name}")
                    continue
                try:
                    shutil.copy2(processed_path, target_path)
                    total_suspects += 1
                    logger.debug(f"[COPIED] Suspect: {target_path.name}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to copy suspect {processed_path.name}: {e}")

    logger.debug(f"[SUMMARY] Total processed: {total_processed} | Total suspects: {total_suspects}")


def _select_folder(base_images_dir: Path, prefix: str, folder_type: str) -> Optional[str]:
    """List and select a folder by prefix."""
    folders = sorted([
        folder for folder in base_images_dir.iterdir()
        if folder.is_dir() and folder.name.startswith(prefix)
    ])
    
    if not folders:
        print(f"\nNenhuma pasta '{prefix}*' encontrada em images/")
        return None
    
    print(f"\nPastas {folder_type} disponíveis:")
    for idx, folder in enumerate(folders, 1):
        print(f"  {idx}. {folder.name}")
    
    while True:
        try:
            choice = input(f"\nEscolha o número da pasta {folder_type} (ou 'q' para sair): ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(folders):
                return folders[choice_idx].name
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número.")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fast Forensic Suspect Detector")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help=f"Number of worker threads (default: {MAX_WORKERS})")
    args = parser.parse_args()
    
    base_dir = Path(r"D:\Development\forense")
    base_images_dir = base_dir / "images"
    
    raw_folder = _select_folder(base_images_dir, "raw", "raw")
    if raw_folder is None:
        print("Cancelado.")
        return
    
    processed_folder = _select_folder(base_images_dir, "processed", "processed")
    if processed_folder is None:
        print("Cancelado.")
        return
    
    default_suspect = f"suspect_{raw_folder}"
    suspect_input = input(f"\nNome da pasta suspect (Enter para '{default_suspect}'): ").strip()
    suspect_folder = suspect_input if suspect_input else default_suspect
    
    print(f"\n--- Configuração ---")
    print(f"  Raw: {raw_folder}")
    print(f"  Processed: {processed_folder}")
    print(f"  Suspect: {suspect_folder}")
    print(f"  Workers: {args.workers}")
    
    confirm = input("\nIniciar processamento? (s/n): ").strip().lower()
    if confirm != 's':
        print("Cancelado.")
        return
    
    print()
    logger.debug("[START] Fast suspect detection from processed images")
    logger.debug(f"[CONFIG] Using {args.workers} worker threads")
    detect_suspects(base_dir, raw_folder, processed_folder, suspect_folder, max_workers=args.workers)
    logger.debug("[END] Suspect detection completed")
    print("\nProcessamento concluído!")


if __name__ == "__main__":
    main()
