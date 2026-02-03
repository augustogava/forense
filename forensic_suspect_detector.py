import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

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
    "clahe",
    "histogram_eq",
    "gamma_0.5",
    "gamma_1.5",
    "gamma_2.0",
    "canny",
    "sobel_x",
    "sobel_y",
    "sobel_combined",
    "laplacian",
    "gradient_magnitude",
    "red_channel",
    "green_channel",
    "blue_channel",
    "ir_simulation",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "lab_l",
    "lab_a",
    "lab_b",
    "ycrcb_y",
    "ycrcb_cr",
    "ycrcb_cb",
    "unsharp",
    "highpass",
    "emboss",
    "negative",
    "false_color",
    "skin_enhanced",
    "bilateral",
    "lbp",
    "dct_blocks",
    "wavelet",
    "ela",
    "noise_analysis",
}

TECHNIQUE_CATEGORIES = {
    "edge": {
        "canny", "sobel_x", "sobel_y", "sobel_combined",
        "laplacian", "gradient_magnitude", "highpass", "emboss",
        "dct_blocks", "wavelet",
    },
    "enhancement": {
        "clahe", "histogram_eq", "gamma_0.5", "gamma_1.5", "gamma_2.0",
        "unsharp", "negative", "false_color",
    },
    "channel": {
        "red_channel", "green_channel", "blue_channel",
        "hsv_h", "hsv_s", "hsv_v",
        "lab_l", "lab_a", "lab_b",
        "ycrcb_y", "ycrcb_cr", "ycrcb_cb",
    },
    "specialized": {
        "ir_simulation", "skin_enhanced", "bilateral", "lbp",
    },
    "forensic": {
        "ela", "noise_analysis",
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

TECHNIQUE_TO_CATEGORY = {
    tech: cat for cat, techs in TECHNIQUE_CATEGORIES.items() for tech in techs
}

TECHNIQUES_BY_LENGTH = sorted(TECHNIQUES, key=len, reverse=True)

_heif_registered = False
_ssim_checked = False
_ssim_func = None


def _get_technique_category(technique: str) -> str:
    return TECHNIQUE_TO_CATEGORY.get(technique, "enhancement")


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
    
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    if h_blocks == 0 or w_blocks == 0:
        return 0.0, 0.0
    
    block_stds = np.empty(h_blocks * w_blocks, dtype=np.float64)
    idx = 0
    for y in range(h_blocks):
        for x in range(w_blocks):
            block = gray[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size]
            laplacian = cv2.Laplacian(block, cv2.CV_64F)
            block_stds[idx] = np.std(laplacian)
            idx += 1
    
    noise_mean = float(np.mean(block_stds))
    noise_std = float(np.std(block_stds))
    noise_inconsistency = noise_std / (noise_mean + 1e-6)
    
    return noise_mean, noise_inconsistency


def _register_heif() -> bool:
    global _heif_registered
    if _heif_registered:
        return True
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        _heif_registered = True
        return True
    except Exception as e:
        logger.warning("HEIC support is not available, skipping HEIC files")
        logger.debug(f"HEIC support error: {e}")
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
    for technique in TECHNIQUES_BY_LENGTH:
        suffix = f"_{technique}"
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _extract_technique(filename: str) -> str:
    stem = Path(filename).stem
    for technique in TECHNIQUES_BY_LENGTH:
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
    if not _ssim_checked:
        try:
            from skimage.metrics import structural_similarity as ssim
            _ssim_func = ssim
        except Exception as e:
            _ssim_func = None
            logger.warning("SSIM is not available, skipping SSIM scoring")
            logger.debug(f"SSIM import error: {e}")
        _ssim_checked = True

    if _ssim_func is None:
        return None

    try:
        return float(_ssim_func(raw_gray, processed_gray))
    except Exception as e:
        logger.debug(f"SSIM failed: {e}")
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


def detect_suspects(base_dir: Path) -> None:
    raw_dir = base_dir / "images" / "raw"
    processed_dir = base_dir / "images" / "processed"
    suspect_dir = base_dir / "images" / "suspect"

    logger.debug(f"[CONFIG] raw_dir={raw_dir}")
    logger.debug(f"[CONFIG] processed_dir={processed_dir}")
    logger.debug(f"[CONFIG] suspect_dir={suspect_dir}")

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
    missing_raw = 0
    read_errors = 0

    for raw_name, raw_path in raw_paths.items():
        raw_img = _load_image(raw_path)
        if raw_img is None:
            continue
        
        logger.debug(f"[LOAD] Raw: {raw_path.name} ({raw_img.shape[1]}x{raw_img.shape[0]})")
        raw_gray = _to_gray(raw_img)
        raw_gray_cache[raw_name] = raw_gray
        
        ela_img = _compute_ela(raw_img)
        ela_gray = _to_gray(ela_img)
        ela_mean = float(np.mean(ela_gray))
        ela_std = float(np.std(ela_gray))
        ela_max_region = float(np.percentile(ela_gray, 99))
        
        ela_score = (ela_mean / 2.0) + (ela_std / 1.5) + (ela_max_region / 3.0)
        
        logger.debug(
            f"[forensic/ela] {raw_name} | score={ela_score:.2f} | ela_mean={ela_mean:.2f} | ela_std={ela_std:.2f} | ela_max={ela_max_region:.2f}"
        )
        scored_by_raw.setdefault(raw_name, []).append((raw_path, "ela", ela_score, "forensic"))
        
        noise_mean, noise_inconsistency = _analyze_noise_consistency(raw_gray)
        noise_score = noise_inconsistency * 50.0 + (noise_mean / 2.0)
        
        logger.debug(
            f"[forensic/noise_analysis] {raw_name} | score={noise_score:.2f} | noise_mean={noise_mean:.2f} | noise_inconsistency={noise_inconsistency:.4f}"
        )
        scored_by_raw.setdefault(raw_name, []).append((raw_path, "noise_analysis", noise_score, "forensic"))

    for processed_path in processed_paths:
        base_name = _extract_base_name(processed_path.name)
        technique = _extract_technique(processed_path.name)

        raw_path = raw_paths.get(base_name)
        if raw_path is None:
            missing_raw += 1
            logger.debug(f"[SKIP] No raw match: {processed_path.name}")
            continue

        raw_gray = raw_gray_cache.get(base_name)
        if raw_gray is None:
            missing_raw += 1
            logger.debug(f"[SKIP] Raw not in cache: {processed_path.name}")
            continue

        processed_img = cv2.imread(str(processed_path))
        if processed_img is None:
            read_errors += 1
            logger.warning(f"[ERROR] Could not read: {processed_path.name}")
            continue

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

        scored_by_raw.setdefault(base_name, []).append((processed_path, technique, score, category))

    if missing_raw > 0:
        logger.debug(f"[WARN] Processed images without raw match: {missing_raw}")
    if read_errors > 0:
        logger.debug(f"[WARN] Image read errors: {read_errors}")

    total_suspects = 0
    total_processed = 0

    global_forensic_items: List[Tuple[Path, str, float, str, str]] = []
    for base_name, items in scored_by_raw.items():
        for item in items:
            if item[3] == "forensic":
                global_forensic_items.append((item[0], item[1], item[2], item[3], base_name))

    forensic_threshold = CATEGORY_THRESHOLDS["forensic"]["min_threshold"]
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


def main() -> None:
    base_dir = Path(r"D:\Development\forense")
    logger.debug("[START] Suspect detection from processed images")
    detect_suspects(base_dir)
    logger.debug("[END] Suspect detection completed")


if __name__ == "__main__":
    main()
