#!/usr/bin/env python3
"""
Forensic Image Processor

Script Python para processar imagens forenses aplicando múltiplas técnicas 
de análise para revelar marcas, padrões e detalhes ocultos em pele/corpo.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import cv2
import numpy as np
from skimage import exposure
from PIL import Image

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_WORKERS = 8


class ForensicImageProcessor:
    """Main processor class for forensic image analysis."""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the processor.

        Args:
            input_dir: Path to directory with raw images
            output_dir: Path to directory for processed images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized processor - Input: {input_dir}, Output: {output_dir}")

    def process_all_images(self, max_workers: int = MAX_WORKERS) -> None:
        """Process all images in the input directory using parallel threads."""
        converted = self._convert_heic_images()
        if converted > 0:
            logger.debug(f"Converted {converted} HEIC images to PNG")

        image_files = self._get_image_files()

        if not image_files:
            logger.warning("No image files found in input directory")
            return

        total = len(image_files)
        logger.debug(f"Found {total} images to process with {max_workers} workers")
        print(f"Processando {total} imagens com {max_workers} threads...")

        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_image_safe, img_path): img_path
                for img_path in image_files
            }

            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
                    failed += 1

                print(f"\r  Progresso: {completed + failed}/{total} ({completed} ok, {failed} erro)", end="", flush=True)

        print()
        logger.debug(f"Batch processing completed: {completed} success, {failed} failed")

    def _process_image_safe(self, image_path: Path) -> bool:
        """Process a single image with error handling for threading."""
        try:
            self.process_single_image(image_path)
            return True
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return False

    def _convert_heic_images(self) -> int:
        heic_files = [
            path for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".heic"
        ]

        if not heic_files:
            return 0

        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception as e:
            logger.warning("HEIC support is not available, skipping conversion")
            logger.debug(f"HEIC support error: {e}")
            return 0

        converted = 0

        for heic_path in heic_files:
            target_path = self.input_dir / f"{heic_path.stem}.png"
            if target_path.exists():
                logger.debug(f"PNG already exists for {heic_path.name}")
                continue

            try:
                with Image.open(heic_path) as image:
                    image.save(target_path, format="PNG")
                converted += 1
                logger.debug(f"Converted HEIC to PNG: {target_path.name}")
            except Exception as e:
                logger.error(f"Failed to convert {heic_path.name}: {e}")

        return converted

    def _get_image_files(self) -> List[Path]:
        """Get list of supported image files in input directory."""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_dir.glob(f"*{ext}"))
            files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        return sorted(set(files))

    def process_single_image(self, image_path: Path) -> None:
        """
        Process a single image with all forensic techniques.

        Args:
            image_path: Path to the input image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        base_name = image_path.stem
        logger.debug(f"Processing: {base_name}")

        self._apply_contrast_enhancements(image, base_name)
        self._apply_edge_detection(image, base_name)
        self._apply_color_analysis(image, base_name)
        self._apply_detail_enhancement(image, base_name)
        self._apply_specialized_filters(image, base_name)

        logger.debug(f"Completed processing: {base_name}")

    def _save_image(self, image: np.ndarray, base_name: str, suffix: str) -> None:
        """Save processed image with appropriate naming."""
        output_path = self.output_dir / f"{base_name}_{suffix}.jpg"
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # ========== CONTRAST ENHANCEMENT METHODS ==========

    def _apply_contrast_enhancements(self, image: np.ndarray, base_name: str) -> None:
        """Apply all contrast enhancement techniques."""
        logger.debug(f"Applying contrast enhancements to {base_name}")

        clahe_result = self.apply_clahe(image)
        self._save_image(clahe_result, base_name, "clahe")

        for clip_limit in [1.0, 4.0, 8.0]:
            clahe_var = self.apply_clahe(image, clip_limit=clip_limit)
            self._save_image(clahe_var, base_name, f"clahe_{clip_limit}")

        clahe_16 = self.apply_clahe(image, clip_limit=2.0, tile_size=16)
        self._save_image(clahe_16, base_name, "clahe_16x16")

        hist_eq_result = self.apply_histogram_eq(image)
        self._save_image(hist_eq_result, base_name, "histogram_eq")

        for gamma in [0.2, 0.3, 0.4, 0.5, 0.7, 1.5, 2.0, 2.5, 3.0, 4.0]:
            gamma_result = self.apply_gamma(image, gamma)
            self._save_image(gamma_result, base_name, f"gamma_{gamma}")

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Enhances local contrast to reveal subtle bruises and marks.
        Critical for detecting sub-dermal patterns not visible to naked eye.
        
        Args:
            clip_limit: Threshold for contrast limiting (1.0-8.0 recommended)
            tile_size: Size of grid for histogram equalization
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            l_enhanced = clahe.apply(l_channel)

            enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            result = clahe.apply(image)

        return result

    def apply_histogram_eq(self, image: np.ndarray) -> np.ndarray:
        """
        Apply global histogram equalization.
        
        Global contrast enhancement for overall visibility improvement.
        """
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

            y_eq = cv2.equalizeHist(y_channel)

            enhanced_ycrcb = cv2.merge([y_eq, cr_channel, cb_channel])
            result = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            result = cv2.equalizeHist(image)

        return result

    def apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction.
        
        Adjusts brightness curves to reveal shadows/highlights.
        
        Args:
            gamma: Gamma value (< 1 brightens, > 1 darkens)
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")

        return cv2.LUT(image, table)

    # ========== EDGE DETECTION METHODS ==========

    def _apply_edge_detection(self, image: np.ndarray, base_name: str) -> None:
        """Apply all edge detection techniques."""
        logger.debug(f"Applying edge detection to {base_name}")

        canny_result = self.apply_canny(image)
        self._save_image(canny_result, base_name, "canny")

        canny_configs = [
            ("canny_sensitive", 20, 80),
            ("canny_balanced", 50, 150),
            ("canny_strict", 100, 200),
        ]
        for suffix, low, high in canny_configs:
            canny_var = self.apply_canny(image, low_threshold=low, high_threshold=high)
            self._save_image(canny_var, base_name, suffix)

        sobel_x, sobel_y, sobel_combined = self.apply_sobel(image)
        self._save_image(sobel_x, base_name, "sobel_x")
        self._save_image(sobel_y, base_name, "sobel_y")
        self._save_image(sobel_combined, base_name, "sobel_combined")

        sobel_x5, sobel_y5, sobel_combined5 = self.apply_sobel(image, ksize=5)
        self._save_image(sobel_combined5, base_name, "sobel_5x5")

        laplacian_result = self.apply_laplacian(image)
        self._save_image(laplacian_result, base_name, "laplacian")

        laplacian_5 = self.apply_laplacian(image, ksize=5)
        self._save_image(laplacian_5, base_name, "laplacian_5")

        gradient_result = self.apply_gradient_magnitude(image)
        self._save_image(gradient_result, base_name, "gradient_magnitude")

        dog_result = self.apply_difference_of_gaussians(image)
        self._save_image(dog_result, base_name, "dog")

        morph_grad = self.apply_morphological_gradient(image)
        self._save_image(morph_grad, base_name, "morph_gradient")

    def apply_canny(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """
        Apply Canny edge detection.
        
        Detects boundaries of marks, lines, and patterns.
        
        Args:
            low_threshold: Lower threshold for hysteresis
            high_threshold: Upper threshold for hysteresis
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_difference_of_gaussians(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Difference of Gaussians (DoG) filter.
        
        Multi-scale edge detection that reveals marks at different sizes.
        Effective for detecting bruise boundaries of varying sizes.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blur1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        blur2 = cv2.GaussianBlur(gray, (9, 9), 2.0)

        dog = cv2.subtract(blur1, blur2)
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.cvtColor(dog.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def apply_morphological_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological gradient.
        
        Difference between dilation and erosion, highlights object boundaries.
        Effective for detecting edges of lesions and marks.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    def apply_sobel(self, image: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Sobel filter for X and Y gradients.
        
        Highlights directional edges (useful for ligature/pressure marks).
        
        Args:
            ksize: Kernel size (3 or 5). Larger kernels detect broader edges.
        
        Returns:
            Tuple of (sobel_x, sobel_y, sobel_combined)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        sobel_x_abs = cv2.convertScaleAbs(sobel_x)
        sobel_y_abs = cv2.convertScaleAbs(sobel_y)

        sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

        return (
            cv2.cvtColor(sobel_x_abs, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sobel_y_abs, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
        )

    def apply_laplacian(self, image: np.ndarray, ksize: int = 1) -> np.ndarray:
        """
        Apply Laplacian filter.
        
        Second derivative for fine detail enhancement.
        
        Args:
            ksize: Kernel size (1, 3, or 5). Larger kernels detect broader features.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        laplacian_abs = cv2.convertScaleAbs(laplacian)

        return cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR)

    def apply_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate gradient magnitude.
        
        Shows intensity changes indicating marks or patterns.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = magnitude.astype(np.uint8)

        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    # ========== COLOR CHANNEL ANALYSIS METHODS ==========

    def _apply_color_analysis(self, image: np.ndarray, base_name: str) -> None:
        """Apply all color channel analysis techniques."""
        logger.debug(f"Applying color analysis to {base_name}")

        red, green, blue = self.extract_rgb_channels(image)
        self._save_image(red, base_name, "red_channel")
        self._save_image(green, base_name, "green_channel")
        self._save_image(blue, base_name, "blue_channel")

        ir_result = self.simulate_infrared(image)
        self._save_image(ir_result, base_name, "ir_simulation")

        ms_480 = self.simulate_multispectral(image, wavelength=480)
        self._save_image(ms_480, base_name, "ms_480nm")

        ms_620 = self.simulate_multispectral(image, wavelength=620)
        self._save_image(ms_620, base_name, "ms_620nm")

        ms_850 = self.simulate_multispectral(image, wavelength=850)
        self._save_image(ms_850, base_name, "ms_850nm")

        hemo = self.enhance_hemoglobin(image)
        self._save_image(hemo, base_name, "hemoglobin")

        als_415 = self.simulate_als(image, wavelength=415)
        self._save_image(als_415, base_name, "als_415nm")

        als_450 = self.simulate_als(image, wavelength=450)
        self._save_image(als_450, base_name, "als_450nm")

        als_365 = self.simulate_als(image, wavelength=365)
        self._save_image(als_365, base_name, "als_365nm")

        h, s, v = self.extract_hsv_channels(image)
        self._save_image(h, base_name, "hsv_h")
        self._save_image(s, base_name, "hsv_s")
        self._save_image(v, base_name, "hsv_v")

        l_chan, a_chan, b_chan = self.extract_lab_channels(image)
        self._save_image(l_chan, base_name, "lab_l")
        self._save_image(a_chan, base_name, "lab_a")
        self._save_image(b_chan, base_name, "lab_b")

        y_chan, cr_chan, cb_chan = self.extract_ycrcb_channels(image)
        self._save_image(y_chan, base_name, "ycrcb_y")
        self._save_image(cr_chan, base_name, "ycrcb_cr")
        self._save_image(cb_chan, base_name, "ycrcb_cb")

    def extract_rgb_channels(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract RGB channels separately.
        
        Skin marks appear differently per channel.
        
        Returns:
            Tuple of (red, green, blue) channel images
        """
        b, g, r = cv2.split(image)

        red_img = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        green_img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        blue_img = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

        return red_img, green_img, blue_img

    def simulate_infrared(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate near-infrared imaging.
        
        Simulates near-infrared by boosting red and reducing blue.
        Research shows bruises visible at specific wavelengths.
        Formula: 0.6*R + 0.3*G + 0.1*B
        """
        b, g, r = cv2.split(image)

        ir = (0.6 * r.astype(np.float32) +
              0.3 * g.astype(np.float32) +
              0.1 * b.astype(np.float32))

        ir = np.clip(ir, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ir_enhanced = clahe.apply(ir)

        return cv2.cvtColor(ir_enhanced, cv2.COLOR_GRAY2BGR)

    def simulate_multispectral(self, image: np.ndarray, wavelength: int) -> np.ndarray:
        """
        Simulate multispectral imaging at specific wavelengths.
        
        Based on forensic research:
        - 480nm: Best for fresh bruises (blue light)
        - 620nm: Best for healing bruises (red-orange light)
        - 850nm: Best for deep/old bruises (near-infrared)
        
        Args:
            wavelength: Target wavelength in nm (480, 620, or 850)
        """
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)

        if wavelength == 480:
            result = 0.7 * b + 0.2 * g + 0.1 * r
        elif wavelength == 620:
            result = 0.1 * b + 0.3 * g + 0.6 * r
        elif wavelength == 850:
            result = 0.05 * b + 0.15 * g + 0.8 * r
        else:
            result = 0.33 * b + 0.33 * g + 0.34 * r

        result = np.clip(result, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(result)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def enhance_hemoglobin(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance hemoglobin visibility for bruise detection.
        
        Hemoglobin absorbs strongly at ~415nm (Soret band) and ~540-580nm.
        This simulates detection of hemoglobin degradation products
        (biliverdin, bilirubin) which cause bruise color changes.
        """
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)

        hemo = r - 0.5 * g - 0.5 * b
        hemo = np.clip(hemo + 128, 0, 255).astype(np.uint8)

        bilirubin = g - 0.3 * r - 0.3 * b
        bilirubin = np.clip(bilirubin + 128, 0, 255).astype(np.uint8)

        combined = cv2.addWeighted(hemo, 0.6, bilirubin, 0.4, 0)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(combined)

        return cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)

    def simulate_als(self, image: np.ndarray, wavelength: int) -> np.ndarray:
        """
        Simulate Alternative Light Source (ALS) with orange/yellow filter.
        
        Based on forensic research:
        - 415nm + yellow/orange filter: Increases bruise detection across skin tones
        - 450nm + yellow/orange filter: Effective for subcutaneous injuries
        
        Args:
            wavelength: ALS wavelength (415 or 450 nm)
        """
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)

        if wavelength == 415:
            als = 0.8 * b + 0.15 * g + 0.05 * r
            filter_color = np.array([0, 100, 200], dtype=np.float32)
        else:
            als = 0.6 * b + 0.3 * g + 0.1 * r
            filter_color = np.array([0, 150, 255], dtype=np.float32)

        als = np.clip(als, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(als)

        colored = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        colored = colored.astype(np.float32)
        for i in range(3):
            colored[:, :, i] = colored[:, :, i] * (filter_color[i] / 255.0 + 0.3)
        colored = np.clip(colored, 0, 255).astype(np.uint8)

        return colored

    def extract_hsv_channels(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract HSV (Hue, Saturation, Value) channels.
        
        Returns:
            Tuple of (H, S, V) channel images
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h_img = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
        s_img = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        v_img = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

        return h_img, s_img, v_img

    def extract_lab_channels(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract LAB (Lightness, A, B) color opponent channels.
        
        Returns:
            Tuple of (L, A, B) channel images
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        l_img = cv2.cvtColor(l_chan, cv2.COLOR_GRAY2BGR)
        a_img = cv2.cvtColor(a_chan, cv2.COLOR_GRAY2BGR)
        b_img = cv2.cvtColor(b_chan, cv2.COLOR_GRAY2BGR)

        return l_img, a_img, b_img

    def extract_ycrcb_channels(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract YCrCb (Luminance, Chrominance) channels.
        
        Returns:
            Tuple of (Y, Cr, Cb) channel images
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_chan, cr_chan, cb_chan = cv2.split(ycrcb)

        y_img = cv2.cvtColor(y_chan, cv2.COLOR_GRAY2BGR)
        cr_img = cv2.cvtColor(cr_chan, cv2.COLOR_GRAY2BGR)
        cb_img = cv2.cvtColor(cb_chan, cv2.COLOR_GRAY2BGR)

        return y_img, cr_img, cb_img

    # ========== DETAIL ENHANCEMENT METHODS ==========

    def _apply_detail_enhancement(self, image: np.ndarray, base_name: str) -> None:
        """Apply all detail enhancement techniques."""
        logger.debug(f"Applying detail enhancement to {base_name}")

        unsharp_result = self.apply_unsharp_mask(image)
        self._save_image(unsharp_result, base_name, "unsharp")

        highpass_result = self.apply_highpass(image)
        self._save_image(highpass_result, base_name, "highpass")

        emboss_result = self.apply_emboss(image)
        self._save_image(emboss_result, base_name, "emboss")

        median_result = self.apply_median_filter(image)
        self._save_image(median_result, base_name, "median")

    def apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking.
        
        Sharpens fine details like pattern marks.
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        return unsharp

    def apply_highpass(self, image: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter.
        
        Removes low-frequency data, enhances texture/marks.
        """
        lowpass = cv2.GaussianBlur(image, (21, 21), 0)

        highpass = cv2.subtract(image, lowpass)
        highpass = cv2.add(highpass, 128)

        return highpass

    def apply_emboss(self, image: np.ndarray) -> np.ndarray:
        """
        Apply emboss filter.
        
        Relief effect to reveal surface texture marks.
        """
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = cv2.add(embossed, 128)

        return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

    def apply_median_filter(self, image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """
        Apply median filter.
        
        Removes noise while preserving edges, revealing subtle patterns.
        Useful for cleaning up images before further analysis.
        """
        return cv2.medianBlur(image, ksize)

    # ========== SPECIALIZED FORENSIC FILTERS ==========

    def _apply_specialized_filters(self, image: np.ndarray, base_name: str) -> None:
        """Apply all specialized forensic filters."""
        logger.debug(f"Applying specialized filters to {base_name}")

        negative_result = self.apply_negative(image)
        self._save_image(negative_result, base_name, "negative")

        false_color_result = self.apply_false_color(image)
        self._save_image(false_color_result, base_name, "false_color")

        skin_result = self.enhance_skin_regions(image)
        self._save_image(skin_result, base_name, "skin_enhanced")

        bilateral_result = self.apply_bilateral(image)
        self._save_image(bilateral_result, base_name, "bilateral")

        lbp_result = self.apply_lbp(image)
        self._save_image(lbp_result, base_name, "lbp")

        dct_result = self.apply_dct_blocks(image)
        self._save_image(dct_result, base_name, "dct_blocks")

        wavelet_result = self.apply_wavelet(image)
        self._save_image(wavelet_result, base_name, "wavelet")

        ssr_result = self.apply_single_scale_retinex(image)
        self._save_image(ssr_result, base_name, "retinex_ssr")

        msr_result = self.apply_multi_scale_retinex(image)
        self._save_image(msr_result, base_name, "retinex_msr")

        freq_result = self.apply_frequency_filter(image)
        self._save_image(freq_result, base_name, "freq_filter")

        freq_high = self.apply_frequency_filter(image, filter_type="high")
        self._save_image(freq_high, base_name, "freq_highpass")

        cross_pol = self.simulate_cross_polarization(image)
        self._save_image(cross_pol, base_name, "cross_polarized")

    def apply_negative(self, image: np.ndarray) -> np.ndarray:
        """
        Create negative/inverted image.
        
        Can reveal hidden patterns in skin tones.
        """
        return cv2.bitwise_not(image)

    def apply_false_color(self, image: np.ndarray) -> np.ndarray:
        """
        Apply false color mapping.
        
        Applies color gradients to grayscale to highlight variations.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        false_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        return false_color

    def enhance_skin_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance skin regions for focused analysis.
        
        Isolates skin tones using HSV range [0-33, 58-255, 30-255] per research.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 58, 30], dtype=np.uint8)
        upper_skin = np.array([33, 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        skin_region = cv2.bitwise_and(image, image, mask=skin_mask)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        skin_lab = cv2.cvtColor(skin_region, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(skin_lab)
        l_enhanced = clahe.apply(l_chan)
        enhanced_lab = cv2.merge([l_enhanced, a_chan, b_chan])
        enhanced_skin = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        result = image.copy()
        result[skin_mask > 0] = enhanced_skin[skin_mask > 0]

        return result

    def apply_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter.
        
        Edge-preserving smoothing that reveals subtle skin anomalies
        while reducing noise. Useful for detecting faint marks.
        """
        bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        diff = cv2.absdiff(image, bilateral)
        diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        return diff_enhanced

    def apply_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Local Binary Pattern (LBP) texture analysis (vectorized).
        
        Detects texture inconsistencies that may indicate manipulation
        or cloning. Each pixel compared to 8 neighbors.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.int16)
        rows, cols = gray.shape
        
        center = gray[1:rows-1, 1:cols-1]
        
        lbp = np.zeros((rows - 2, cols - 2), dtype=np.uint8)
        lbp |= ((gray[0:rows-2, 0:cols-2] >= center).astype(np.uint8) << 7)
        lbp |= ((gray[0:rows-2, 1:cols-1] >= center).astype(np.uint8) << 6)
        lbp |= ((gray[0:rows-2, 2:cols] >= center).astype(np.uint8) << 5)
        lbp |= ((gray[1:rows-1, 2:cols] >= center).astype(np.uint8) << 4)
        lbp |= ((gray[2:rows, 2:cols] >= center).astype(np.uint8) << 3)
        lbp |= ((gray[2:rows, 1:cols-1] >= center).astype(np.uint8) << 2)
        lbp |= ((gray[2:rows, 0:cols-2] >= center).astype(np.uint8) << 1)
        lbp |= ((gray[1:rows-1, 0:cols-2] >= center).astype(np.uint8) << 0)

        lbp_colored = cv2.applyColorMap(lbp, cv2.COLORMAP_VIRIDIS)
        return lbp_colored

    def apply_dct_blocks(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize DCT block boundaries (8x8 JPEG blocks).
        
        Highlights JPEG compression artifacts. Manipulated regions
        may show different block patterns or misalignment.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        gray = gray[:h8, :w8].astype(np.float32)

        block_energy = np.zeros((h8 // 8, w8 // 8), dtype=np.float32)

        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = gray[i:i+8, j:j+8]
                dct = cv2.dct(block)
                dct[0, 0] = 0
                block_energy[i // 8, j // 8] = np.sum(np.abs(dct))

        block_energy = cv2.normalize(block_energy, None, 0, 255, cv2.NORM_MINMAX)
        block_energy = block_energy.astype(np.uint8)
        block_energy = cv2.resize(block_energy, (w8, h8), interpolation=cv2.INTER_NEAREST)

        result = cv2.applyColorMap(block_energy, cv2.COLORMAP_HOT)
        return result

    def apply_wavelet(self, image: np.ndarray) -> np.ndarray:
        """
        Apply wavelet-like decomposition using Laplacian pyramid.
        
        Multi-scale analysis reveals details at different frequencies.
        Manipulation artifacts often appear at specific scales.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gaussian = gray.copy()
        layers = []

        for _ in range(4):
            down = cv2.pyrDown(gaussian)
            up = cv2.pyrUp(down, dstsize=(gaussian.shape[1], gaussian.shape[0]))
            laplacian = cv2.subtract(gaussian, up)
            layers.append(laplacian)
            gaussian = down

        result = np.zeros_like(gray, dtype=np.float32)
        weights = [0.4, 0.3, 0.2, 0.1]

        for layer, weight in zip(layers, weights):
            resized = cv2.resize(layer, (gray.shape[1], gray.shape[0]))
            result += weight * resized.astype(np.float32)

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        result = result.astype(np.uint8)

        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def apply_single_scale_retinex(self, image: np.ndarray, sigma: float = 80.0) -> np.ndarray:
        """
        Apply Single Scale Retinex (SSR).
        
        Enhances local contrast and removes illumination effects.
        Useful for seeing through semi-transparent materials or
        revealing details hidden by uneven lighting.
        
        Args:
            sigma: Gaussian blur sigma (larger = more smoothing)
        """
        img_float = image.astype(np.float32) + 1.0

        if len(image.shape) == 3:
            result = np.zeros_like(img_float)
            for i in range(3):
                blur = cv2.GaussianBlur(img_float[:, :, i], (0, 0), sigma)
                result[:, :, i] = np.log10(img_float[:, :, i]) - np.log10(blur + 1.0)
        else:
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
            result = np.log10(img_float) - np.log10(blur + 1.0)

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)

    def apply_multi_scale_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Multi-Scale Retinex (MSR).
        
        Combines multiple SSR at different scales for better detail
        enhancement across various feature sizes. Excellent for
        revealing bruises under varying skin tones and lighting.
        """
        sigmas = [15, 80, 250]
        weights = [1.0 / len(sigmas)] * len(sigmas)

        img_float = image.astype(np.float32) + 1.0

        if len(image.shape) == 3:
            result = np.zeros_like(img_float)
            for sigma, weight in zip(sigmas, weights):
                for i in range(3):
                    blur = cv2.GaussianBlur(img_float[:, :, i], (0, 0), sigma)
                    result[:, :, i] += weight * (np.log10(img_float[:, :, i]) - np.log10(blur + 1.0))
        else:
            result = np.zeros_like(img_float)
            for sigma, weight in zip(sigmas, weights):
                blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
                result += weight * (np.log10(img_float) - np.log10(blur + 1.0))

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)

    def apply_frequency_filter(self, image: np.ndarray, filter_type: str = "bandpass") -> np.ndarray:
        """
        Apply frequency domain filtering using FFT.
        
        Can remove periodic patterns (like fabric texture) while
        preserving anomalies underneath.
        
        Args:
            filter_type: "bandpass" or "high" for different filtering
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)

        if filter_type == "bandpass":
            inner_radius = 10
            outer_radius = min(rows, cols) // 4
            mask_2d = ((dist > inner_radius) & (dist < outer_radius)).astype(np.float32)
        else:
            radius = 30
            mask_2d = (dist > radius).astype(np.float32)

        mask = np.stack([mask_2d, mask_2d], axis=-1)

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        result = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def simulate_cross_polarization(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate cross-polarized light imaging.
        
        Cross-polarization reduces specular reflections and glare,
        revealing subsurface features like bruises that would
        otherwise be hidden by skin shine or clothing reflections.
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
        else:
            l_channel = image.copy()
            a_channel = np.zeros_like(l_channel)
            b_channel = np.zeros_like(l_channel)

        blur = cv2.GaussianBlur(l_channel, (0, 0), 5)
        specular = cv2.subtract(l_channel, blur)
        specular = cv2.threshold(specular, 20, 255, cv2.THRESH_BINARY)[1]

        specular_dilated = cv2.dilate(specular, np.ones((5, 5), np.uint8), iterations=2)
        inpainted = cv2.inpaint(l_channel, specular_dilated, 5, cv2.INPAINT_TELEA)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inpainted)

        if len(image.shape) == 3:
            result_lab = cv2.merge([enhanced, a_channel, b_channel])
            result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        else:
            result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return result


def main():
    """Main entry point."""
    base_images_dir = Path(r"D:\Development\forense\images")
    
    raw_folders = sorted([
        folder for folder in base_images_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("raw")
    ])
    
    if not raw_folders:
        logger.error("Nenhuma pasta 'raw*' encontrada em images/")
        return
    
    print("\nPastas disponíveis para processamento:")
    for idx, folder in enumerate(raw_folders, 1):
        print(f"  {idx}. {folder.name}")
    
    while True:
        try:
            choice = input("\nEscolha o número da pasta (ou 'q' para sair): ").strip()
            if choice.lower() == 'q':
                print("Cancelado.")
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(raw_folders):
                selected_folder = raw_folders[choice_idx]
                break
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número.")
    
    input_dir = str(selected_folder)
    
    processed_folders = sorted([
        folder for folder in base_images_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("processed")
    ])
    
    default_output = f"processed_{selected_folder.name}"
    
    print("\nPastas de saída existentes:")
    for idx, folder in enumerate(processed_folders, 1):
        print(f"  {idx}. {folder.name}")
    print(f"  0. Criar nova: {default_output}")
    
    while True:
        try:
            out_choice = input(f"\nEscolha o número da pasta de saída (ou 'q' para sair): ").strip()
            if out_choice.lower() == 'q':
                print("Cancelado.")
                return
            
            out_idx = int(out_choice)
            if out_idx == 0:
                output_dir = str(base_images_dir / default_output)
                output_name = default_output
                break
            elif 1 <= out_idx <= len(processed_folders):
                output_dir = str(processed_folders[out_idx - 1])
                output_name = processed_folders[out_idx - 1].name
                break
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número.")
    
    print(f"\nProcessando: {selected_folder.name}")
    print(f"Saída: {output_name}\n")

    try:
        processor = ForensicImageProcessor(input_dir, output_dir)
        processor.process_all_images()
        logger.debug("Processing complete")
        print("\nProcessamento concluído!")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
