#!/usr/bin/env python3
"""
Forensic Report Generator

Analyzes processed and suspect images, generates an HTML report
with findings grouped by original raw image.
"""

import os
import logging
import base64
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re

import cv2
import numpy as np

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Analysis data for a single raw image."""
    raw_name: str
    raw_path: str
    processed_images: List[str] = field(default_factory=list)
    suspect_images: List[str] = field(default_factory=list)
    overlay_path: Optional[str] = None
    individual_overlays: List[Dict] = field(default_factory=list)
    zoom_crops: List[Dict] = field(default_factory=list)
    ai_analyses: Dict[str, str] = field(default_factory=dict)
    
    @property
    def has_suspects(self) -> bool:
        return len(self.suspect_images) > 0
    
    @property
    def suspect_count(self) -> int:
        return len(self.suspect_images)
    
    def get_verdict(self) -> str:
        if not self.has_suspects:
            return "NENHUMA EVIDÊNCIA DETECTADA"
        elif self.suspect_count >= 5:
            return "ALTA PROBABILIDADE DE EVIDÊNCIA"
        elif self.suspect_count >= 3:
            return "MÉDIA PROBABILIDADE DE EVIDÊNCIA"
        else:
            return "BAIXA PROBABILIDADE DE EVIDÊNCIA"
    
    def get_verdict_class(self) -> str:
        if not self.has_suspects:
            return "verdict-none"
        elif self.suspect_count >= 5:
            return "verdict-high"
        elif self.suspect_count >= 3:
            return "verdict-medium"
        else:
            return "verdict-low"


TECHNIQUE_DESCRIPTIONS = {
    "clahe": "CLAHE - Realce de contraste local adaptativo",
    "clahe_1.0": "CLAHE 1.0 - Realce suave",
    "clahe_4.0": "CLAHE 4.0 - Realce forte",
    "clahe_8.0": "CLAHE 8.0 - Realce máximo",
    "clahe_16x16": "CLAHE 16x16 - Realce com tiles maiores",
    "histogram_eq": "Equalização de Histograma - Realce global de contraste",
    "gamma_0.2": "Correção Gamma 0.2 - Clareia extremo sombras",
    "gamma_0.3": "Correção Gamma 0.3 - Clareia muito sombras",
    "gamma_0.4": "Correção Gamma 0.4 - Clareia sombras forte",
    "gamma_0.5": "Correção Gamma 0.5 - Clareia sombras",
    "gamma_0.7": "Correção Gamma 0.7 - Clareia sombras suave",
    "gamma_1.5": "Correção Gamma 1.5 - Escurece highlights",
    "gamma_2.0": "Correção Gamma 2.0 - Forte escurecimento",
    "gamma_2.5": "Correção Gamma 2.5 - Escurecimento intenso",
    "gamma_3.0": "Correção Gamma 3.0 - Escurecimento máximo",
    "gamma_4.0": "Correção Gamma 4.0 - Escurecimento extremo",
    "canny": "Detecção Canny - Bordas e contornos",
    "canny_sensitive": "Canny Sensível - Detecta bordas sutis",
    "canny_balanced": "Canny Balanceado - Detecção moderada",
    "canny_strict": "Canny Restrito - Apenas bordas fortes",
    "sobel_x": "Sobel X - Gradientes horizontais",
    "sobel_y": "Sobel Y - Gradientes verticais",
    "sobel_combined": "Sobel Combinado - Todas as direções",
    "sobel_5x5": "Sobel 5x5 - Bordas mais amplas",
    "laplacian": "Laplaciano - Detalhes finos",
    "laplacian_5": "Laplaciano 5 - Detalhes mais amplos",
    "gradient_magnitude": "Magnitude do Gradiente - Mudanças de intensidade",
    "dog": "DoG - Diferença de Gaussianas (multi-escala)",
    "morph_gradient": "Gradiente Morfológico - Bordas de lesões",
    "red_channel": "Canal Vermelho - Isolado",
    "green_channel": "Canal Verde - Isolado",
    "blue_channel": "Canal Azul - Isolado",
    "ir_simulation": "Simulação Infravermelha - Revela hematomas",
    "ms_480nm": "Multiespectral 480nm - Hematomas recentes (azul)",
    "ms_620nm": "Multiespectral 620nm - Hematomas em cicatrização",
    "ms_850nm": "Multiespectral 850nm - Hematomas profundos/antigos",
    "hemoglobin": "Hemoglobina - Detecta degradação sanguínea",
    "als_365nm": "ALS 365nm - Luz UV-A (Wood's lamp)",
    "als_415nm": "ALS 415nm - Luz alternativa violeta",
    "als_450nm": "ALS 450nm - Luz alternativa azul",
    "hsv_h": "HSV Matiz - Tonalidade de cor",
    "hsv_s": "HSV Saturação - Intensidade de cor",
    "hsv_v": "HSV Valor - Luminosidade",
    "lab_l": "LAB Luminosidade",
    "lab_a": "LAB Canal A - Verde/Vermelho",
    "lab_b": "LAB Canal B - Azul/Amarelo",
    "ycrcb_y": "YCrCb Luminância",
    "ycrcb_cr": "YCrCb Crominância Vermelha",
    "ycrcb_cb": "YCrCb Crominância Azul",
    "unsharp": "Máscara de Nitidez - Realce de detalhes",
    "highpass": "Filtro Passa-Alta - Texturas e marcas",
    "emboss": "Relevo - Textura de superfície",
    "median": "Filtro Mediana - Remove ruído preservando bordas",
    "negative": "Negativo - Padrões ocultos",
    "false_color": "Cores Falsas - Mapeamento de variações",
    "skin_enhanced": "Pele Realçada - Foco em regiões de pele",
    "bilateral": "Filtro Bilateral - Revela anomalias sutis preservando bordas",
    "lbp": "LBP - Padrão Binário Local para detecção de texturas",
    "dct_blocks": "Blocos DCT - Visualiza artefatos de compressão JPEG",
    "wavelet": "Wavelet - Análise multi-escala de detalhes",
    "retinex_ssr": "Retinex SSR - Visibilidade através de tecidos",
    "retinex_msr": "Retinex MSR - Revela detalhes sob iluminação variada",
    "freq_filter": "Filtro Frequência - Remove padrões de tecido",
    "freq_highpass": "Filtro Frequência Alta - Destaca anomalias",
    "cross_polarized": "Polarização Cruzada - Remove reflexos",
    "ela_suspect": "ELA - Análise de Nível de Erro (detecta edições JPEG)",
    "noise_suspect": "Análise de Ruído - Detecta inconsistências de ruído",
    "jpeg_ghost_suspect": "JPEG Ghost - Detecta regiões com compressão diferente",
}

TECHNIQUE_CATEGORIES = {
    "Realce de Contraste": ["clahe", "clahe_1.0", "clahe_4.0", "clahe_8.0", "clahe_16x16", "histogram_eq", 
                           "gamma_0.2", "gamma_0.3", "gamma_0.4", "gamma_0.5", "gamma_0.7", "gamma_1.5", 
                           "gamma_2.0", "gamma_2.5", "gamma_3.0", "gamma_4.0"],
    "Detecção de Bordas": ["canny", "canny_sensitive", "canny_balanced", "canny_strict",
                          "sobel_x", "sobel_y", "sobel_combined", "sobel_5x5", "laplacian", "laplacian_5",
                          "gradient_magnitude", "dog", "morph_gradient", "dct_blocks", "wavelet"],
    "Canais RGB": ["red_channel", "green_channel", "blue_channel"],
    "Análise HSV": ["hsv_h", "hsv_s", "hsv_v"],
    "Análise LAB": ["lab_l", "lab_a", "lab_b"],
    "Análise YCrCb": ["ycrcb_y", "ycrcb_cr", "ycrcb_cb"],
    "Simulação IR/Multiespectral": ["ir_simulation", "ms_480nm", "ms_620nm", "ms_850nm", 
                                    "hemoglobin", "als_365nm", "als_415nm", "als_450nm"],
    "Realce de Detalhes": ["unsharp", "highpass", "emboss", "median"],
    "Filtros Especiais": ["negative", "false_color", "skin_enhanced", "bilateral", "lbp",
                         "retinex_ssr", "retinex_msr", "freq_filter", "freq_highpass", "cross_polarized"],
    "Análise Forense": ["ela_suspect", "noise_suspect", "jpeg_ghost_suspect"],
}


DEFAULT_SUSPECT_FOLDER = "suspect_fast_v2"


class ForensicReportGenerator:
    """Generates HTML report from processed forensic images."""

    def __init__(
        self,
        base_dir: str,
        raw_folder: str = "raw",
        processed_folder: str = "processed",
        suspect_folder: str = DEFAULT_SUSPECT_FOLDER,
        enable_ai_analysis: bool = False
    ):
        self.base_dir = Path(base_dir)
        self.raw_folder = raw_folder
        self.raw_dir = self.base_dir / "images" / raw_folder
        self.processed_dir = self.base_dir / "images" / processed_folder
        self.suspect_dir = self.base_dir / "images" / suspect_folder
        self.overlay_dir = self.base_dir / "images" / f"overlays_{raw_folder}"
        self.output_file = self.base_dir / f"index_{raw_folder}.html"
        self.ai_analysis_file = self.base_dir / f"ai_analysis_{raw_folder}.json"
        
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyses: Dict[str, ImageAnalysis] = {}
        self.enable_ai_analysis = enable_ai_analysis
        self.ai_client = None
        
        if self.enable_ai_analysis and ANTHROPIC_AVAILABLE:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found. AI analysis disabled.")
                self.enable_ai_analysis = False
            else:
                self.ai_client = anthropic.Anthropic(api_key=api_key)

    def generate_report(self) -> None:
        """Generate the complete HTML report."""
        logger.debug("Starting report generation")
        
        self._scan_raw_images()
        self._scan_processed_images()
        self._scan_suspect_images()
        self._generate_overlay_images()
        
        self._load_ai_analyses_from_json()
        
        if self.enable_ai_analysis:
            self._perform_ai_analysis()
        
        self._generate_html()
        
        logger.debug(f"Report generated: {self.output_file}")

    def _load_ai_analyses_from_json(self) -> None:
        """Load AI analyses from JSON file if it exists."""
        if not self.ai_analysis_file.exists():
            logger.debug(f"No AI analysis file found at {self.ai_analysis_file}")
            return
        
        try:
            with open(self.ai_analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            loaded_count = 0
            for base_name, ai_data in data.get("analyses", {}).items():
                if base_name in self.analyses:
                    for suspect_path, analysis_text in ai_data.items():
                        self.analyses[base_name].ai_analyses[suspect_path] = analysis_text
                        loaded_count += 1
            
            logger.debug(f"Loaded {loaded_count} AI analyses from {self.ai_analysis_file}")
            print(f"  ✓ Carregadas {loaded_count} análises de IA do arquivo JSON")
        except Exception as e:
            logger.error(f"Error loading AI analyses from JSON: {e}")

    def export_images_for_analysis(self) -> str:
        """Export list of images that need AI analysis to a JSON file."""
        self._scan_raw_images()
        self._scan_suspect_images()
        
        export_data = {
            "raw_folder": self.raw_folder,
            "base_dir": str(self.base_dir),
            "images_to_analyze": []
        }
        
        for base_name, analysis in self.analyses.items():
            if not analysis.has_suspects:
                continue
            
            raw_full_path = self.base_dir / analysis.raw_path
            
            for suspect_path in analysis.suspect_images:
                technique = self._extract_technique(suspect_path)
                technique_desc = TECHNIQUE_DESCRIPTIONS.get(technique, technique)
                suspect_full_path = self.base_dir / suspect_path
                
                export_data["images_to_analyze"].append({
                    "base_name": base_name,
                    "raw_path": str(raw_full_path).replace("\\", "/"),
                    "suspect_path": str(suspect_full_path).replace("\\", "/"),
                    "suspect_rel_path": suspect_path,
                    "technique": technique,
                    "technique_description": technique_desc
                })
        
        export_file = self.base_dir / f"images_to_analyze_{self.raw_folder}.json"
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Exported {len(export_data['images_to_analyze'])} images for analysis to {export_file}")
        return str(export_file)

    def _scan_raw_images(self) -> None:
        """Scan raw images directory."""
        if not self.raw_dir.exists():
            logger.warning(f"Raw directory not found: {self.raw_dir}")
            return

        for img_path in self.raw_dir.iterdir():
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                base_name = img_path.stem
                rel_path = img_path.relative_to(self.base_dir)
                self.analyses[base_name] = ImageAnalysis(
                    raw_name=base_name,
                    raw_path=str(rel_path).replace("\\", "/")
                )
        
        logger.debug(f"Found {len(self.analyses)} raw images")

    def _extract_base_name(self, filename: str) -> str:
        """Extract base name from processed filename."""
        stem = Path(filename).stem
        
        if stem.endswith("_ela_suspect"):
            return stem[:-len("_ela_suspect")]
        if stem.endswith("_noise_suspect"):
            return stem[:-len("_noise_suspect")]
        if stem.endswith("_jpeg_ghost_suspect"):
            return stem[:-len("_jpeg_ghost_suspect")]
        
        for technique in TECHNIQUE_DESCRIPTIONS.keys():
            suffix = f"_{technique}"
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        
        return stem

    def _extract_technique(self, filename: str) -> str:
        """Extract technique name from processed filename."""
        stem = Path(filename).stem
        
        if stem.endswith("_ela_suspect"):
            return "ela_suspect"
        if stem.endswith("_noise_suspect"):
            return "noise_suspect"
        if stem.endswith("_jpeg_ghost_suspect"):
            return "jpeg_ghost_suspect"
        
        for technique in TECHNIQUE_DESCRIPTIONS.keys():
            suffix = f"_{technique}"
            if stem.endswith(suffix):
                return technique
        
        return "unknown"

    def _scan_processed_images(self) -> None:
        """Scan processed images directory."""
        if not self.processed_dir.exists():
            logger.warning(f"Processed directory not found: {self.processed_dir}")
            return

        processed_total = 0
        matched_total = 0

        for img_path in self.processed_dir.iterdir():
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                processed_total += 1
                base_name = self._extract_base_name(img_path.name)
                rel_path = img_path.relative_to(self.base_dir)
                
                if base_name in self.analyses:
                    self.analyses[base_name].processed_images.append(
                        str(rel_path).replace("\\", "/")
                    )
                    matched_total += 1

        if processed_total > 0:
            unmatched_total = processed_total - matched_total
            logger.debug(
                f"Found {processed_total} processed images, {matched_total} linked to raw images"
            )
            if unmatched_total > 0:
                logger.debug(f"{unmatched_total} processed images without raw match")

    def _scan_suspect_images(self) -> None:
        """Scan suspect images directory."""
        if not self.suspect_dir.exists():
            logger.warning(f"Suspect directory not found: {self.suspect_dir}")
            return

        for img_path in self.suspect_dir.iterdir():
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                base_name = self._extract_base_name(img_path.name)
                rel_path = img_path.relative_to(self.base_dir)
                
                if base_name in self.analyses:
                    self.analyses[base_name].suspect_images.append(
                        str(rel_path).replace("\\", "/")
                    )
        
        suspect_count = sum(1 for a in self.analyses.values() if a.has_suspects)
        logger.debug(f"Found {suspect_count} images with suspect findings")

    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as f:
                return base64.standard_b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def _get_media_type(self, image_path: Path) -> str:
        """Get media type from image extension."""
        ext = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(ext, "image/jpeg")

    def _perform_ai_analysis(self) -> None:
        """Perform AI analysis on each suspect finding."""
        if not self.ai_client:
            logger.warning("AI client not initialized. Skipping AI analysis.")
            return
        
        total_suspects = sum(len(a.suspect_images) for a in self.analyses.values() if a.has_suspects)
        logger.debug(f"Starting AI analysis for {total_suspects} suspect images")
        
        processed_count = 0
        
        for base_name, analysis in self.analyses.items():
            if not analysis.has_suspects:
                continue
            
            raw_full_path = self.base_dir / analysis.raw_path
            raw_base64 = self._encode_image_to_base64(raw_full_path)
            raw_media_type = self._get_media_type(raw_full_path)
            
            if not raw_base64:
                logger.warning(f"Could not encode original image: {raw_full_path}")
                continue
            
            for suspect_path in analysis.suspect_images:
                technique = self._extract_technique(suspect_path)
                technique_desc = TECHNIQUE_DESCRIPTIONS.get(technique, technique)
                
                suspect_full_path = self.base_dir / suspect_path
                suspect_base64 = self._encode_image_to_base64(suspect_full_path)
                suspect_media_type = self._get_media_type(suspect_full_path)
                
                if not suspect_base64:
                    logger.warning(f"Could not encode suspect image: {suspect_full_path}")
                    continue
                
                try:
                    ai_response = self._call_ai_for_analysis(
                        raw_base64, raw_media_type,
                        suspect_base64, suspect_media_type,
                        technique, technique_desc
                    )
                    
                    if ai_response:
                        analysis.ai_analyses[suspect_path] = ai_response
                        processed_count += 1
                        logger.debug(f"AI analysis completed for {base_name} - {technique}")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"AI analysis failed for {base_name} - {technique}: {e}")
        
        logger.debug(f"AI analysis completed: {processed_count}/{total_suspects} images analyzed")

    def _call_ai_for_analysis(
        self,
        raw_base64: str,
        raw_media_type: str,
        suspect_base64: str,
        suspect_media_type: str,
        technique: str,
        technique_desc: str
    ) -> Optional[str]:
        """Call AI API to analyze the suspect finding."""
        if not self.ai_client:
            return None
        
        prompt = f"""Você é um especialista em análise forense de imagens. Analise as duas imagens fornecidas:

1. IMAGEM ORIGINAL: A primeira imagem é a foto original sem processamento.
2. IMAGEM PROCESSADA: A segunda imagem foi processada com a técnica "{technique}" ({technique_desc}).

Esta imagem processada foi marcada como SUSPEITA pelo sistema de detecção automática.

Por favor, analise e forneça sua opinião profissional sobre:
- O que você observa na imagem processada que pode indicar evidência relevante
- Se há indicações de lesões, hematomas, marcas ou alterações na pele
- Qual a probabilidade (baixa/média/alta) de haver evidência forense relevante
- Qualquer observação adicional que possa ser útil para a análise forense

Responda de forma concisa e objetiva em português, em no máximo 3-4 frases."""

        try:
            message = self.ai_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": raw_media_type,
                                    "data": raw_base64
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": suspect_media_type,
                                    "data": suspect_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            if message.content and len(message.content) > 0:
                return message.content[0].text
            
            return None
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return None

    def _generate_overlay_images(self) -> None:
        """Generate overlay images combining original with suspect findings."""
        logger.debug("Generating overlay images")
        
        for base_name, analysis in self.analyses.items():
            if not analysis.has_suspects:
                continue
            
            try:
                raw_full_path = self.base_dir / analysis.raw_path
                original = cv2.imread(str(raw_full_path))
                if original is None:
                    logger.warning(f"Could not read original image: {raw_full_path}")
                    continue
                
                overlay, all_regions = self._create_overlay(analysis, original)
                if overlay is not None:
                    overlay_filename = f"{base_name}_overlay.jpg"
                    overlay_path = self.overlay_dir / overlay_filename
                    cv2.imwrite(str(overlay_path), overlay)
                    
                    rel_path = overlay_path.relative_to(self.base_dir)
                    analysis.overlay_path = str(rel_path).replace("\\", "/")
                    logger.debug(f"Created overlay: {overlay_filename}")
                
                self._create_individual_overlays(analysis, original)
                
                self._create_zoom_crops(analysis, original, all_regions)
                
            except Exception as e:
                logger.error(f"Error creating overlay for {base_name}: {e}")

    def _create_individual_overlays(self, analysis: ImageAnalysis, original: np.ndarray) -> None:
        """Create separate overlay for each suspect technique."""
        height, width = original.shape[:2]
        edge_techniques = {'canny', 'sobel_x', 'sobel_y', 'sobel_combined', 'laplacian', 'gradient_magnitude', 'highpass', 'emboss'}
        
        for suspect_path in analysis.suspect_images:
            technique = self._extract_technique(suspect_path)
            suspect_full_path = self.base_dir / suspect_path
            suspect_img = cv2.imread(str(suspect_full_path))
            
            if suspect_img is None:
                continue
            
            if suspect_img.shape[:2] != (height, width):
                suspect_img = cv2.resize(suspect_img, (width, height))
            
            gray = cv2.cvtColor(suspect_img, cv2.COLOR_BGR2GRAY)
            
            if technique in edge_techniques:
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                mask = thresh.astype(np.float32) / 255.0
            else:
                diff = cv2.absdiff(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), gray)
                diff_normalized = diff.astype(np.float32) / 255.0
                mask = np.clip(diff_normalized * 2, 0, 1)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            overlay = original.copy().astype(np.float32)
            highlight_color = np.array([0, 165, 255], dtype=np.float32)
            
            overlay_intensity = 0.25
            for c in range(3):
                overlay[:, :, c] = (
                    original[:, :, c].astype(np.float32) * (1 - mask * overlay_intensity) +
                    highlight_color[c] * mask * overlay_intensity
                )
            
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            contours_mask = (mask > 0.2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(contours_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
            
            label_height = 35
            labeled = np.zeros((height + label_height, width, 3), dtype=np.uint8)
            labeled[:label_height, :] = (40, 30, 20)
            labeled[label_height:, :] = overlay
            
            desc = TECHNIQUE_DESCRIPTIONS.get(technique, technique)
            text = f"{technique.upper()}: {desc[:50]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            cv2.putText(labeled, text, (10, 24), font, font_scale, (0, 200, 255), thickness)
            
            filename = f"{analysis.raw_name}_overlay_{technique}.jpg"
            filepath = self.overlay_dir / filename
            cv2.imwrite(str(filepath), labeled)
            
            rel_path = filepath.relative_to(self.base_dir)
            analysis.individual_overlays.append({
                'path': str(rel_path).replace("\\", "/"),
                'technique': technique,
                'description': TECHNIQUE_DESCRIPTIONS.get(technique, technique)
            })
            logger.debug(f"Created individual overlay: {filename}")

    def _create_zoom_crops(self, analysis: ImageAnalysis, original: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> None:
        """Create zoomed crops of detected regions."""
        height, width = original.shape[:2]
        
        merged_regions = self._merge_overlapping_regions(regions, width, height)
        
        for idx, (x, y, w, h) in enumerate(merged_regions[:5]):
            padding = max(w, h) // 3
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            crop = original[y1:y2, x1:x2].copy()
            
            if crop.size == 0:
                continue
            
            crop_h, crop_w = crop.shape[:2]
            min_size = 300
            if crop_w < min_size or crop_h < min_size:
                scale = max(min_size / crop_w, min_size / crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            crop_h, crop_w = crop.shape[:2]
            
            cv2.rectangle(crop, (padding, padding), (crop_w - padding, crop_h - padding), (0, 255, 255), 2)
            
            label_height = 30
            labeled = np.zeros((crop_h + label_height, crop_w, 3), dtype=np.uint8)
            labeled[:label_height, :] = (50, 30, 30)
            labeled[label_height:, :] = crop
            
            text = f"ZOOM #{idx + 1} - Regiao detectada"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(labeled, text, (10, 22), font, 0.5, (100, 200, 255), 1)
            
            filename = f"{analysis.raw_name}_zoom_{idx + 1}.jpg"
            filepath = self.overlay_dir / filename
            cv2.imwrite(str(filepath), labeled)
            
            rel_path = filepath.relative_to(self.base_dir)
            analysis.zoom_crops.append({
                'path': str(rel_path).replace("\\", "/"),
                'region': (x, y, w, h),
                'index': idx + 1
            })
            logger.debug(f"Created zoom crop: {filename}")

    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]], img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes."""
        if not regions:
            return []
        
        min_area = (img_width * img_height) * 0.001
        filtered = [(x, y, w, h) for x, y, w, h in regions if w * h >= min_area]
        
        if not filtered:
            return []
        
        merged = []
        used = [False] * len(filtered)
        
        for i, (x1, y1, w1, h1) in enumerate(filtered):
            if used[i]:
                continue
            
            current = [x1, y1, x1 + w1, y1 + h1]
            used[i] = True
            
            changed = True
            while changed:
                changed = False
                for j, (x2, y2, w2, h2) in enumerate(filtered):
                    if used[j]:
                        continue
                    
                    box2 = [x2, y2, x2 + w2, y2 + h2]
                    
                    if self._boxes_overlap(current, box2):
                        current[0] = min(current[0], box2[0])
                        current[1] = min(current[1], box2[1])
                        current[2] = max(current[2], box2[2])
                        current[3] = max(current[3], box2[3])
                        used[j] = True
                        changed = True
            
            merged.append((current[0], current[1], current[2] - current[0], current[3] - current[1]))
        
        merged.sort(key=lambda r: r[2] * r[3], reverse=True)
        return merged

    def _boxes_overlap(self, box1: List[int], box2: List[int]) -> bool:
        """Check if two boxes overlap."""
        margin = 20
        return not (box1[2] + margin < box2[0] or box2[2] + margin < box1[0] or
                    box1[3] + margin < box2[1] or box2[3] + margin < box1[1])

    def _create_overlay(self, analysis: ImageAnalysis, original: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """Create an overlay image combining original with suspect findings."""
        height, width = original.shape[:2]
        
        combined_mask = np.zeros((height, width), dtype=np.float32)
        all_regions = []
        
        edge_techniques = {'canny', 'sobel_x', 'sobel_y', 'sobel_combined', 'laplacian', 'gradient_magnitude', 'highpass', 'emboss'}
        
        for suspect_path in analysis.suspect_images:
            technique = self._extract_technique(suspect_path)
            suspect_full_path = self.base_dir / suspect_path
            suspect_img = cv2.imread(str(suspect_full_path))
            
            if suspect_img is None:
                continue
            
            if suspect_img.shape[:2] != (height, width):
                suspect_img = cv2.resize(suspect_img, (width, height))
            
            gray = cv2.cvtColor(suspect_img, cv2.COLOR_BGR2GRAY)
            
            if technique in edge_techniques:
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                mask = thresh.astype(np.float32) / 255.0
            else:
                diff = cv2.absdiff(
                    cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
                    gray
                )
                diff_normalized = diff.astype(np.float32) / 255.0
                mask = np.clip(diff_normalized * 2, 0, 1)
            
            contours_temp = (mask > 0.2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(contours_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                all_regions.append((x, y, w, h))
            
            combined_mask = np.maximum(combined_mask, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        
        overlay = original.copy().astype(np.float32)
        
        highlight_color = np.array([0, 165, 255], dtype=np.float32)
        
        overlay_intensity = 0.25
        for c in range(3):
            overlay[:, :, c] = (
                original[:, :, c].astype(np.float32) * (1 - combined_mask * overlay_intensity) +
                highlight_color[c] * combined_mask * overlay_intensity
            )
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        contours_mask = (combined_mask > 0.2).astype(np.uint8) * 255
        contours, _ = cv2.findContours(contours_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        
        label_height = 40
        labeled = np.zeros((height + label_height, width, 3), dtype=np.uint8)
        labeled[:label_height, :] = (30, 30, 50)
        labeled[label_height:, :] = overlay
        
        text = f"OVERLAY COMBINADO - {analysis.suspect_count} achados suspeitos"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        cv2.putText(labeled, text, (text_x, text_y), font, font_scale, (0, 200, 255), thickness)
        
        return labeled, all_regions

    def _generate_html(self) -> None:
        """Generate the HTML report."""
        sorted_analyses = sorted(
            self.analyses.values(),
            key=lambda x: (-x.suspect_count, x.raw_name)
        )
        
        total_with_evidence = sum(1 for a in sorted_analyses if a.has_suspects)
        total_suspects = sum(a.suspect_count for a in sorted_analyses)
        
        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Análise Forense</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            padding: 30px;
            text-align: center;
            border-bottom: 3px solid #e94560;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            color: #fff;
        }}
        
        .header .subtitle {{
            color: #aaa;
            font-size: 1.1em;
        }}
        
        .summary {{
            background: #16213e;
            padding: 20px 30px;
            display: flex;
            justify-content: center;
            gap: 50px;
            flex-wrap: wrap;
            border-bottom: 1px solid #333;
        }}
        
        .summary-item {{
            text-align: center;
        }}
        
        .summary-item .number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #e94560;
        }}
        
        .summary-item .label {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .filter-bar {{
            background: #0f3460;
            padding: 15px 30px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 25px;
            flex-wrap: wrap;
            border-bottom: 1px solid #333;
        }}
        
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .filter-group label {{
            color: #aaa;
            font-size: 0.9em;
        }}
        
        .filter-group select,
        .filter-group input[type="number"],
        .filter-group input[type="text"] {{
            background: #1a1a2e;
            border: 1px solid #333;
            color: #fff;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        
        .filter-group input[type="number"] {{
            width: 70px;
        }}
        
        .filter-group input[type="text"] {{
            width: 180px;
        }}
        
        .filter-group button {{
            background: #e94560;
            border: none;
            color: #fff;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.2s;
        }}
        
        .filter-group button:hover {{
            background: #d63850;
        }}
        
        .filter-results {{
            color: #aaa;
            font-size: 0.9em;
            margin-left: 20px;
        }}
        
        .image-section.hidden {{
            display: none;
        }}
        
        .image-section.collapsed .section-content {{
            display: none;
        }}
        
        .image-section.collapsed .collapse-icon {{
            transform: rotate(-90deg);
        }}
        
        .collapse-icon {{
            display: inline-block;
            transition: transform 0.2s;
            margin-right: 8px;
        }}
        
        .section-header {{
            cursor: pointer;
        }}
        
        .section-header:hover {{
            background: #1a4a70;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .image-section {{
            background: #16213e;
            border-radius: 10px;
            margin-bottom: 30px;
            overflow: hidden;
            border: 1px solid #333;
        }}
        
        .section-header {{
            background: #0f3460;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .section-title {{
            font-size: 1.3em;
            color: #fff;
        }}
        
        .verdict {{
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .verdict-none {{
            background: #2d4a3e;
            color: #4ade80;
        }}
        
        .verdict-low {{
            background: #4a4a2d;
            color: #facc15;
        }}
        
        .verdict-medium {{
            background: #4a3a2d;
            color: #fb923c;
        }}
        
        .verdict-high {{
            background: #4a2d2d;
            color: #f87171;
        }}
        
        .section-content {{
            padding: 20px;
        }}
        
        .images-comparison {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .raw-image-container,
        .overlay-image-container {{
            text-align: center;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 8px;
            flex: 0 1 auto;
        }}
        
        .raw-image-container img,
        .overlay-image-container img {{
            max-width: 500px;
            max-height: 400px;
            border-radius: 5px;
            border: 2px solid #333;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        
        .raw-image-container img:hover,
        .overlay-image-container img:hover {{
            transform: scale(1.02);
        }}
        
        .overlay-image-container {{
            border: 2px solid #e94560;
        }}
        
        .overlay-image-container img {{
            border-color: #e94560;
        }}
        
        .raw-image-container .label,
        .overlay-image-container .label {{
            margin-top: 10px;
            color: #888;
            font-size: 0.9em;
        }}
        
        .overlay-image-container .label {{
            color: #e94560;
            font-weight: bold;
        }}
        
        .suspect-section {{
            background: rgba(233, 69, 96, 0.1);
            border: 2px solid #e94560;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .suspect-section h3 {{
            color: #e94560;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .suspect-section h3::before {{
            content: "⚠";
        }}
        
        .suspect-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }}
        
        .suspect-item {{
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e94560;
        }}
        
        .suspect-item img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: #000;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        
        .suspect-item img:hover {{
            transform: scale(1.02);
        }}
        
        .suspect-item .info {{
            padding: 10px;
            font-size: 0.85em;
        }}
        
        .suspect-item .technique {{
            color: #e94560;
            font-weight: bold;
        }}
        
        .suspect-item .description {{
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .suspect-item .ai-analysis {{
            background: rgba(138, 43, 226, 0.15);
            border: 1px solid #8a2be2;
            border-radius: 5px;
            padding: 8px;
            margin-top: 8px;
            font-size: 0.85em;
            color: #d8b4fe;
        }}
        
        .suspect-item .ai-analysis-label {{
            color: #a855f7;
            font-weight: bold;
            font-size: 0.8em;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .suspect-item .ai-analysis-label::before {{
            content: "🤖";
        }}
        
        .processed-section {{
            margin-top: 20px;
        }}
        
        .processed-section h3 {{
            color: #888;
            margin-bottom: 15px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .processed-section h3:hover {{
            color: #aaa;
        }}
        
        .processed-section h3::before {{
            content: "▶";
            font-size: 0.8em;
            transition: transform 0.2s;
        }}
        
        .processed-section.expanded h3::before {{
            transform: rotate(90deg);
        }}
        
        .category {{
            margin-bottom: 20px;
        }}
        
        .category-title {{
            color: #0f9;
            font-size: 1em;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #333;
        }}
        
        .processed-grid {{
            display: none;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}
        
        .processed-section.expanded .processed-grid {{
            display: grid;
        }}
        
        .processed-item {{
            background: #1a1a2e;
            border-radius: 5px;
            overflow: hidden;
            border: 1px solid #333;
        }}
        
        .processed-item.is-suspect {{
            border-color: #e94560;
        }}
        
        .processed-item img {{
            width: 100%;
            height: 150px;
            object-fit: contain;
            background: #000;
            cursor: pointer;
        }}
        
        .processed-item .info {{
            padding: 8px;
            font-size: 0.75em;
            color: #888;
        }}
        
        .no-evidence {{
            text-align: center;
            padding: 30px;
            color: #4ade80;
        }}
        
        .no-evidence .icon {{
            font-size: 3em;
            margin-bottom: 10px;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        
        .modal.active {{
            display: flex;
        }}
        
        .modal img {{
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }}
        
        .modal-close {{
            position: fixed;
            top: 20px;
            right: 30px;
            font-size: 2em;
            color: #fff;
            cursor: pointer;
            z-index: 1002;
        }}
        
        .modal-nav {{
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            font-size: 3em;
            color: #fff;
            cursor: pointer;
            padding: 20px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 5px;
            user-select: none;
            z-index: 1002;
            transition: background 0.2s;
        }}
        
        .modal-nav:hover {{
            background: rgba(233, 69, 96, 0.8);
        }}
        
        .modal-prev {{
            left: 20px;
        }}
        
        .modal-next {{
            right: 20px;
        }}
        
        .modal-counter {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: #fff;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9em;
            z-index: 1002;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 1px solid #333;
            margin-top: 30px;
        }}
        
        .individual-overlays {{
            background: rgba(255, 165, 0, 0.1);
            border: 1px solid #ffa500;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .individual-overlays h3 {{
            color: #ffa500;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .individual-overlays h3::before {{
            content: "🔍";
        }}
        
        .individual-overlays-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 15px;
        }}
        
        .individual-overlay-item {{
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #ffa500;
        }}
        
        .individual-overlay-item img {{
            width: 100%;
            height: 280px;
            object-fit: contain;
            background: #000;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        
        .individual-overlay-item img:hover {{
            transform: scale(1.02);
        }}
        
        .individual-overlay-item .info {{
            padding: 10px;
            font-size: 0.85em;
        }}
        
        .individual-overlay-item .technique {{
            color: #ffa500;
            font-weight: bold;
        }}
        
        .zoom-section {{
            background: rgba(100, 200, 255, 0.1);
            border: 1px solid #64c8ff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .zoom-section h3 {{
            color: #64c8ff;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .zoom-section h3::before {{
            content: "🔎";
        }}
        
        .zoom-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .zoom-item {{
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #64c8ff;
        }}
        
        .zoom-item img {{
            width: 100%;
            height: 250px;
            object-fit: contain;
            background: #000;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        
        .zoom-item img:hover {{
            transform: scale(1.05);
        }}
        
        .zoom-item .info {{
            padding: 10px;
            font-size: 0.85em;
            color: #64c8ff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Relatório de Análise Forense</h1>
        <p class="subtitle">Análise de imagens processadas com técnicas forenses</p>
    </div>
    
    <div class="summary">
        <div class="summary-item">
            <div class="number">{len(sorted_analyses)}</div>
            <div class="label">Imagens Analisadas</div>
        </div>
        <div class="summary-item">
            <div class="number">{total_with_evidence}</div>
            <div class="label">Com Evidências Suspeitas</div>
        </div>
        <div class="summary-item">
            <div class="number">{total_suspects}</div>
            <div class="label">Total de Achados</div>
        </div>
    </div>
    
    <div class="filter-bar">
        <div class="filter-group">
            <label>Filtrar por Prioridade:</label>
            <select id="priorityFilter" onchange="applyFilters()">
                <option value="all">Todas</option>
                <option value="high">Alta (5+ achados)</option>
                <option value="medium">Média (3-4 achados)</option>
                <option value="low">Baixa (1-2 achados)</option>
                <option value="none">Sem achados</option>
                <option value="with-suspects">Apenas com suspeitos</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Mínimo de achados:</label>
            <input type="number" id="minSuspects" min="0" value="0" onchange="applyFilters()">
        </div>
        <div class="filter-group">
            <label>Buscar por nome:</label>
            <input type="text" id="searchName" placeholder="Digite o nome..." oninput="applyFilters()">
        </div>
        <div class="filter-group">
            <button onclick="resetFilters()">Limpar Filtros</button>
        </div>
        <div class="filter-group">
            <button onclick="expandAll()">Expandir Todos</button>
            <button onclick="collapseAll()">Colapsar Todos</button>
        </div>
        <div class="filter-results">
            <span id="visibleCount">{len(sorted_analyses)}</span> de {len(sorted_analyses)} imagens visíveis
        </div>
    </div>
    
    <div class="container">
"""
        
        for analysis in sorted_analyses:
            html += self._generate_image_section(analysis)
        
        html += """
    </div>
    
    <div class="modal" id="imageModal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <span class="modal-nav modal-prev" onclick="navigateModal(-1)">&#10094;</span>
        <img id="modalImage" src="" alt="Enlarged view">
        <span class="modal-nav modal-next" onclick="navigateModal(1)">&#10095;</span>
        <div class="modal-counter" id="modalCounter"></div>
    </div>
    
    <div class="footer">
        <p>Relatório gerado automaticamente pelo Forensic Image Processor</p>
    </div>
    
    <script>
        function toggleSection(element) {
            const section = element.closest('.processed-section');
            section.classList.toggle('expanded');
        }
        
        function toggleImageSection(header) {
            const section = header.closest('.image-section');
            section.classList.toggle('collapsed');
        }
        
        function expandAll() {
            document.querySelectorAll('.image-section').forEach(s => s.classList.remove('collapsed'));
        }
        
        function collapseAll() {
            document.querySelectorAll('.image-section').forEach(s => s.classList.add('collapsed'));
        }
        
        let currentImages = [];
        let currentIndex = 0;
        
        function openModal(src, imgElement) {
            // Find all images in the same section
            const section = imgElement ? imgElement.closest('.image-section') : null;
            if (section) {
                currentImages = Array.from(section.querySelectorAll('img[onclick]')).map(img => img.src);
                currentIndex = currentImages.indexOf(src);
                if (currentIndex === -1) currentIndex = 0;
            } else {
                currentImages = [src];
                currentIndex = 0;
            }
            
            showModalImage();
            document.getElementById('imageModal').classList.add('active');
        }
        
        function showModalImage() {
            document.getElementById('modalImage').src = currentImages[currentIndex];
            document.getElementById('modalCounter').textContent = 
                (currentIndex + 1) + ' / ' + currentImages.length;
        }
        
        function navigateModal(direction) {
            currentIndex += direction;
            if (currentIndex < 0) currentIndex = currentImages.length - 1;
            if (currentIndex >= currentImages.length) currentIndex = 0;
            showModalImage();
        }
        
        function closeModal() {
            document.getElementById('imageModal').classList.remove('active');
        }
        
        document.getElementById('imageModal').addEventListener('click', function(e) {
            if (e.target === this) closeModal();
        });
        
        document.addEventListener('keydown', function(e) {
            const modal = document.getElementById('imageModal');
            if (!modal.classList.contains('active')) return;
            
            if (e.key === 'Escape') closeModal();
            if (e.key === 'ArrowLeft') navigateModal(-1);
            if (e.key === 'ArrowRight') navigateModal(1);
        });
        
        function applyFilters() {
            const priorityFilter = document.getElementById('priorityFilter').value;
            const minSuspects = parseInt(document.getElementById('minSuspects').value) || 0;
            const searchName = document.getElementById('searchName').value.toLowerCase();
            
            const sections = document.querySelectorAll('.image-section');
            let visibleCount = 0;
            
            sections.forEach(section => {
                const priority = section.dataset.priority;
                const suspects = parseInt(section.dataset.suspects);
                const name = section.dataset.name;
                
                let visible = true;
                
                // Priority filter
                if (priorityFilter === 'with-suspects') {
                    visible = suspects > 0;
                } else if (priorityFilter !== 'all') {
                    visible = priority === priorityFilter;
                }
                
                // Min suspects filter
                if (visible && suspects < minSuspects) {
                    visible = false;
                }
                
                // Name search filter
                if (visible && searchName && !name.includes(searchName)) {
                    visible = false;
                }
                
                section.classList.toggle('hidden', !visible);
                if (visible) visibleCount++;
            });
            
            document.getElementById('visibleCount').textContent = visibleCount;
        }
        
        function resetFilters() {
            document.getElementById('priorityFilter').value = 'all';
            document.getElementById('minSuspects').value = '0';
            document.getElementById('searchName').value = '';
            applyFilters();
        }
    </script>
</body>
</html>
"""
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_image_section(self, analysis: ImageAnalysis) -> str:
        """Generate HTML section for a single image analysis."""
        priority = "none"
        if analysis.suspect_count >= 5:
            priority = "high"
        elif analysis.suspect_count >= 3:
            priority = "medium"
        elif analysis.suspect_count >= 1:
            priority = "low"
        
        collapsed_class = "" if analysis.has_suspects else "collapsed"
        html = f"""
        <div class="image-section {collapsed_class}" data-priority="{priority}" data-suspects="{analysis.suspect_count}" data-name="{analysis.raw_name.lower()}">
            <div class="section-header" onclick="toggleImageSection(this)">
                <div class="section-title">
                    <span class="collapse-icon">▼</span>
                    {analysis.raw_name} <span style="color: #888; font-size: 0.8em;">({analysis.suspect_count} achados)</span>
                </div>
                <div class="verdict {analysis.get_verdict_class()}">{analysis.get_verdict()}</div>
            </div>
            <div class="section-content">
                <div class="images-comparison">
                    <div class="raw-image-container">
                        <img src="{analysis.raw_path}" alt="Original" loading="lazy" onclick="openModal(this.src, this)">
                        <div class="label">Imagem Original</div>
                    </div>
"""
        
        if analysis.overlay_path:
            html += f"""
                    <div class="overlay-image-container">
                        <img src="{analysis.overlay_path}" alt="Overlay" loading="lazy" onclick="openModal(this.src, this)">
                        <div class="label">Overlay com Achados</div>
                    </div>
"""
        
        html += """
                </div>
"""
        
        if analysis.has_suspects:
            html += """
                <div class="suspect-section">
                    <h3>Achados Suspeitos</h3>
                    <div class="suspect-grid">
"""
            for suspect_path in analysis.suspect_images:
                technique = self._extract_technique(suspect_path)
                desc = TECHNIQUE_DESCRIPTIONS.get(technique, "Técnica desconhecida")
                ai_analysis = analysis.ai_analyses.get(suspect_path, "")
                ai_html = ""
                if ai_analysis:
                    ai_escaped = ai_analysis.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                    ai_html = f"""
                                <div class="ai-analysis">
                                    <div class="ai-analysis-label">Análise IA</div>
                                    {ai_escaped}
                                </div>"""
                html += f"""
                        <div class="suspect-item">
                            <img src="{suspect_path}" alt="{technique}" loading="lazy" onclick="openModal(this.src, this)">
                            <div class="info">
                                <div class="technique">{technique}</div>
                                <div class="description">{desc}</div>{ai_html}
                            </div>
                        </div>
"""
            html += """
                    </div>
                </div>
"""
            
            if analysis.individual_overlays:
                html += """
                <div class="individual-overlays">
                    <h3>Overlays Individuais por Técnica</h3>
                    <div class="individual-overlays-grid">
"""
                for overlay_info in analysis.individual_overlays:
                    html += f"""
                        <div class="individual-overlay-item">
                            <img src="{overlay_info['path']}" alt="{overlay_info['technique']}" loading="lazy" onclick="openModal(this.src, this)">
                            <div class="info">
                                <div class="technique">{overlay_info['technique'].upper()}</div>
                            </div>
                        </div>
"""
                html += """
                    </div>
                </div>
"""
            
            if analysis.zoom_crops:
                html += """
                <div class="zoom-section">
                    <h3>Zoom nas Regiões Detectadas</h3>
                    <div class="zoom-grid">
"""
                for zoom_info in analysis.zoom_crops:
                    html += f"""
                        <div class="zoom-item">
                            <img src="{zoom_info['path']}" alt="Zoom {zoom_info['index']}" loading="lazy" onclick="openModal(this.src, this)">
                            <div class="info">Região #{zoom_info['index']}</div>
                        </div>
"""
                html += """
                    </div>
                </div>
"""
        else:
            html += """
                <div class="no-evidence">
                    <div class="icon">✓</div>
                    <p>Nenhuma evidência suspeita detectada nas análises</p>
                </div>
"""
        
        if analysis.processed_images:
            html += """
                <div class="processed-section">
                    <h3 onclick="toggleSection(this)">Todas as Análises Processadas</h3>
                    <div class="processed-grid">
"""
            suspect_filenames = {Path(s).name for s in analysis.suspect_images}
            
            categorized = defaultdict(list)
            for proc_path in analysis.processed_images:
                technique = self._extract_technique(proc_path)
                for cat_name, techniques in TECHNIQUE_CATEGORIES.items():
                    if technique in techniques:
                        categorized[cat_name].append((proc_path, technique))
                        break
            
            for cat_name, items in categorized.items():
                for proc_path, technique in items:
                    is_suspect = Path(proc_path).name in suspect_filenames
                    suspect_class = "is-suspect" if is_suspect else ""
                    desc = TECHNIQUE_DESCRIPTIONS.get(technique, technique)
                    html += f"""
                        <div class="processed-item {suspect_class}">
                            <img src="{proc_path}" alt="{technique}" loading="lazy" onclick="openModal(this.src, this)">
                            <div class="info">{desc}</div>
                        </div>
"""
            
            html += """
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        return html


def _select_folder(images_dir: Path, prefix: str, folder_type: str) -> Optional[str]:
    """List and select a folder by prefix."""
    folders = sorted([
        folder for folder in images_dir.iterdir()
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


def main():
    """Main entry point."""
    base_dir = r"D:\Development\forense"
    images_dir = Path(base_dir) / "images"
    
    print("\n=== Forensic Report Generator ===")
    print("\nOpções:")
    print("  1. Gerar relatório")
    print("  2. Exportar lista de imagens para análise de IA")
    print("  3. Sair")
    
    option = input("\nEscolha uma opção: ").strip()
    
    if option == '3' or option.lower() == 'q':
        print("Saindo.")
        return
    
    raw_folder = _select_folder(images_dir, "raw", "raw")
    if raw_folder is None:
        print("Cancelado.")
        return
    
    if option == '2':
        generator = ForensicReportGenerator(base_dir, raw_folder)
        export_file = generator.export_images_for_analysis()
        print(f"\n✓ Lista de imagens exportada para: {export_file}")
        print("\nPróximos passos:")
        print("  1. Peça ao assistente IA para analisar as imagens")
        print(f"  2. O assistente criará: ai_analysis_{raw_folder}.json")
        print("  3. Execute novamente e escolha 'Gerar relatório'")
        return
    
    processed_folder = _select_folder(images_dir, "processed", "processed")
    if processed_folder is None:
        print("Cancelado.")
        return
    
    suspect_folder = _select_folder(images_dir, "suspect", "suspect")
    if suspect_folder is None:
        print("Cancelado.")
        return
    
    ai_analysis_file = Path(base_dir) / f"ai_analysis_{raw_folder}.json"
    has_ai_json = ai_analysis_file.exists()
    
    enable_ai = False
    if has_ai_json:
        print(f"\n✓ Arquivo de análise IA encontrado: ai_analysis_{raw_folder}.json")
    else:
        print(f"\n✗ Nenhuma análise de IA encontrada (ai_analysis_{raw_folder}.json)")
        if ANTHROPIC_AVAILABLE:
            ai_choice = input("Habilitar análise de IA via API? (s/n): ").strip().lower()
            if ai_choice == 's':
                if os.environ.get("ANTHROPIC_API_KEY"):
                    enable_ai = True
                    print("  ✓ Análise de IA via API habilitada")
                else:
                    print("  ✗ ANTHROPIC_API_KEY não encontrada.")
    
    print(f"\n--- Configuração ---")
    print(f"  Raw: {raw_folder}")
    print(f"  Processed: {processed_folder}")
    print(f"  Suspect: {suspect_folder}")
    print(f"  Análise IA (JSON): {'Sim' if has_ai_json else 'Não'}")
    print(f"  Análise IA (API): {'Sim' if enable_ai else 'Não'}")
    print(f"  Output: index_{raw_folder}.html")
    
    confirm = input("\nGerar relatório? (s/n): ").strip().lower()
    if confirm != 's':
        print("Cancelado.")
        return
    
    print()
    
    try:
        generator = ForensicReportGenerator(base_dir, raw_folder, processed_folder, suspect_folder, enable_ai)
        generator.generate_report()
        logger.debug("Report generation complete")
        print(f"\nRelatório gerado: {generator.output_file}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
