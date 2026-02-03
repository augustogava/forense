# Forensic Image Analysis System

Sistema completo de análise forense de imagens para detecção de manipulações, anomalias e evidências suspeitas.

## Visão Geral

O sistema consiste em 3 scripts principais que trabalham em pipeline:

```
┌─────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────┐
│ forensic_processor  │ -> │ forensic_suspect_detector│ -> │  forensic_report    │
│  (gera 34 técnicas) │    │   (detecta anomalias)    │    │  (gera relatório)   │
└─────────────────────┘    └──────────────────────────┘    └─────────────────────┘
```

## Estrutura de Pastas

```
forense/
├── images/
│   ├── raw/              # Imagens originais para análise
│   ├── processed/        # Imagens processadas (34 técnicas por raw)
│   ├── suspect/          # Suspeitos detectados (versão original)
│   ├── suspect_fast/     # Suspeitos detectados (versão paralela)
│   ├── suspect_fast_v2/  # Suspeitos detectados (versão com thresholds ajustados)
│   └── overlays/         # Overlays gerados para o relatório
├── forensic_processor.py
├── forensic_suspect_detector.py
├── forensic_suspect_detector_fast.py
├── forensic_report.py
└── index.html            # Relatório HTML gerado
```

---

## Scripts

### 1. forensic_processor.py

**Função:** Processa imagens raw aplicando ~58 técnicas de análise forense (multi-thread).

**Técnicas Aplicadas:**

| Categoria | Técnicas |
|-----------|----------|
| **Contraste** | CLAHE (1.0, 2.0, 4.0, 8.0), Histogram Eq, Gamma (0.3-3.0) |
| **Detecção de Bordas** | Canny (sensitive/balanced/strict), Sobel, Laplacian, DoG, Morph Gradient |
| **Análise de Canais** | RGB, HSV, LAB, YCrCb |
| **Multiespectral** | 480nm, 620nm, 850nm, Hemoglobina, ALS 415nm/450nm |
| **Realce de Detalhes** | Unsharp, Highpass, Emboss |
| **Filtros Especiais** | Negative, False Color, Skin Enhanced, Bilateral, LBP |
| **Retinex** | SSR, MSR (visibilidade através de tecidos) |
| **Frequência** | FFT Bandpass, FFT Highpass (remove padrões de tecido) |
| **Polarização** | Cross-polarized (remove reflexos) |
| **Análise Forense** | DCT Blocks, Wavelet |

**Uso:**
```bash
python forensic_processor.py
```

**Características:**
- Processamento paralelo com 8 threads (configurável)
- ~58 imagens por cada raw image em `images/processed/`

---

### 2. forensic_suspect_detector.py

**Função:** Analisa imagens processadas e detecta anomalias estatísticas.

**Técnicas de Detecção:**

| Técnica | Descrição |
|---------|-----------|
| **ELA (Error Level Analysis)** | Detecta edições JPEG por inconsistências de compressão |
| **Noise Analysis** | Detecta regiões com ruído inconsistente |
| **Thresholding Adaptativo** | Usa IQR (Interquartile Range) por categoria |

**Categorias e Thresholds:**

| Categoria | IQR Multiplier | Min Threshold |
|-----------|----------------|---------------|
| Edge | 1.8 | 60.0 |
| Enhancement | 1.3 | 35.0 |
| Channel | 1.3 | 30.0 |
| Specialized | 1.3 | 25.0 |
| Forensic | 1.0 | 20.0 |

**Fórmula do Threshold:**
```
threshold = max(Q3 + (IQR_mult * IQR), median * 1.2, min_threshold)
```

**Uso:**
```bash
python forensic_suspect_detector.py
```

**Saída:** Imagens suspeitas copiadas para `images/suspect/`

---

### 3. forensic_suspect_detector_fast.py

**Função:** Versão paralela (multi-thread) do detector, ~3-5x mais rápido.

**Diferenças da versão original:**
- Usa `ThreadPoolExecutor` com 8 workers
- Processa raw images em paralelo (Fase 1)
- Processa imagens processadas em paralelo (Fase 2)
- Thread-safe com locks para dados compartilhados
- Inclui técnica adicional: **JPEG Ghost Analysis**

**JPEG Ghost Analysis:**
- Recomprime a imagem em múltiplas qualidades (60, 70, 80, 90, 95)
- Analisa variância das diferenças
- Detecta regiões com histórico de compressão diferente

**Uso:**
```bash
python forensic_suspect_detector_fast.py
# ou com workers personalizados:
python forensic_suspect_detector_fast.py --workers 4
```

**Saída:** Imagens suspeitas em `images/suspect_fast_v2/`

---

### 4. forensic_report.py

**Função:** Gera relatório HTML interativo com todas as análises.

**Funcionalidades:**
- **Ordenação:** Por prioridade (mais suspeitos primeiro)
- **Filtros:** Por prioridade, mínimo de achados, busca por nome
- **Overlays:** Combina original com achados suspeitos
- **Zoom:** Recortes das regiões detectadas
- **Modal:** Clique em qualquer imagem para ampliar
- **Seção Expansível:** "Todas as Análises Processadas"
- **Análise de IA:** Opinião da IA para cada achado suspeito (opcional)

**Níveis de Prioridade:**

| Prioridade | Achados | Cor |
|------------|---------|-----|
| Alta | 5+ | Vermelho |
| Média | 3-4 | Laranja |
| Baixa | 1-2 | Amarelo |
| Nenhum | 0 | Verde |

**Uso:**
```bash
python forensic_report.py
```

O script pergunta qual pasta de suspects usar:
```
Pastas de suspects disponíveis:
  1. suspect
  2. suspect_fast
  3. suspect_fast_v2

Pasta padrão: suspect_fast_v2
Digite o nome da pasta (Enter para padrão):
```

**Saída:** `index.html` (abrir no navegador)

---

## Pipeline Completo

### Execução Rápida (Recomendada)
```bash
# 1. Processar imagens (se ainda não feito)
python forensic_processor.py

# 2. Detectar suspeitos (versão rápida)
python forensic_suspect_detector_fast.py

# 3. Gerar relatório
python forensic_report.py
# (pressione Enter para usar pasta padrão)
```

### Execução Completa
```bash
# Processar + Detectar + Relatório em sequência
python forensic_processor.py && python forensic_suspect_detector_fast.py && python forensic_report.py
```

---

## Métricas de Detecção

### Scores Calculados

| Métrica | Descrição |
|---------|-----------|
| `diff_ratio` | Proporção de pixels diferentes (Otsu threshold) |
| `diff_mean` | Média da diferença absoluta |
| `diff_std` | Desvio padrão da diferença |
| `ssim` | Structural Similarity Index (1 - SSIM usado) |
| `edge_density` | Densidade de bordas detectadas |

### Fórmula do Score
```python
score = (
    weights["diff_ratio"] * (diff_ratio * 100) +
    weights["diff_mean"] * (diff_mean / 2) +
    weights["ssim"] * ((1 - ssim) * 100) +
    weights["edge_density"] * (edge_density * 100)
)
```

### Scores Forenses

| Técnica | Fórmula |
|---------|---------|
| ELA | `(ela_mean/2) + (ela_std/1.5) + (ela_max_region/3)` |
| Noise | `(noise_inconsistency * 50) + (noise_mean/2)` |
| JPEG Ghost | `(ghost_inconsistency * 40) + (ghost_mean * 2) + (ghost_std * 0.5)` |

---

## Requisitos

```bash
pip install -r requirements.txt
```

Ou manualmente:
```bash
pip install opencv-python numpy pillow scikit-image anthropic
# Opcional para HEIC:
pip install pillow-heif
```

### Análise de IA (Opcional)

Existem duas formas de adicionar análise de IA aos achados suspeitos:

**Opção 1: Via Cursor/Assistente IA (Recomendado - sem API key)**
```bash
# 1. Exportar lista de imagens para análise
python forensic_report.py
# Escolha opção 2: "Exportar lista de imagens para análise de IA"

# 2. Peça ao assistente IA no Cursor:
#    "Analise as imagens suspeitas de raw_X"
#    O assistente criará: ai_analysis_raw_X.json

# 3. Gerar relatório com as análises
python forensic_report.py
# Escolha opção 1: "Gerar relatório"
# O relatório incluirá automaticamente as análises do JSON
```

**Opção 2: Via API Anthropic (requer API key)**
```bash
# Linux/Mac
export ANTHROPIC_API_KEY="sua-chave-aqui"

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sua-chave-aqui"

# Windows (CMD)
set ANTHROPIC_API_KEY=sua-chave-aqui

pip install anthropic
```

---

## Interpretação dos Resultados

### No Relatório HTML

1. **Imagens com borda vermelha** = Detectadas como suspeitas
2. **Seção "Achados Suspeitos"** = Técnicas que indicaram anomalia
3. **Overlay** = Visualização das regiões anômalas
4. **Clique em "► Todas as Análises"** = Ver todas as 34 técnicas

### Nos Logs

```
[SUSPECT] IMG_8591/enhancement: 2 suspects detected (threshold=45.23)
[SUSPECT] Marking: IMG_8591_negative.jpg | score=72.96 | threshold=45.23
```

- `score > threshold` = Imagem marcada como suspeita
- Quanto maior o score em relação ao threshold, mais anômala a imagem

---

## Dicas de Uso

1. **Muitos falsos positivos?** Aumente os valores em `CATEGORY_THRESHOLDS`
2. **Poucos suspeitos?** Diminua os valores ou o `iqr_mult`
3. **Processamento lento?** Use `forensic_suspect_detector_fast.py`
4. **Imagens HEIC?** Instale `pillow-heif`

---

## Licença

Uso interno para análise forense.
