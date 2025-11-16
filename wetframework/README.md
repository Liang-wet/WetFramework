# WetFramework: A Deep Learning Framework for Coastal Wetland Boundary Extraction and Inundation Frequency Estimation

This repository contains the official implementation of **WetFramework**, a hybrid Transformer–Mamba deep learning framework designed for:

- Coastal wetland boundary extraction  
- Long-term inundation frequency estimation  
- Hydrological process interpretation through Fourier-based analysis  

WetFramework integrates spatial deep learning with temporal hydrological modelling to support coastal wetland monitoring, inundation dynamics analysis, and science-based wetland management decisions.

---

## 📦 Repository Structure

```
wetframework/
│
├── Encoder/                                    # Encoder components (TMHM & TDAM)
│   ├── encoder.py                              # WFEncoder (4-stage Transformer–Mamba hybrid)
│   ├── patch_embed.py                          # Overlap Patch Embedding
│   ├── transformer_block.py                    # Transformer block (MHSA + MLP)
│   ├── mamba_block.py                          # Mamba block (SSM-based)
│   ├── tdam.py                                 # Token-Driven Attention Module
│   └── __init__.py
│
├── Decoder/                                    # Decoder components (Wavelet Enhancement)
│   ├── decoder.py                              # WFDecoder (feature fusion + reconstruction)
│   ├── werm.py                                 # Wavelet-Enhanced Reconstruction Module (WERM)
│   └── __init__.py
│
├── model.py                                    # Full WetFramework assembly
├── fbiem.m                                     # MATLAB implementation for Fourier-Based Inundation Estimation
├── __init__.py
│
└── README.md                                   # Documentation
```

---

## 🔍 Correspondence Between Code and Paper Figures

Each core module includes explicit annotations mapping back to the corresponding figures in the paper:

| File | Paper Figure | Description |
|------|--------------|-------------|
| `patch_embed.py` | Fig. 1 (Overlap Patch Embedding) | Patch embedding for input images |
| `transformer_block.py` | Fig. 2(a) | Transformer block (MHSA + MLP) |
| `mamba_block.py` | Fig. 2(b) | Mamba block (Selective SSM) |
| `tdam.py` | Fig. 2(c) | Token-Driven Attention Module |
| `encoder.py` | Fig. 1 (Stage 1–4) | Full Transformer–Mamba hybrid encoder |
| `werm.py` | Fig. 4(a) | Wavelet-Enhanced Reconstruction Module |
| `decoder.py` | Fig. 1 & Fig. 4(b) | Decoder & multi-scale fusion |
| `fbiem.m` | Fig. 1 (Bottom block) | Fourier-based inundation fitting |


## ⚙️ Runtime Environment & Key Package Versions

Only packages required by WetFramework are listed.

### **Python & PyTorch**
| Package | Version |
|--------|---------|
| Python | 3.10–3.11 |
| torch | 2.2.0 |
| torchvision | 0.17.0 |
| torchaudio | 2.2.0 |

### **State-Space Model (SSM) / Mamba**
| Package | Version |
|--------|---------|
| mamba-ssm | 2.2.2 |
| causal-conv1d | 1.4.0 |

### **Wavelet Transform (WERM)**
| Package | Version |
|--------|---------|
| pytorch-wavelets | 1.3.0 |
| PyWavelets | 1.3.0 |

### **Raster & Geospatial IO**
| Package | Version |
|--------|---------|
| rasterio | 1.3.11 |
| GDAL | 3.4.3 |

### **Other Required Dependencies**
| Package | Version |
|--------|---------|
| numpy | 1.22.3 |
| scipy | 1.6.2 |
| einops | 0.8.1 |
| tqdm | 4.66.5 |

---

## 🚀 Model Forward Sanity Test

`model.py` provides a simple forward test:

```bash
python model.py
```

Expected output:

```
Input shape : torch.Size([4, 3, 512, 512])
Output shape: torch.Size([4, 1, 512, 512])
Model forward test SUCCESS!
```

---

## 🧬 MATLAB Version of FBIEM

`fbiem.m` includes:

- Multi-temporal MNDWI loading  
- Savitzky–Golay smoothing  
- Two-term Fourier fitting  
- Output of per-pixel coefficients (a0, a1, a2, b1, b2)  

This script is provided as a reproducible hydrological supplement.


Feel free to open issues or contribute improvements.
