# AxisNet: Multimodal Microbiome-Brain Network Fusion for Neurological Disease Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.4+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-Latest-green.svg" alt="PyTorch Geometric">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Architecture](#architecture)
- [Multimodal Fusion Mechanism](#multimodal-fusion-mechanism)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Model (GCN+Transformer)](#model-axisnetgcntransformer-gcntransformer)
- [Configuration Options](#configuration-options)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

**AxisNet** is a multimodal deep learning framework that combines functional brain network data (fMRI) with gut microbiome profiles to predict neurological disorders such as Autism Spectrum Disorder (ASD). The framework extends the Edge-Variational Graph Convolutional Network (EV-GCN) architecture with:

- **Microbiome modality integration** via a dedicated encoder
- **Cross-modal contrastive learning** for modality alignment
- **Graph consistency regularization** leveraging microbiome similarities
- **GCN+Transformer hybrid** (ChebConv layers + Transformer encoder)

### Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal Fusion** | Integrates fMRI brain connectivity with gut microbiome data |
| **Edge-Variational Learning** | Learns adaptive edge weights via Pairwise Affinity Estimator (PAE) |
| **Contrastive Learning** | Label-aware cross-modal alignment using NT-Xent loss |
| **GCN+Transformer** | ChebConv layers with JK + Transformer encoder for graph and modality fusion |
| **Intelligent Data Pairing** | Smart pseudo-pairing when direct sample correspondence is unavailable |
| **Backward Compatible** | Fully supports single-modality (fMRI-only) training |

---

## Scientific Background

### The Microbiome-Gut-Brain Axis

The **microbiome-gut-brain axis** represents the bidirectional communication network linking the gut microbiota with the central nervous system. Research has shown that:

1. **Gut dysbiosis** is associated with various neurological and psychiatric conditions including ASD, Parkinson's disease, and depression
2. **Microbial metabolites** (e.g., short-chain fatty acids, neurotransmitters) can influence brain function
3. **Inflammatory pathways** modulated by gut bacteria affect neural development and function

### Why Combine fMRI and Microbiome Data?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MICROBIOME-GUT-BRAIN AXIS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    Vagus Nerve    ┌─────────────┐                     │
│  │   GUT       │◄──────────────────►│   BRAIN     │                     │
│  │ MICROBIOME  │    Metabolites     │  FUNCTION   │                     │
│  │             │    Immune Signals  │             │                     │
│  │ 2500+ taxa  │    Hormones        │ fMRI-based  │                     │
│  │             │                    │ connectivity│                     │
│  └─────────────┘                    └─────────────┘                     │
│        ▲                                   ▲                            │
│        │                                   │                            │
│        └───────────────┬───────────────────┘                            │
│                        │                                                │
│                        ▼                                                │
│              ┌─────────────────┐                                        │
│              │    AxisNet      │                                        │
│              │   Integration   │                                        │
│              └─────────────────┘                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

By jointly modeling both modalities, AxisNet can:
- Capture complementary biological information
- Identify cross-modal biomarkers
- Improve classification robustness and accuracy
- Discover microbiome-brain interaction patterns

---

## Architecture

### Overall System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                    AxisNet Framework                                 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           INPUT LAYER                                          │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                      │  │
│  │  │   fMRI       │    │  Phenotypic  │    │  Microbiome  │                      │  │
│  │  │ Connectivity │    │    Data      │    │   Abundance  │                      │  │
│  │  │   Matrix     │    │ (age, sex,   │    │   Profile    │                      │  │
│  │  │  [N × ROIs²] │    │   site)      │    │  [N × 2503]  │                      │  │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                      │  │
│  │         │                   │                   │                              │  │
│  └─────────┼───────────────────┼───────────────────┼──────────────────────────────┘  │
│            │                   │                   │                                 │
│            ▼                   ▼                   ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                        PREPROCESSING LAYER                                      │ │
│  │                                                                                 │ │
│  │  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐         │ │
│  │  │  Feature Select.  │   │  Graph Construct. │   │   CLR Transform   │         │ │
│  │  │  (RFE, top-2000)  │   │  (Affinity-based) │   │   + PCA (64-dim)  │         │ │
│  │  └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘         │ │
│  │            │                       │                       │                   │ │
│  └────────────┼───────────────────────┼───────────────────────┼───────────────────┘ │
│               │                       │                       │                     │
│               ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           MODEL LAYER                                           │ │
│  │                                                                                 │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐     │ │
│  │  │                    Edge-Variational GCN Core                          │     │ │
│  │  │                                                                       │     │ │
│  │  │    ┌─────────┐      ┌─────────┐      ┌─────────┐                     │     │ │
│  │  │    │  Edge   │─────►│ Edge    │─────►│ ChebConv│──┐                  │     │ │
│  │  │    │ Encoder │      │ Weights │      │ Layer 1 │  │                  │     │ │
│  │  │    │  (PAE)  │      │         │      │         │  │ Jumping          │     │ │
│  │  │    └─────────┘      └─────────┘      └─────────┘  │ Knowledge        │     │ │
│  │  │                                            │      │ Connection       │     │ │
│  │  │                                            ▼      │                  │     │ │
│  │  │                                      ┌─────────┐  │                  │     │ │
│  │  │                                      │ ChebConv│──┼──►┌─────────┐   │     │ │
│  │  │                                      │ Layer 2 │  │   │Concat JK│   │     │ │
│  │  │                                      └─────────┘  │   │ Features│   │     │ │
│  │  │                                            │      │   └────┬────┘   │     │ │
│  │  │                                            ▼      │        │        │     │ │
│  │  │                                      ┌─────────┐  │        │        │     │ │
│  │  │                                      │ ChebConv├──┘        │        │     │ │
│  │  │                                      │ Layer N │           │        │     │ │
│  │  │                                      └─────────┘           │        │     │ │
│  │  │                                                            │        │     │ │
│  │  └────────────────────────────────────────────────────────────┼────────┘     │ │
│  │                                                               │              │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    Multimodal Fusion Module                           │   │ │
│  │  │                                                                       │   │ │
│  │  │   ┌─────────────────┐          ┌──────────────────┐                  │   │ │
│  │  │   │   Microbiome    │          │  Brain Network   │◄─────────────────┼───┤ │
│  │  │   │    Encoder      │          │    Projector     │   JK Features    │   │ │
│  │  │   │ (MLP: 2503→128) │          │  (MLP: JK→128)   │                  │   │ │
│  │  │   └────────┬────────┘          └────────┬─────────┘                  │   │ │
│  │  │            │                            │                            │   │ │
│  │  │            ▼                            ▼                            │   │ │
│  │  │   ┌────────────────────────────────────────────────┐                 │   │ │
│  │  │   │         Contrastive Learning Loss              │                 │   │ │
│  │  │   │    (NT-Xent with label-aware sampling)         │                 │   │ │
│  │  │   └────────────────────────────────────────────────┘                 │   │ │
│  │  │                                                                       │   │ │
│  │  └───────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                              │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                        │
│                                          ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │                           OUTPUT LAYER                                       │ │
│  │                                                                              │ │
│  │    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │ │
│  │    │   Classifier     │    │   Edge Weights   │    │  Cross-Modal     │     │ │
│  │    │   (JK → 256 → 2) │    │   (for analysis) │    │  Embeddings      │     │ │
│  │    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘     │ │
│  │             │                       │                       │               │ │
│  │             ▼                       ▼                       ▼               │ │
│  │    ┌──────────────┐        ┌──────────────┐        ┌──────────────┐        │ │
│  │    │  ASD / HC    │        │ Brain Graph  │        │  Biomarker   │        │ │
│  │    │  Prediction  │        │  Structure   │        │  Discovery   │        │ │
│  │    └──────────────┘        └──────────────┘        └──────────────┘        │ │
│  │                                                                              │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Edge Encoder (Pairwise Affinity Estimator)

The `EdgeEncoder` learns to predict edge weights between subjects based on their phenotypic similarity:

```python
class EdgeEncoder(nn.Module):
    """
    Pairwise Affinity Estimator for edge weight prediction.
    
    Input: Concatenated phenotypic features of two subjects [x_i || x_j]
    Output: Edge weight in [0, 1] representing subject similarity
    """
    def forward(self, x):
        x1 = x[:, 0:self.input_dim]  # Subject i features
        x2 = x[:, self.input_dim:]   # Subject j features
        h1 = self.parser(x1)         # Embed subject i
        h2 = self.parser(x2)         # Embed subject j
        return (self.cos(h1, h2) + 1) * 0.5  # Normalized similarity
```

**Key innovations:**
- **Learnable affinity**: Unlike fixed similarity metrics, EdgeEncoder learns task-specific relationships
- **Variational dropout**: Edges are randomly dropped during training for regularization
- **Phenotype-aware**: Uses clinical features (age, sex, site) for graph construction

#### 2. Chebyshev Graph Convolution

AxisNet uses spectral graph convolutions with Chebyshev polynomial approximation:

```
h^(l+1) = σ(∑_{k=0}^{K-1} T_k(L̃) · h^(l) · W_k)
```

Where:
- `T_k(L̃)` is the k-th Chebyshev polynomial of the normalized Laplacian
- `K=3` provides a 3-hop neighborhood aggregation
- Symmetric normalization ensures stable training

#### 3. Jumping Knowledge (JK) Connections

To preserve multi-scale information, features from all GCN layers are concatenated:

```
JK = [h^(1) || h^(2) || ... || h^(L)]
```

This allows the classifier to leverage both local (early layers) and global (later layers) graph structure.

#### 4. Microbiome Encoder

A lightweight MLP processes microbiome abundance profiles:

```python
class MicrobiomeEncoder(nn.Module):
    def __init__(self, input_dim=2503, embed_dim=128):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
```

**Preprocessing applied:**
1. **CLR Transform**: Centered log-ratio transformation for compositional data
2. **PCA Reduction**: Dimensionality reduction to 64 components
3. **Normalization**: Feature standardization

#### 5. Contrastive Learning Module

Cross-modal alignment using NT-Xent loss:

```python
def contrastive_loss(microbiome_embed, brain_embed, labels, temperature=0.5):
    """
    Label-aware contrastive loss for modality alignment.
    
    Positive pairs: Same subject's microbiome and brain embeddings
    Negative pairs: Different subjects' cross-modal embeddings
    """
    # Normalize embeddings
    microbiome_embed = F.normalize(microbiome_embed, dim=1)
    brain_embed = F.normalize(brain_embed, dim=1)
    
    # Compute NT-Xent loss
    for each sample i:
        pos_sim = microbiome[i] · brain[i] / τ
        neg_sims = [microbiome[i] · brain[j], brain[i] · microbiome[j]] for j ≠ i
        loss += -log(exp(pos_sim) / (exp(pos_sim) + Σexp(neg_sims)))
```

---

## Multimodal Fusion Mechanism

### The Challenge: Unpaired Data

In many real-world scenarios, fMRI and microbiome data come from different cohorts without direct sample correspondence. AxisNet addresses this through **intelligent pseudo-pairing**.

### Pseudo-Pairing Strategy

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        PSEUDO-PAIRING PIPELINE                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   fMRI Subject Pool              Microbiome Subject Pool                       │
│   ┌─────────────────┐            ┌─────────────────┐                          │
│   │ Subject 1       │            │ Microbiome A    │                          │
│   │ - Age: 12       │            │ - Age: 11       │                          │
│   │ - Sex: Male     │            │ - Sex: Male     │                          │
│   │ - Site: UCLA    │            │ - Cohort: AGP   │                          │
│   └────────┬────────┘            └─────────────────┘                          │
│            │                                                                   │
│            │  ┌─────────────────────────────────────┐                         │
│            └──►      MATCHING ALGORITHM             │                         │
│               │                                     │                         │
│               │  1. Filter by sex (strict match)    │                         │
│               │  2. Sort by age difference          │                         │
│               │  3. Select top-K closest matches    │                         │
│               │  4. Aggregate (mean) features       │                         │
│               │                                     │                         │
│               └──────────────┬──────────────────────┘                         │
│                              │                                                │
│                              ▼                                                │
│               ┌─────────────────────────────────────┐                         │
│               │  Pseudo-paired Microbiome Feature   │                         │
│               │  (Aggregated from top-K matches)    │                         │
│               └─────────────────────────────────────┘                         │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Algorithm Details

```python
def create_pseudo_pairing(fmri_subjects, microbiome_pool, top_k=5):
    for each fMRI subject:
        # Step 1: Sex-based filtering
        candidates = microbiome_pool[sex == fmri_subject.sex]
        
        # Step 2: Age-based sorting
        age_diff = |candidates.age - fmri_subject.age|
        sorted_candidates = candidates[argsort(age_diff)]
        
        # Step 3: Select top-K
        top_k_matches = sorted_candidates[:top_k]
        
        # Step 4: Feature aggregation
        pseudo_microbiome = mean(top_k_matches.features)
        
        yield (fmri_subject, pseudo_microbiome)
```

### Fusion Loss Function

The total training loss combines multiple objectives:

```
L_total = L_classification + λ₁·L_contrastive + λ₂·L_consistency
```

Where:
- **L_classification**: Cross-entropy loss for ASD/HC prediction
- **L_contrastive**: NT-Xent loss for cross-modal alignment
- **L_consistency**: MSE loss encouraging edge weights to reflect microbiome similarity

```python
# Graph consistency regularization
def graph_consistency_loss(edge_index, edge_weight, microbiome_features):
    micro_norm = F.normalize(microbiome_features, p=2, dim=1)
    src, dst = edge_index[0], edge_index[1]
    
    # Target: normalized microbiome similarity
    target = (micro_norm[src] * micro_norm[dst]).sum(dim=1)
    target = (target + 1.0) * 0.5  # Scale to [0, 1]
    
    return F.mse_loss(edge_weight, target)
```

---

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/AxisNet.git
cd AxisNet

# Create virtual environment (optional but recommended)
conda create -n axisnet python=3.8
conda activate axisnet

# Install all dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "
import torch
import torch_geometric
print(f'PyTorch: {torch.__version__}')
print(f'PyG: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

## Data Preparation

### ABIDE Dataset (fMRI)

#### Automatic Download

Run the data fetching script in the `data` folder:

```bash
cd data
python fetch_data.py
```

#### Manual Download

1. Download from the [ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/) repository
2. Place files in `data/ABIDE_pcp/cpac/filt_noglobal/`
3. Download phenotypic file: `Phenotypic_V1_0b_preprocessed1.csv`

#### Expected Structure

```
data/
├── ABIDE_pcp/
│   ├── cpac/
│   │   └── filt_noglobal/
│   │       ├── 50002/
│   │       │   ├── 50002_rois_ho.1D         # Time series
│   │       │   └── 50002_ho_correlation.mat # Connectivity matrix
│   │       ├── 50003/
│   │       └── ...
│   └── Phenotypic_V1_0b_preprocessed1.csv
└── subject_IDs.txt
```

### Microbiome Data

#### Supported Formats

1. **CSV Format**:
```csv
ID,Bacteroides,Prevotella,Faecalibacterium,...
sub-001,0.15,0.08,0.12,...
sub-002,0.12,0.09,0.11,...
```

2. **BIOM Format** (from QIIME2):
```
feature-table.biom
```

3. **HDF5 Format**:
```python
# Structure: {sample_id: abundance_vector}
with h5py.File('microbiome.hdf5', 'r') as f:
    sample_ids = list(f.keys())
    features = np.stack([f[k][:] for k in sample_ids])
```

#### Metadata File (Optional)

```csv
ID,AGE_AT_SCAN,SEX,DX_GROUP
sub-001,12.5,1,1
sub-002,10.2,2,2
```

### Using Simulated Data

If you don't have real microbiome data, AxisNet can generate realistic simulated data:

```bash
python -m scripts.train_eval --train=1 --use_multimodal --model_type=gcn_transformer
# Simulated microbiome data will be automatically generated
```

---

## Usage

All commands should be run from the **AxisNet directory** .

### Basic Training (Single Modality)

Train GCN+Transformer on fMRI data only (no microbiome):

```bash
python scripts.train_eval --mode=train --model_type=gcn_transformer
```

### Multimodal Training (GCN+Transformer)

Train AxisNetGcnTransformer (GCN + Transformer encoder) with microbiome integration:

```bash
# With simulated microbiome data
python scripts.train_eval --mode=train --use_multimodal --model_type=gcn_transformer

# With real microbiome data (CSV)
python scripts.train_eval --mode=train --use_multimodal --model_type=gcn_transformer

# With BIOM format (bundled sample data)
python scripts.train_eval --mode=train --use_multimodal --model_type=gcn_transformer
```

### Evaluation

```bash
python scripts.train_eval --mode=eval --model_type=gcn_transformer
```

### Batch Experiments

Run GCN+Transformer ablation (unimodal vs multimodal, phenotype variants):

```bash
python scripts.run_experiments \
    --microbiome_path=data/feature-table.biom \
    --model_type=gcn_transformer \
    --out_csv=result0_gcn_transformer.csv
```

The batch runner tests:
- Unimodal vs multimodal
- 4 phenotype variants: full, drop_age, drop_sex, drop_age_sex

### ABIDE I (LOSO) and ABIDE II (Cross-Validation)

**ABIDE I: LOSO (Leave-One-Site-Out)** — one site as test each fold:
```bash
python scripts/run_abide1_loso.py
```

Multimodal LOSO:
```bash
python scripts/run_abide1_loso.py --use_multimodal --microbiome_path data/feature-table.biom
```

**ABIDE II: cross-validation** — configure paths and run stratified K-fold (default 10 folds):
```bash
python scripts/run_abide2_cv.py \
  --data_folder /path/to/ABIDE_II/cpac/filt_noglobal \
  --phenotype_path /path/to/ABIDE_II_phenotype.csv
```

Phenotype CSV should include: `SUB_ID`, `FILE_ID`, `SITE_ID`, `DX_GROUP`, `AGE_AT_SCAN`, `SEX`. Optional: `--subject_ids_path` if subject list is not at `<data_folder>/subject_IDs.txt`. Add `--use_multimodal` and `--microbiome_path` for multimodal runs.

### Programmatic Usage

```python
import torch
import torch.nn.functional as F
from data.loader import AxisNetDataLoader
from data.multimodal_loader import AxisNetMicrobiomeLoader
from core.transformer_gcn import AxisNetGcnTransformer

# Initialize data loader
dl = AxisNetDataLoader()
mm_loader = AxisNetMicrobiomeLoader(dl)

# Load multimodal data
fmri_data, labels, clinical_data, microbiome_features = \
    mm_loader.load_multimodal(
        microbiome_path='path/to/microbiome.csv',
        top_k=5,
        microbiome_pca_dim=64
    )

# Initialize GCN+Transformer model
model = AxisNetGcnTransformer(
    input_dim=2000,
    num_classes=2,
    dropout=0.2,
    edgenet_input_dim=6,
    edge_dropout=0.3,
    hgc=16,
    lg=4,
    microbiome_dim=64,
    contrastive_weight=0.5,
    n_heads=4,
    n_layers=2
)

# Forward pass
features = torch.tensor(node_features, dtype=torch.float32)
edge_index = torch.tensor(edge_index, dtype=torch.long)
edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32)
microbiome = torch.tensor(microbiome_features, dtype=torch.float32)

logits, edge_weights, micro_embed, brain_embed = model(
    features, edge_index, edgenet_input, microbiome
)

# Compute losses
classification_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
contrastive_loss = model.contrastive_loss(micro_embed, brain_embed, labels)
consistency_loss = model.graph_consistency_loss(edge_index, edge_weights, microbiome)

total_loss = classification_loss + 0.5 * contrastive_loss + 0.05 * consistency_loss
```

### Full Training Example

```python
from config.opt import AxisNetOptions
from scripts.train_eval import run_cv

# Configure and run (GCN+Transformer)
argv = [
    '--train=1',
    '--use_multimodal',
    '--microbiome_path=data/feature-table.biom',
    '--model_type=gcn_transformer',
    '--num_iter=100',
    '--seed=42'
]
opt = AxisNetOptions(argv=argv).initialize()
results = run_cv(opt)

print(f"Accuracy: {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
print(f"AUC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
```

---

## Model: AxisNetGcnTransformer (GCN+Transformer)

AxisNet uses the **GCN+Transformer** hybrid: ChebConv layers with Jumping Knowledge (JK) followed by a Transformer encoder, plus microbiome fusion and contrastive learning.

### Architecture

```
fMRI → ChebConv layers → JK concatenation → Transformer Encoder → Multimodal Fusion → Classifier
```

### Usage

```python
from core.transformer_gcn import AxisNetGcnTransformer

model = AxisNetGcnTransformer(
    input_dim=2000,
    num_classes=2,
    dropout=0.2,
    edgenet_input_dim=6,
    edge_dropout=0.3,
    hgc=16,
    lg=4,
    microbiome_dim=64,
    contrastive_weight=0.5,
    n_heads=4,
    n_layers=2  # Transformer encoder layers
)
```

**Key properties:**
- **Graph Conv**: ChebConv for multi-scale aggregation
- **Attention**: Transformer encoder on JK features
- **Multimodal**: Microbiome encoder + contrastive loss
- **Parameters**: ~100K

---

## Configuration Options

### General Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | train | Execution mode: `train` or `eval` |
| `--use_cpu` | flag | False | Force CPU usage |
| `--seed` | int | 123 | Random seed for reproducibility |
| `--ckpt_path` | str | ./save_models/axisnet | Checkpoint save directory |

### Model Architecture

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_type` | str | gcn_transformer | Model: `gcn_transformer` (GCN+Transformer) |
| `--hidden_dim` | int | 16 | Hidden units in graph conv layers |
| `--num_layers` | int | 4 | Number of graph conv layers |
| `--dropout` | float | 0.2 | Node feature dropout rate |
| `--edge_dropout` | float | 0.3 | Edge dropout rate |
| `--num_classes` | int | 2 | Number of output classes |

### Training

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | 0.01 | Learning rate |
| `--weight_decay` | float | 5e-5 | Weight decay |
| `--epochs` | int | 300 | Maximum number of training epochs |

### Multimodal

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_multimodal` | flag | False | Enable microbiome data integration |
| `--microbiome_path` | str | None | Path to microbiome data file |
| `--microbiome_pca_dim` | int | 64 | PCA-reduced dimension |
| `--microbiome_top_k` | int | 5 | Top-K for pseudo-pairing |
| `--contrastive_weight` | float | 0.5 | Contrastive loss weight |
| `--consistency_weight` | float | 0.05 | Graph consistency loss weight |
| `--warmup_epochs` | int | 10 | Warmup before regularization |
| `--drop_age` | flag | False | Exclude age from phenotypes |
| `--drop_sex` | flag | False | Exclude sex from phenotypes |

### Example Configurations

**Quick test (GCN+Transformer):**
```bash
python scripts.train_eval --mode=train --use_multimodal --model_type=gcn_transformer \
    --epochs=10 --hidden_dim=8 --num_layers=2
```

**Full training (GCN+Transformer):**
```bash
python scripts.train_eval --mode=train --use_multimodal --model_type=gcn_transformer \
    --hidden_dim=32 --num_layers=4 --lr=0.005 --epochs=500 \
    --contrastive_weight=0.3 --consistency_weight=0.1
```

---

## Project Structure

```
AxisNet/
├── __init__.py
├── README.md
│
├── config/
│   ├── __init__.py
│   └── opt.py                  # AxisNetOptions - CLI argument parser
│
├── core/
│   ├── __init__.py
│   ├── axisnet_model.py        # AxisNetGCN (baseline)
│   ├── edge_encoder.py         # EdgeEncoder (PAE equivalent)
│   └── transformer_gcn.py      # AxisNetGcnTransformer (GCN+Transformer)
│
├── data/
│   ├── __init__.py
│   ├── abide_parser.py         # ABIDE dataset parser
│   ├── loader.py               # AxisNetDataLoader
│   ├── multimodal_loader.py    # AxisNetMicrobiomeLoader
│   ├── feature-table.biom      # Sample microbiome data (BIOM format)
│   ├── microbe_data.csv        # Sample microbiome metadata
│   ├── microbe.hdf5            # Sample microbiome features (HDF5)
│   └── subject_IDs.txt         # ABIDE subject IDs
│
├── scripts/
│   ├── __init__.py
│   ├── train_eval.py           # Main training/evaluation script
│   └── run_experiments.py      # Batch experiment runner
│
└── utils/
    ├── __init__.py
    ├── gcn_utils.py            # Graph utilities
    └── metrics.py              # Evaluation metrics (accuracy, AUC, PRF)
```

---

## Results

### AxisNetGcnTransformer (GCN+Transformer) Ablation (seed 123)

From `result0_gcn_transformer.csv`. 10-fold cross-validation on the ABIDE dataset. 

| Variant      | Modality   | Accuracy (mean ± std) | AUC (mean ± std) | Sensitivity | Specificity | F1-Score |
|-------------|------------|------------------------|------------------|-------------|-------------|----------|
| full        | multimodal | 79.3% ± 13.3%         | 0.821 ± 0.149    | 0.834 ± 0.153 | 0.792 ± 0.119 | 0.803 ± 0.110 |
| full        | unimodal   | 77.6% ± 13.1%         | 0.784 ± 0.150    | 0.814 ± 0.164 | 0.757 ± 0.099 | 0.780 ± 0.122 |
| drop_age    | multimodal | 81.7% ± 10.5%         | 0.824 ± 0.108    | 0.868 ± 0.120 | 0.770 ± 0.109 | 0.812 ± 0.104 |
| drop_age    | unimodal   | 77.4% ± 13.7%         | 0.749 ± 0.208    | 0.807 ± 0.171 | 0.813 ± 0.124 | 0.794 ± 0.106 |
| drop_sex    | multimodal | 78.7% ± 17.0%         | 0.786 ± 0.182    | 0.771 ± 0.291 | 0.702 ± 0.290 | 0.730 ± 0.288 |
| drop_sex    | unimodal   | 78.6% ± 15.7%         | 0.761 ± 0.209    | 0.817 ± 0.172 | 0.806 ± 0.141 | 0.801 ± 0.135 |
| drop_age_sex| multimodal | 82.0% ± 12.3%         | 0.846 ± 0.133    | 0.866 ± 0.131 | 0.775 ± 0.126 | 0.816 ± 0.121 |
| drop_age_sex| unimodal   | 79.6% ± 15.2%         | 0.789 ± 0.175    | 0.833 ± 0.154 | 0.770 ± 0.155 | 0.795 ± 0.147 |


---

## Citation

If you use AxisNet in your research, please cite:

```bibtex

@article{axisnet2026,
  title={AxisNet: Multimodal Microbiome-Brain Network Fusion for Neurological Disease Prediction},
  author={Xu},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Acknowledgments

This work builds upon:

- **ABIDE Dataset**: Autism Brain Imaging Data Exchange consortium
- **PyTorch Geometric**: Graph neural network library
- **nilearn**: Neuroimaging machine learning library

### Related Work
- Huang et al., "Edge-variational Graph Convolutional Networks
for Uncertainty-aware Disease Prediction", MICCAI 2020
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
- Cryan et al., "The Microbiota-Gut-Brain Axis", Physiological Reviews, 2019
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please:
1. Open a GitHub issue
2. Check existing documentation and READMEs
3. Review the source code comments

---

**Last Updated**: January 2026  
**Version**: 2.0.0-multimodal
