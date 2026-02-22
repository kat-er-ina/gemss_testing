# Real-World Datasets for GEMSS Evaluation

This directory contains curated real-world datasets prepared for evaluating GEMSS feature selection. All datasets are provided as clean CSV files ready for use with the [GEMSS Explorer application](https://github.com/kat-er-ina/gemss/tree/main/app).

---

## Available Datasets

### Metabolomics Datasets

All metabolomics datasets are sourced from [MetaboLights](https://www.ebi.ac.uk/metabolights/), preprocessed, and stored in [`data/metabolights/`](data/metabolights/).

#### MTBLS1: Type 2 Diabetes
- **Source**: [MetaboLights MTBLS1](https://www.ebi.ac.uk/metabolights/MTBLS1)
- **Task**: Binary classification (diabetes vs. healthy)
- **Samples**: 132 human urinary metabolite profiles
- **Description**: NMR-based metabolomic study comparing unmedicated type 2 diabetes patients with good dietary control against healthy controls
- **Features**: ~220 metabolite measurements
- **Target variable**: `metadata__Factor Value[Metabolic syndrome]`

#### MTBLS2: Arabidopsis Genotype Comparison
- **Source**: [MetaboLights MTBLS2](https://www.ebi.ac.uk/metabolights/MTBLS2)
- **Task**: Binary classification (2 genotypes)
- **Samples**: 16 plant samples (*Arabidopsis thaliana*)
- **Description**: LC/MS-based profiling comparing wild-type and cyp79B2 cyp79B3 double knockout plants. High-dimensional challenge (p >> n)
- **Features**: ~15274 metabolite measurements (two variants available: full and identified-only with 43 features)
- **Target variable**: `metadata__Factor Value[Genotype]`
- **Challenge**: Extreme high-dimensional scenario

#### MTBLS8: Yeast Growth Control
- **Source**: [MetaboLights MTBLS8](https://www.ebi.ac.uk/metabolights/MTBLS8)
- **Task**: Regression or multi-class classification
- **Samples**: 126 yeast samples (*Saccharomyces cerevisiae*)
- **Description**: Systems biology study of eukaryotic cell growth control using chemostat culture at different growth rates
- **Features**: ~383 metabolite measurements
- **Target variables**: Multiple growth-related factors available
- **Note**: Can be used for growth rate prediction (regression) or classification by growth conditions

#### MTBLS12968: PCOS and Preterm Birth
- **Source**: [MetaboLights MTBLS12968](https://www.ebi.ac.uk/metabolights/MTBLS12968)
- **Task**: Binary classification (2 independent outcomes)
- **Samples**: 149 pregnant women plasma samples
- **Description**: Targeted metabolomics study investigating mid-trimester plasma profiles for PCOS and preterm birth biomarkers
- **Features**: ~490 metabolite measurements (UHPLC-QTRAP MS/MS)
- **Target variables**: 
  - `metadata__PCOS`: PCOS vs. control
  - `metadata__PRETERM`: Preterm vs. term delivery
- **Multi-task potential**: Two related binary classification problems from the same samples

### Medical Imaging Dataset

#### Colonoscopy: Gastrointestinal Lesions
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/495/gastrointestinal+lesions+in+regular+colonoscopy)
- **Task**: Binary classification (benign vs. malignant)
- **Samples**: 152 observations (76 lesions × 2 imaging modalities)
- **Description**: Texture, color, and shape features extracted from colonoscopy images
- **Features**: 698 image-derived features
- **Target variable**: `response` (0=benign/hyperplasic, 1=malignant/adenoma)
- **Metadata**: 
  - `lesion_type`: Original 3-class label (1=hyperplasic, 2=serrated adenoma, 3=adenoma)
  - `light_type`: Imaging modality (1=White Light, 2=Narrow Band Imaging)
- **Location**: [`data/colonoscopy/`](data/colonoscopy/)
- **Status**: Preprocessing notebook available at [`notebooks/preprocessing_colonoscopy.ipynb`](notebooks/preprocessing_colonoscopy.ipynb)

---

## Dataset Characteristics

**Diversity of Challenges:**
- **High-dimensional**: MTBLS2 (p >> n, extreme scenario)
- **Small sample**: MTBLS2 (n=16)
- **Multi-task**: MTBLS12968 (two related outcomes)
- **Imaging-derived**: Colonoscopy (texture/color/shape features)
- **Class imbalance**: Various minority class scenarios

**Preprocessing Status:**
- ✅ **All metabolomics datasets**: Clean CSV format with metadata columns prefixed `metadata__`
- ✅ **Colonoscopy**: Preprocessing notebook provided, ready for use

**File Format:**
- CSV files with samples as rows, features as columns
- Index column contains sample identifiers
- Metadata columns (prefixed `metadata__`) contain potential target variables and stratification factors
- Numeric features represent metabolite intensities or image-derived measurements

---

## Using These Datasets

### With GEMSS Explorer

These datasets are designed for the **[GEMSS Explorer](https://github.com/kat-er-ina/gemss/tree/main/app)** interactive application.

**Quick Start:**
1. Clone GEMSS repository: `git clone https://github.com/kat-er-ina/gemss.git`
2. Install dependencies: `uv sync` (from GEMSS repo root)
3. Launch app: `uv run marimo run app/gemss_explorer.py`
4. Upload a dataset CSV from this directory
5. Select index column and target variable (look for `metadata__` prefix)
6. Configure GEMSS parameters and run analysis

### Dataset Selection Guide

**For testing high-dimensional performance:**
- Use MTBLS2 (full version with 15,274 features)

**For small sample challenges:**
- Use MTBLS2 (only 16 samples)

**For multi-solution exploration:**
- Use MTBLS12968 (two related classification tasks)
- Compare solutions for PCOS vs. preterm prediction

**For imaging feature analysis:**
- Use Colonoscopy dataset
- Compare solutions between imaging modalities

**For standard benchmarking:**
- Use MTBLS1 (diabetes, n=132)
- Use MTBLS8 (yeast, n=126)

---

## Adding New Datasets

### From MetaboLights

See [`notebooks/get_data_from_metabolights.ipynb`](notebooks/get_data_from_metabolights.ipynb) for examples of:
- Downloading studies from MetaboLights FTP repository
- Extracting metabolite intensity matrices
- Processing metadata and experimental factors
- Saving as clean CSV files

### From Other Sources

**Requirements:**
1. CSV format with samples as rows
2. Numeric features (missing values acceptable)
3. At least one target variable for classification or regression
4. Clear documentation of:
   - Data source and citation
   - Sample characteristics
   - Feature descriptions
   - Target variable(s)

**Preprocessing Guidelines:**
- Prefix metadata/target columns with `metadata__`
- Use descriptive feature names when available
- Document preprocessing steps in a notebook
- Include data source and citation information

---

## Directory Structure

```
real_world_applications/
├── data/
│   ├── metabolights/          # MetaboLights datasets (preprocessed)
│   │   ├── metabolights_MTBLS1.csv
│   │   ├── metabolights_MTBLS2.csv
│   │   ├── metabolights_MTBLS8.csv
│   │   ├── metabolights_MTBLS12968.csv
│   │   └── [additional preprocessed variants]
│   └── colonoscopy/           # Colonoscopy imaging dataset
│       ├── data.txt           # Raw data file
│       └── info.txt           # Dataset documentation
├── notebooks/                 # Data acquisition & preprocessing
│   ├── get_data_from_metabolights.ipynb
│   └── preprocessing_colonoscopy.ipynb
├── results/                   # Reserved for GEMSS Explorer outputs (git-ignored)
└── README.md                  # This file
```

---

## Citation

When using these datasets, please cite both:

1. **The original data source** (see links above for each dataset)
2. **The GEMSS paper** if using GEMSS for analysis:

```bibtex
@misc{henclova2026gemssvariationalbayesianmethod,
      title={GEMSS: A Variational Bayesian Method for Discovering Multiple Sparse Solutions in Classification and Regression Problems}, 
      author={Kateřina Henclová and Václav Šmídl},
      year={2026},
      eprint={2602.08913},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.08913}, 
}
```

