# FedSPOC: Federated Slice Prediction with Offloading Control

This repository implements **FedSPOC**, a federated learning framework for 5G network slice prediction with:

- Fog-based client clustering
- Fairness-aware client selection
- Meta-reward shaping
- Global holdout validation
- SHAP explainability
- Sensitivity analysis
- Cross-seed aggregation

---

## 📊 Dataset: CRAWDAD 5G

This project uses raw 5G RAN monitoring data from the **CRAWDAD wireless data archive**.

Official Source:
https://crawdad.org/

Example 5G dataset (ElasticMon 5G):
https://crawdad.org/eurecom/elasticmon5G2019/

⚠️ NOTE:
The dataset is **NOT included in this repository** due to licensing and size constraints.

---

## 📥 How to Obtain the Dataset

1. Visit:
   https://crawdad.org/eurecom/elasticmon5G2019/

2. Download:
   - 01-RawDatasets.zip
   - 02-PreprocessedDatasets.zip

3. Extract the files.

4. Convert or preprocess into the format expected by this project:
   - crawdad_250.pkl
   - crawdad_250_df.pkl
   - crawdad_250_renamed.parquet
   - crawdad_250.duckdb

---

## 📁 Project Structure

```
FedSPOC/
│
├── newfedspoc500.ipynb
├── requirements.txt
├── README.md
│
├── data/                 ← Place CRAWDAD files here
│   ├── crawdad_250.pkl
│   ├── crawdad_250_df.pkl
│   ├── crawdad_250_renamed.parquet
│   └── crawdad_250.duckdb
│
├── results/
└── checkpoints/
```

Update in the notebook:

```python
PKL_DIR = "./data"
```

---

## 🚀 Installation

Create environment:

```bash
pip install -r requirements.txt
```

If running in Colab:

```python
!pip install -r requirements.txt
```

---

## ▶️ Running the Experiment

Inside the notebook:

```python
NUM_CLIENTS = 500
NUM_ROUNDS = 30
USE_FEDSPOC = True
USE_GLOBAL_HOLDOUT = True
```

Then execute all cells.

---

## 📈 Outputs

Results are saved in:

- `results/federated_metrics_seed_X.csv`
- `results/fog_metrics_seed_X.csv`
- `results/shap_summary.png`
- `results/sensitivity_analysis.csv`
- `results/summary_all_seeds.csv`

---

## 🧠 Key Features

- Fairness-aware client selection (Jain index, entropy fairness)
- Meta-weight adaptive reward
- Curriculum-aware masking
- SHAP feature attribution
- Sensitivity analysis
- Cross-seed reproducibility

---

## 📌 Reproducibility Notes

- Google Drive paths should be replaced with relative paths.
- GPU is recommended.
- Ray simulation may take significant runtime.
- Dataset must be downloaded separately from CRAWDAD.

---

## 📚 Citation

If you use the CRAWDAD dataset, please cite:

CRAWDAD wireless data archive.
https://crawdad.org/

ElasticMon 5G dataset:
https://crawdad.org/eurecom/elasticmon5G2019/

---

## ⚖ License Notice

CRAWDAD datasets are subject to their respective licenses.
This repository does NOT redistribute CRAWDAD data.
Users must obtain data directly from the official CRAWDAD archive.
