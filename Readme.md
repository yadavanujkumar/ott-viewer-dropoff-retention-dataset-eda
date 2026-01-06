# 🎬 OTT Viewer Drop-Off & Retention Risk Dataset (v1.0)

## 📌 Overview

This dataset provides **episode-level viewer behavior data** for OTT (streaming) TV series, focused on **drop-off patterns, retention risk, and engagement dynamics** across episodes and seasons.

Unlike traditional catalog datasets (genres, ratings, cast), this dataset is designed to support **realistic retention analysis**, similar to how streaming platforms study *when* and *why* viewers stop watching.

Each row represents **one episode**, enabling fine-grained analysis across seasons and episode progressions.

---

## 🚀 Ready to Use: Complete Data Science/AI/ML Analysis

This repository now includes **comprehensive analysis tools** demonstrating what you can do with this dataset:

### ⚡ Quick Start (5 minutes)
```bash
pip install -r requirements.txt
python quick_demo.py
```

### 🔬 Full Analysis
```bash
python ott_comprehensive_analysis.py
```

### 📓 Interactive Notebook
```bash
jupyter notebook OTT_Analysis_Notebook.ipynb
```

**📚 Documentation:** See [`QUICK_START.md`](QUICK_START.md) and [`ANALYSIS_DOCUMENTATION.md`](ANALYSIS_DOCUMENTATION.md)

---

## 🎯 What Can You Do with This Dataset?

### ✅ Implemented Analyses:

1. **Machine Learning Models**
   - ✅ Drop-off prediction (99.5% accuracy)
   - ✅ Retention risk classification (99.0% accuracy)
   - ✅ Probability regression (R² = 1.000)

2. **Advanced Analytics**
   - ✅ Clustering analysis (viewer segmentation)
   - ✅ Time series analysis (episode progression)
   - ✅ Feature importance ranking

3. **Business Intelligence**
   - ✅ High-risk episode identification
   - ✅ Platform & genre benchmarking
   - ✅ Actionable recommendations

---

---

## 📦 Files in This Repository

- **`ott_viewer_dropoff_retention_us_v1.0.csv`** - Main dataset (33K+ episodes)
- **`quick_demo.py`** - 5-minute quick demonstration
- **`ott_comprehensive_analysis.py`** - Full analysis script
- **`OTT_Analysis_Notebook.ipynb`** - Interactive Jupyter notebook
- **`requirements.txt`** - Python dependencies
- **`QUICK_START.md`** - Getting started guide
- **`ANALYSIS_DOCUMENTATION.md`** - Comprehensive documentation
- **`Readme.md`** - This file (dataset info + analysis overview)

---

## 🎯 Purpose of the Dataset

The dataset is built to help answer real-world OTT analytics questions such as:

* At which episode do viewers start dropping off?
* How does retention change from Season 1 to later seasons?
* Do finales improve or hurt retention?
* How do pacing, hook strength, and cognitive load affect churn?
* Which episodes are risky for late-night recommendations?

---

## 📊 Dataset Highlights

* **450+ unique TV shows**
* **Episode-level granularity** (one row = one episode)
* **Up to 4 seasons per show**
* **All episodes included per season**
* US-region streaming platform availability
* Suitable for ML, EDA, clustering, and time-series analysis

The dataset is intentionally large to allow meaningful modeling rather than toy examples.

---

## 🧠 Data Source & Methodology

* TV show metadata is collected from **TMDB (The Movie Database) API**
* Shows are sampled using TMDB’s **dynamic “Popular TV” ranking**, which reflects *current audience interest* rather than all-time popularity
* Streaming platforms are resolved using TMDB Watch Providers (**US region**)

Viewer engagement and retention signals are **synthetically generated**, but grounded in **realistic OTT viewing behavior assumptions**, such as:

* Strong hooks reduce early drop-off
* High cognitive load increases churn risk
* Mid-season fatigue affects retention
* Viewer behavior differs between early and late episodes

No real user data is included.

---

## 🧾 Column Description

| Column                 | Description                           |
| ---------------------- | ------------------------------------- |
| `show_id`              | TMDB TV show ID                       |
| `title`                | TV show title                         |
| `platform`             | Primary streaming platform (US)       |
| `genre`                | Primary genre                         |
| `release_year`         | First air year                        |
| `season_number`        | Season index                          |
| `episode_number`       | Episode index within season           |
| `episode_duration_min` | Episode runtime (minutes)             |
| `pacing_score`         | Narrative pacing (1–10)               |
| `hook_strength`        | Immediate viewer hook (1–10)          |
| `dialogue_density`     | Dialogue intensity (1–10)             |
| `visual_intensity`     | Visual stimulation level (1–10)       |
| `avg_watch_percentage` | Average watch completion (%)          |
| `pause_count`          | Estimated pause events                |
| `rewind_count`         | Estimated rewind events               |
| `skip_intro`           | Whether intro is skipped (0/1)        |
| `cognitive_load`       | Mental effort required (1–10)         |
| `attention_required`   | Low / Medium / High                   |
| `night_watch_safe`     | Suitable for late-night viewing (0/1) |
| `drop_off`             | Drop-off indicator (0/1)              |
| `drop_off_probability` | Drop-off likelihood (0–1)             |
| `retention_risk`       | Low / Medium / High                   |
| `dataset_version`      | Dataset version                       |

---

## ⚠️ Important Notes

* Repeated `show_id` values are **expected**, as each show appears once per episode
* Retention labels are **derived from behavioral signals**, not manually assigned
* The dataset is **intentionally imbalanced**, reflecting real OTT ecosystems where most content falls into medium engagement
* Classic and newer shows may appear together depending on current popularity signals

---

## 💡 Suggested Use Cases

* OTT churn prediction models
* Episode-level retention analysis
* Season fatigue and finale impact studies
* Content recommendation research
* Feature importance and explainability (SHAP)
* Machine learning coursework and portfolios

---

## 📜 License & Attribution

* TMDB data used in accordance with TMDB API terms
* Dataset intended for **educational and research purposes only**

---

## ⭐ Final Note

This dataset is designed to encourage **analysis, modeling, and experimentation**, not just exploration.

If you create notebooks or models using it, feel free to share them with the community.

---