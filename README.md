# 🏛️ RVB Societal Challenge – Clustering & Solution Recommendation Tool

## Introduction

This repository contains a **decision-support tool developed for the Rijksvastgoedbedrijf (RVB)**.
The tool is designed to:

- **Cluster buildings and energy projects** based on spatial and attribute data
- **Analyze societal and technical characteristics** of these clusters
- **Automatically link suitable solutions** to each cluster
- Support **policy analysis, exploration, and scenario-based decision making**

The application is built as a **Streamlit web tool**, supported by a robust data preparation and clustering pipeline.

---

## High-Level Workflow

1. Raw data ingestion  
2. One-time data cleaning & normalization  
3. Clustering of locations and assets  
4. Linking policy / technical solutions to clusters  
5. Interactive exploration via Streamlit UI  
6. Optional export of selected results  

---

## 📁 Project Structure

```
Societal-Challenge/
│
├── cleaned_data/
│   └── Intermediate cleaned datasets
│
├── Data/
│   ├── 20250827_export_VKAs/
│   ├── Bouwwerken_netcongestie/
│   ├── Netherlands_shapefile/
│   ├── energieprojecten.gpkg
│   └── TUD Basislijst Bekende aansluitingen (sept 25).xlsx
│
├── streamlit2.0/
│   ├── .streamlit_uploads/        # Auto-loaded user uploads
│   ├── afbeeldingen/              # Images shown in the UI
│   ├── outputs/                   # Exported user-selected results
│   │
│   ├── app.py                     # Streamlit entry point (main app)
│   ├── clustering_pipeline.py     # Clustering + solution linking logic
│   ├── data_access.py             # Centralized data loading helpers
│   ├── recommendations.py         # Recommendation logic
│   ├── toolbox_recommender.py     # Solution matching engine
│   ├── toolbox_solutions.yaml     # Definition of all solutions
│   ├── ui_components.py           # UI building blocks
│   ├── ui_state.py                # Streamlit session state handling
│   ├── validators.py              # Input & data validation
│   └── combined.csv               # Example combined dataset
│
├── Datacleaning.py                # One-time raw data cleaning pipeline
├── get_energydata_from_URL.py     # Fetches external energy data
│
├── requirements.txt
├── README.md
└── .gitattributes
```

---
## Supporting content

In the supporting content files are the data management plan and the informed consent files from the interviews. 

## ⚙️ Installation & Running the Tool (IMPORTANT)

### Open a Terminal in the Project Root

**Windows**
- Navigate to the project folder
- Hold **Shift + Right Click**
- Choose **Open PowerShell window here**

**macOS / Linux**
```bash
cd path/to/Societal-Challenge
```

---

### Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Activate:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Tool on a local server
### It is really important to open an terminal in the streamlit2.0 directiory otherwise it won't work.
#### (you can do this by right click on the streamlit2.0 directory --> open in --> terminal)
#### Run the next line in de streamlit2.0 terminal
```bash
streamlit run app.py
```

If not opened automatically open the printed URL (usually http://localhost:8501) in your browser.

---
## 🔧 How the Tool Works

The tool is structured as a **clear, sequential pipeline**, where each component has a specific responsibility and connects explicitly to the next step.

**1. Raw Data (`Data/`)**  
All analyses start from raw input data, including energy projects (VKA’s), building attributes, and spatial datasets. These files are treated as read-only.

**2. External Energy Data (`get_energydata_from_URL.py`)**  
Optional script to retrieve energy-related data from online sources. This step only needs to be run when updated external data is required.

**3. Data Cleaning (`Datacleaning.py`)**  
This script cleans, harmonizes, and merges all raw datasets into a single standardized dataset. It is designed to be run once, and only rerun when new raw data is added.

**4. Intermediate Storage (`cleaned_data/`)**  
Intermediate cleaned datasets are stored here for transparency and intermediate analysis. These files are not directly used by the Streamlit application. But could be used for further analysis.

**5. Core Application (`streamlit2.0/`)**  
This folder contains the decision-support tool itself. Cleaned data is loaded, clustered, and enriched with contextual information.

**6. Clustering & Solution Matching (`clustering_pipeline.py`)**  
Buildings and projects are clustered based on spatial and attribute data. Cluster characteristics are then matched to suitable solutions.

**7. Solution Definitions (`toolbox_solutions.yaml`)**  
All solutions are defined declaratively in a YAML file. Solutions can be added or adjusted without modifying code.

**8. User Interface (`app.py`)**  
The Streamlit interface allows users to explore clusters, inspect linked solutions, and export selected results.

**9. Outputs (`outputs/`)**  
User-selected data and results can be exported and stored for reporting and further analysis.

**Pipeline summary:**

```
Raw Data → Data Cleaning → Clustering → Solution Matching → Streamlit UI → Outputs
```
---

## Target Users

- Rijksvastgoedbedrijf (RVB)
- Policy analysts
- Energy & spatial planning teams
- Researchers
