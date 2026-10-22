# Agricultural Analytics System for Sustainable Crop Yields

An end-to-end Python data analysis platform designed to take the guesswork out of agricultural nutrient management. By converting raw microclimate records into clear, actionable insights, this system helps farmers move away from generic "blanket" rules and make data-driven decisions tailored to their specific crop and soil needs.

---

## Key Features & Capabilities

* **Automated Data Cleaning & Quality Control:** Strips incomplete or duplicate field records to maintain data integrity across thousands of samples.
* **IQR Outlier Winsorization:** Uses Tukey's $1.5 \times \text{IQR}$ (Interquartile Range) method to cap extreme climate sensor misreadings without discarding valuable field data.
* **Visual Data Profiling:** Generates intuitive Matplotlib distributions and Seaborn heatmaps, boxplots, and countplots to analyze environmental impacts on soil Nitrogen, Phosphorous, and Potassium (NPK) levels.
* **Statistical Hypothesis Testing:** Runs SciPy One-Way ANOVA tests to scientifically determine whether soil type alone dictates nutrient levels or if microclimate factors and crop requirements are primary drivers.

---

## Academic Syllabus Alignment

This project practically demonstrates core concepts from our academic Python curriculum:

| Module | Covered Concepts & Implementations |
| :--- | :--- |
| **1. Intro to Python** | Decision logic (`if/else`), `for` loops, and robust `try-except` error handling for dataset loading. |
| **2. Modules & Structure** | Modular programming architecture separating preprocessing logic (`data_processor.py`) from analytics and visualization (`agri2.py`). |
| **3. NumPy & Pandas** | DataFrames, missing value handling (`dropna()`), duplicate removal (`drop_duplicates()`), `groupby()` aggregations, and array bounds-clipping (`np.clip()`, `np.percentile()`). |
| **4. Matplotlib & Seaborn** | Distribution histograms with mean/median overlays (`axvline`), correlation heatmaps, annotated boxplots, and countplots with externalized legends. |
| **5. SciPy Statistics** | Parametric hypothesis testing via `scipy.stats.f_oneway` (One-Way ANOVA). |

---

## Repository Structure

```text
├── data_processor.py   # Data ingestion, cleaning, and IQR outlier winsorization module
├── agri2.py            # Main analytics, visual data profiling, and statistical testing script
├── data_core.csv       # Multi-variate agricultural dataset (8,000 field records)
└── README.md           # Project documentation
