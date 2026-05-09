# Geometric Analysis of Cinema Data: Feature Selection & Separability Study

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

## 📌 Project Overview
This project performs a rigorous geometric and statistical analysis of the **MovieLens 20M Dataset** to determine which 'movie DNA' traits (genome tags) are the most informative for predictive Machine Learning models. 

Rather than blindly feeding thousands of features into a black-box model, this study explores the geometric separability of movie rating classes. By applying **Variance Optimization**, **Multi-Class Fisher Discriminant Analysis**, and **Mutual Information (MI) Scoring**, the project systematically reduces a 1,128-dimensional feature space down to its most mathematically robust components.

---

## 📊 Dataset & Engineering Pipeline

The raw dataset consisted of over **20,000,263 user ratings** and **11.7 million genome tag scores**. To ensure statistical significance, the data was heavily engineered:

1. **Noise Reduction:** Movies with fewer than 50 user ratings were filtered out, mitigating the "Long Tail" sparsity problem.
2. **Matrix Pivoting:** The 11.7 million tag scores were pivoted into a dense feature matrix ($X$), resulting in **8,568 high-quality movie samples** mapped across **1,128 genome features**.
3. **Target Engineering:** Continuous average ratings were grouped into discrete `rating_class` targets.

**Target Class Distribution (Highly Imbalanced):**
* Class 1: 0.25%
* Class 2: 8.18%
* Class 3: 54.02%
* Class 4: 37.55%

---

## 🔬 Feature Selection Methodologies

To identify the most predictive features, three distinct mathematical approaches were evaluated:

### 1. Variance-Based Selection
* **Objective:** Maximizes data spread without supervision (ignores class labels).
* **Top Features:** `comedy`, `action`, `tense`, `criterion`
* **Findings:** Produced the lowest inter-correlation (Mean absolute inter-correlation: 0.2303). However, it predominantly captured general domain frequencies (broad genres) rather than distinct quality signals.

### 2. Multi-Class Fisher Ratio
* **Objective:** Maximizes linear separability by maximizing the distance between class centroids (between-class scatter) while minimizing variance within each rating class (within-class scatter).
* **Top Features:** `overrated`, `predictable`, `bad plot`, `suprisingly clever`
* **Findings:** Provided better-structured feature sets for quality distinction but exhibited high redundancy (0.6114). Its strict linear assumption failed to clearly isolate the highly complex boundaries between mid-tier movies.

### 3. Mutual Information (MI)
* **Objective:** Quantifies the reduction in uncertainty of the rating class given a specific tag, capturing both linear and non-linear dependencies.
* **Top Features:** `bad plot`, `movielens top pick`, `predictable`, `disaster`
* **Findings:** Proved to be the most robust method for this dataset. MI successfully isolated extreme sentiments (1-star vs. highly-rated) much more distinctly than other methods.

---

## 📐 Geometric Separability & Manifold Structure

Pairwise 2D and 3D scatter projections of the feature spaces led to the following geometric conclusions:
* **Non-Linear Topology:** The 5 rating classes are **NOT** strictly linearly separable. The data forms a dense, continuous manifold.
* **Class Overlap:** While extreme sentiment classes occupy distinct spatial densities, intermediate ratings (2, 3, and 4 stars) overlap heavily, suggesting high user subjectivity in mid-tier ratings.

---

## 🚀 Implications for Future ML Modeling

Based on the covariance heatmaps and geometric overlaps identified during this EDA, any future Machine Learning classification pipeline built on this dataset should adhere to the following guidelines:

1. **Model Selection:** Avoid simple linear models (e.g., Logistic Regression, Linear SVM). They will heavily underfit the overlapping boundaries. **Tree-based ensembles (Random Forest, XGBoost)** or **non-linear mapping algorithms (RBF Kernel SVMs, Neural Networks)** are strongly recommended.
2. **Feature Space Reduction:** The correlation matrices confirmed massive redundancy in the raw 1,128-dimensional space. Models should be trained strictly on **MI-selected features** or a Union/Consensus list to ensure robustness and prevent multicollinearity-induced instability.
3. **Handling Class Imbalance:** Due to the heavy overlap and subjectivity of 3-star and 4-star classes, future models should deploy hierarchical classification (e.g., binary classifying "Good" vs. "Bad" first) or aggregate middle-tier classes.

---

## 💻 Tech Stack
* **Python 3**
* **Pandas / NumPy:** Memory-optimized data manipulation and matrix operations.
* **Scikit-Learn:** Mutual Information classification (`mutual_info_classif`).
* **Matplotlib / Seaborn:** High-dimensional 2D/3D visualizations and correlation heatmaps.

---

## 📥 How to Run Locally

*Note: Due to GitHub's 100MB file size limit, the raw `.csv` data files are not included in this repository.*

1. Clone this repository.
2. Download the [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).
3. Extract `rating.csv`, `genome_scores.csv`, `movie.csv`, and `genome_tags.csv` into the root directory.
4. Run the Jupyter Notebook sequentially. 

*(The notebook includes dynamic `gc.collect()` garbage collection to ensure the 11.7 million row pivot operations do not exceed standard local RAM limits).*