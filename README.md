# Machine Learning Assignments

Collection of homework assignments for the **Machine Learning** course (A.Y. 2025/2026).  
**Master of Science in Engineering in Computer Science and Artificial Intelligence**

## Student
**Nicolas Leone** - Student ID: 1986354

---

## 📚 Overview

This repository contains two comprehensive assignments demonstrating mastery of supervised and unsupervised machine learning techniques, neural network architectures, and dimensionality reduction methods. Both assignments utilize real-world datasets from the **UCI Machine Learning Repository** and implement industry-standard workflows including preprocessing, model training, hyperparameter tuning, and rigorous evaluation.

**Key Technologies**: Python, NumPy, Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

---

## 📊 First Assignment: Wine Quality Classification

### Objective
**Binary classification** of wine quality (good vs. poor) using six distinct machine learning algorithms, with comprehensive comparative analysis and hyperparameter optimization.

### Dataset
- **Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Samples**: 6,497 wines (1,599 red + 4,898 white)
- **Features**: 11 physicochemical properties + wine type
- **Target**: Binary quality label (good/poor) derived from expert scores
- **Task**: Binary classification with class imbalance handling

### Implemented Algorithms

#### 1. **Logistic Regression**
- **Baseline linear classifier** with L2 regularization
- Sigmoid activation for probability estimation
- **Grid search**: Regularization parameter C, solver types
- Iterative optimization with convergence monitoring
- **Best Performance**: ~84% accuracy (balanced dataset)

#### 2. **Support Vector Machine (SVM)**
- **Kernel trick** for non-linear decision boundaries
- RBF kernel with configurable gamma parameter
- **Grid search**: C (regularization), gamma (kernel coefficient)
- Support vector identification and margin maximization
- **Best Performance**: ~84% accuracy with RBF kernel

#### 3. **K-Nearest Neighbors (KNN)**
- **Instance-based learning** with distance-weighted voting
- Euclidean distance computation
- **Grid search**: k neighbors (1-30), distance weights
- No explicit training phase (lazy learning)
- **Best Performance**: ~83% accuracy with k=7

#### 4. **Decision Tree**
- **Entropy-based splitting** with information gain
- Recursive tree construction with max depth control
- Gini impurity and entropy splitting criteria
- **Grid search**: max depth, min samples split, criterion
- Feature importance extraction
- **Best Performance**: ~82% accuracy (depth=10)

#### 5. **Random Forest**
- **Ensemble of decision trees** with bootstrap aggregating
- Out-of-bag error estimation
- **Grid search**: n_estimators, max depth, min samples split
- Feature importance via mean decrease in impurity
- **Best Performance**: ~86% accuracy (100 trees)

#### 6. **Multi-Layer Perceptron (MLP)**
- **Neural network** with hidden layers and backpropagation
- ReLU activation in hidden layers, sigmoid output
- **Grid search**: hidden layer sizes, learning rate, alpha (L2 penalty)
- Adam optimizer with adaptive learning rates
- **Best Performance**: ~85% accuracy (100,64 architecture)

### Technical Highlights
- ✅ **Class balancing**: SMOTE (Synthetic Minority Over-sampling) to address 85/15 imbalance
- ✅ **Feature engineering**: Wine type encoding, correlation analysis
- ✅ **Preprocessing**: StandardScaler normalization, train/val/test split (60/20/20)
- ✅ **Hyperparameter tuning**: Exhaustive GridSearchCV with 5-fold cross-validation
- ✅ **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrices
- ✅ **Visualization**: ROC curves, feature importance plots, learning curves

### Key Results
- **Best Algorithm**: Random Forest (86.2% test accuracy)
- **Most Interpretable**: Decision Tree (feature importance analysis)
- **Fastest Training**: Logistic Regression (linear complexity)
- **Most Robust**: SVM with RBF kernel (handles non-linearity effectively)

---

## 🧠 Second Assignment: Dimensionality Reduction & Multi-Task Learning

### Objective
Systematic comparison of **PCA** (linear) vs. **Autoencoders** (non-linear) for dimensionality reduction across three machine learning paradigms: **clustering**, **classification**, and **regression**.

### Datasets

#### 1. **Wine Quality** (Clustering Task)
- **Samples**: 6,497 wines
- **Features**: 11 physicochemical properties
- **Task**: Unsupervised K-Means clustering (K=2 for red vs. white)
- **Validation**: Adjusted Rand Index against true wine types

#### 2. **Optical Handwritten Digits** (Classification Task)
- **Source**: [UCI Optical Recognition Dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- **Samples**: 5,620 digit images (8×8 pixels)
- **Features**: 64 pixel intensities (0-16 range)
- **Classes**: 10 digits (0-9)
- **Task**: Multi-class image classification with CNN

#### 3. **Concrete Compressive Strength** (Regression Task)
- **Source**: [UCI Concrete Dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **Samples**: 1,030 concrete formulations
- **Features**: 8 mixture composition variables (cement, water, aggregates, age)
- **Target**: Compressive strength (MPa), range 2-82 MPa
- **Task**: Continuous prediction with feedforward neural network

### Dimensionality Reduction Techniques

#### **Principal Component Analysis (PCA)**
- **Linear transformation** maximizing variance retention
- Configured for **95% explained variance**
- Automatic component selection
- Results:
  - Wine Quality: 11 → 7 components (36.4% reduction)
  - Optical Digits: 64 → 29 components (54.7% reduction)
  - Concrete Strength: 8 → 6 components (25.0% reduction)

#### **Autoencoders**
- **Non-linear neural network** with encoder-decoder architecture
- Symmetric design with bottleneck latent space
- Latent dimensions matched to PCA components for fair comparison
- MSE reconstruction loss with Adam optimizer
- Early stopping (patience=10) to prevent overfitting
- Training: 100 max epochs, batch size 32 (16 for concrete)

### Model Implementations

#### **K-Means Clustering** (Wine Quality)
- **Algorithm**: Iterative centroid-based clustering
- **Configuration**: K=2 clusters, k-means++ initialization
- **Metrics**: Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index
- **Results**:
  - Original features: Silhouette 0.4217
  - **PCA (Best)**: Silhouette 0.4356 ⭐
  - Autoencoder: Silhouette 0.4183

#### **Convolutional Neural Network** (Optical Digits)
- **Architecture**: 
  - Original: 2 Conv2D layers + MaxPooling + Dense layers
  - PCA/Autoencoder: Dense-only architecture (spatial structure lost)
- **Configuration**: Categorical crossentropy loss, Adam optimizer, dropout 0.3
- **Early Stopping**: Patience=15 epochs
- **Results**:
  - **Original (Best)**: 97.16% test accuracy ⭐
  - Autoencoder: 96.94% test accuracy
  - PCA: 96.72% test accuracy

#### **Feedforward Neural Network** (Concrete Strength)
- **Architecture**: Multi-layer perceptron with 4 hidden layers
- **Configuration**: MSE loss, Adam optimizer, dropout 0.2
- **Metrics**: MAE, RMSE, R² Score
- **Results**:
  - **Original (Best)**: MAE 4.91 MPa, R² 0.8847 ⭐
  - Autoencoder: MAE 5.15 MPa, R² 0.8741
  - PCA: MAE 5.33 MPa, R² 0.8626

### Technical Highlights
- ✅ **No data leakage**: All scalers/transformations fit exclusively on training data
- ✅ **Stratified splits**: Class-balanced splits for classification (60/20/20)
- ✅ **Comprehensive evaluation**: 9 model-dataset combinations (3 tasks × 3 representations)
- ✅ **Visualization suite**: Confusion matrices, residual plots, cluster projections, training histories
- ✅ **Statistical validation**: Silhouette analysis, ROC curves, residual normality tests
- ✅ **Comparative analysis**: Side-by-side performance tables and bar charts

### Key Findings

#### When to Use PCA:
- ✅ **Clustering tasks** (noise reduction improves separation)
- ✅ **Linear feature correlations** (common in chemical/physical measurements)
- ✅ **Fast, deterministic transformation** required
- ✅ **Interpretability** is important (principal components are analyzable)

#### When to Use Autoencoders:
- ✅ **Non-linear patterns** exist (e.g., image data, complex interactions)
- ✅ **Large datasets** available (>10K samples for robust training)
- ✅ **Classification/regression** tasks with complex features
- ✅ **Computational resources** permit iterative optimization

#### When to Keep Original Features:
- ✅ **Already manageable dimensionality** (<50-100 features)
- ✅ **Small datasets** where reduction may lose critical information
- ✅ **Maximum accuracy** is paramount over computational cost

---

## 🎯 Key Features Across Both Assignments

- ✅ **UCI-sourced datasets** meeting academic standards
- ✅ **Rigorous preprocessing** with stratified splits and standardization
- ✅ **Extensive hyperparameter tuning** via grid search and cross-validation
- ✅ **Multiple evaluation metrics** appropriate for each task type
- ✅ **Professional visualizations** (Matplotlib, Seaborn) for results interpretation
- ✅ **Data leakage prevention** through proper train/val/test isolation
- ✅ **Reproducibility**: Fixed random seeds (42) for all stochastic operations
- ✅ **TensorFlow/Keras** for neural network implementations
- ✅ **Comparative analysis** with statistical significance testing
- ✅ **LaTeX documentation** with detailed technical reports

---

## 🔧 Requirements

```python
# Core Libraries
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0  # For SMOTE (First Assignment)

# Deep Learning
tensorflow>=2.13.0
keras>=2.13.0

# Utilities
openpyxl>=3.1.0  # For Excel dataset loading
```

## 📂 Repository Structure

```
.
├── README.md                          # This file
├── First Assignment/
│   ├── First_Assignment_Nicolas_Leone_1986354.ipynb
│   ├── First_Assignment_Nicolas_Leone_1986354.tex
│   ├── First_Assignment_Nicolas_Leone_1986354.pdf
│   ├── Instructions.pdf
│   └── Makefile                       # LaTeX compilation
└── Second Assignment/
    ├── Second_Assignment_Nicolas_Leone_1986354.ipynb
    ├── Second_Assignment_Nicolas_Leone_1986354.tex
    ├── Second_Assignment_Nicolas_Leone_1986354.pdf
    ├── Instructions.pdf
    └── Makefile                       # LaTeX compilation
```

## 🚀 Execution Instructions

### Running Jupyter Notebooks

Both notebooks execute end-to-end without errors. Run cells sequentially from top to bottom.

```bash
# Navigate to assignment directory
cd "First Assignment"   # or "Second Assignment"

# Launch Jupyter
jupyter notebook
```

### Compiling LaTeX Reports

```bash
cd "First Assignment"   # or "Second Assignment"
make                     # Compiles .tex to .pdf
make clean              # Removes auxiliary files
```

---

## 📊 Results Summary

### First Assignment (Classification)
| Algorithm | Test Accuracy | Training Time | Best Hyperparameters |
|-----------|--------------|---------------|---------------------|
| **Random Forest** ⭐ | **86.2%** | Medium | n_estimators=100, max_depth=20 |
| MLP Neural Network | 85.3% | Slow | (100,64), alpha=0.001 |
| SVM (RBF) | 84.1% | Slow | C=10, gamma=0.01 |
| Logistic Regression | 83.8% | **Fast** | C=1.0, L2 penalty |
| KNN | 83.2% | Fast (inference slow) | k=7, uniform weights |
| Decision Tree | 81.9% | **Fast** | max_depth=10, entropy |

### Second Assignment (Dimensionality Reduction)
| Task | Best Representation | Metric | Value |
|------|-------------------|--------|-------|
| **Clustering** (Wine) | **PCA** ⭐ | Silhouette Score | 0.4356 |
| **Classification** (Digits) | **Original** ⭐ | Test Accuracy | 97.16% |
| **Regression** (Concrete) | **Original** ⭐ | R² Score | 0.8847 |

**Key Insight**: Dimensionality reduction is not universally beneficial. PCA excels in clustering (noise reduction), while original features retain critical information for supervised tasks on moderate-dimensional datasets.

---

## 📖 Documentation

Each assignment includes:
- **Jupyter Notebook** (.ipynb): Full implementation with code, visualizations, and analysis
- **LaTeX Report** (.tex/.pdf): Academic-style technical report (25-35 pages)
  - Mathematical formulations
  - Algorithm descriptions
  - Experimental setup
  - Results interpretation
  - Comparative analysis
  - Conclusions and insights
