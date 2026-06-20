# Titanic Survival Predictor

An end-to-end Machine Learning project that predicts passenger survival probability using a client-side web interface. The underlying predictive engine is powered by a Logistic Regression model trained in Python, which is deployed natively using JavaScript coefficients.

## 🚀 Live Demo
You can view the deployed application here: `https://<your-username>.github.io/Titanic_survival_predictor/`

---

## 📋 Project Curriculum & Workflow

This project was built following a structured 5-step data science lifecycle framework:

### 1. Business Understanding & Problem Framing
* **Objective:** Predict whether a passenger would survive the Titanic disaster based on demographic and ticketing data.
* **Problem Type:** Binary Classification (Target: `Survived` where 0 = No, 1 = Yes).
* **Impact:** Demonstrates risk assessment modeling and operationalizing an offline model into a production-ready application.

### 2. Understanding the Titanic Dataset
The model leverages five core features from the canonical Titanic passenger manifest:
* `Pclass` (Passenger Class): 1st, 2nd, or 3rd as a proxy for socioeconomic status.
* `Sex`: Categorical gender metadata ("women and children first").
* `Age`: Continuous numerical age.
* `SibSp`: Number of siblings/spouses aboard.
* `Parch`: Number of parents/children aboard.

### 3. Data Preparation Pipeline with Implementation
The Python pipeline cleans and converts raw data into a machine-learnable format:
* **Missing Value Imputation:** Missing values in the `Age` feature are filled using the dataset's median age.
* **Categorical Encoding:** The `Sex` text string is mapped to binary metrics (`female: 0`, `male: 1`).
* **Data Splitting:** Data is split into an 80% training set and a 20% validation test set to evaluate performance.

### 4. Model Building & Overfitting Control
* **Algorithm:** Logistic Regression with L2 Regularization ($C=1.0$) to prevent overfitting.
* **Generalization Check:** The training and testing metrics were closely aligned, verifying that the model generalizes well to unseen data and does not suffer from high variance.

### 5. Model Evaluation & Final Recommendation
* **Training Accuracy:** ~80.20%
* **Testing Accuracy:** ~81.01%
* **Recommendation:** The mathematical parameters calculated during training were extracted to power the live web app deployment.

---

## 🛠️ Mathematical Implementation (Bridge to Frontend)

Rather than maintaining a heavy, expensive live Python server backend, the optimized mathematical weights derived from **Section 4** were ported directly into client-side JavaScript (`script.js`):

* **Intercept ($\beta_0$):** `4.8872`
* **Pclass Weight ($\beta_1$):** `-1.1895`
* **Sex(Male) Weight ($\beta_2$):** `-2.6052`
* **Age Weight ($\beta_3$):** `-0.0305`
* **SibSp Weight ($\beta_4$):** `-0.3547`
* **Parch Weight ($\beta_5$):** `-0.0784`

When a user submits details on the webpage, JavaScript computes the log-odds equation ($z$) and maps it through a **Sigmoid Activation Function** ($1 / (1 + e^{-z})$) to display an instant, real-time survival percentage.

---

## 📁 Repository Structure

```text
├── titanic_model.py       # Core data science pipeline script (Python)
├── index.html             # UI layout and submission form (HTML5)
├── style.css              # Custom styling sheet (CSS3)
└── script.js              # Live deployment script carrying model weights (JS)