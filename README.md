# Cardiovascular Disease Clinical Decision Support System (CVD CDSS)

An explainable AI (XAI) clinical screening application equipped with HL7 FHIR interoperability, HIPAA audit logging, and dynamic visual risk assessment.

---

## Key Features

*   **Machine Learning Prediction**: Supervised XGBoost classifier predicting cardiovascular risk based on 5 optimized clinical observations.
*   **Explainable AI (XAI)**: Generates feature attributions (SHAP values) color-coded as crimson (risk-enhancing) or emerald (risk-reducing) inside interactive browser-based Plotly.js charts.
*   **HL7 FHIR Interoperability**: Integrates a client that queries mock FHIR patient resources and auto-populates demographics and clinical observations with a single click.
*   **HIPAA Audit Logging & RBAC**: Implements secure logins using Bcrypt encryption and Flask-Login. Separates Clinician activities from Administrator roles, writing compliance entries to a secure database log for every sensitive query, download, or access attempt.
*   **Clinical Guidelines Rule Engine**: Evaluates patient observations against standard ACC/AHA cardiology guidelines, rendering advisory alerts alongside risk predictions.
*   **ReportLab PDF Reporting**: Compiles all patient demographics, clinical entries, guidelines advice, and the explainable AI chart into a print-ready, double-column PDF report.
*   **Model Retraining Pipeline**: Includes a command-line script (`train.py`) to run data loading, encoding, scaling, ensemble feature selection ranking, and decision threshold tuning automatically.

---

## Technical Stack
*   **Backend**: Flask (Python), SQLAlchemy (SQLite ORM)
*   **Authentication**: Flask-Login, Bcrypt
*   **ML & XAI**: Scikit-Learn, XGBoost, Joblib, SHAP, NumPy, Pandas
*   **Visualizations**: Plotly.js, Matplotlib (Agg backend)
*   **PDF Compiler**: ReportLab

---

## Quick Start Setup & Installation

### 1. Prerequisites
Ensure Python 3.8+ is installed on your local machine.

### 2. Project Setup
Open Visual Studio Code, load this folder, and open the integrated terminal (`Ctrl + ~`):

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```
2.  **Activate the Virtual Environment**:
    *   **PowerShell (Windows default)**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **Command Prompt (CMD)**:
        ```cmd
        .\venv\Scripts\activate.bat
        ```
3.  **Install Required Dependencies**:
    ```bash
    pip install flask flask-sqlalchemy flask-login bcrypt scikit-learn xgboost joblib numpy pandas matplotlib shap reportlab
    ```

---

## Running the Application

### 1. Optional: Retrain the Model
If you want to train the model from scratch on `heart.csv` (which automatically matches the versions of your local libraries and silences `InconsistentVersionWarning` logs):
```bash
python train.py
```

### 2. Start the Flask Server
Run the application launcher:
```bash
python app.py
```

Open your web browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## Testing Credentials & FHIR Profiles

### Default Database Users
*   **Clinician**: `doctor` / `doctor123` (Allowed to run predictions, fetch EHRs, and export PDFs).
*   **Administrator**: `admin` / `admin123` (Allowed to access the HIPAA audit log dashboard).

### Mock FHIR IDs (For Instant EHR Pre-population)
Type either ID into the "Fetch from EHR" search box and click **Fetch**:
*   `fhir-1` (Robert Davis - Male, High Cardiovascular risk metrics)
*   `fhir-2` (Alice Smith - Female, Low/Stable metrics)

---

## Project Structure
*   `app.py`: Serves the web endpoints, REST APIs, and database session bindings.
*   `train.py`: Standalone script to execute preprocessing, model training, and asset serialization.
*   `inspect_pickles.py`: Helper script to unpickle and print model and scaler metadata in the console.
*   `heart.csv`: Baseline clinical dataset used to train the classifier.
*   `cvd_analysis.ipynb`: Research notebook demonstrating exploratory data analysis (EDA).
