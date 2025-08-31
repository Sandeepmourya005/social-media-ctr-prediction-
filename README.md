# Social Media Ad CTR Prediction

A concise, reproducible scikit-learn project that benchmarks multiple classifiers for click-through-rate (CTR) prediction,
performs GridSearchCV with cross-validation, selects the best model (often linear SVM on the provided sample), and applies SHAP for explainability.

## Quickstart
```bash
python -m venv .venv
source ./.venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/generate_sample_data.py --n 1500 --seed 42
python src/train.py --data data/sample_ads.csv --outdir outputs --cv 5 --seed 42
python src/explain_shap.py --data data/sample_ads.csv --model outputs/best_model.joblib --outdir outputs --seed 42
```
Artifacts: `outputs/` (cv_results.csv, best_params.json, test_metrics.json, confusion_matrix.png, best_model.joblib, shap plots).
