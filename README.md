
# 🚀 No-Code Machine Learning Pipeline

This project is a **Streamlit-based web application** that allows users to build, train, evaluate, and save machine learning models **without writing a single line of code**. It provides a step-by-step visual interface for dataset handling, model training, and evaluation.

---

## 🧰 Features

- Upload any **CSV dataset**
- Explore data visually (histograms, pie charts, scatter plots, heatmaps)
- Clean data (handle missing values, drop duplicates)
- Feature engineering (encoding, scaling, drop columns)
- Automatic **classification or regression task detection**
- Train-test-validation splitting with stratification
- Model selection from 15+ popular ML algorithms
- Customize hyperparameters via sliders
- Model training with **learning curve visualization**
- Model evaluation (metrics, ROC, confusion matrix, residual plots)
- Save and download trained models

---

## ⚙️ Technologies Used

| Library            | Purpose                          |
|--------------------|----------------------------------|
| `Streamlit`        | UI and workflow                  |
| `pandas`, `numpy`  | Data manipulation                |
| `matplotlib`, `seaborn`, `plotly` | Data visualization         |
| `scikit-learn`     | ML models, preprocessing         |
| `xgboost`, `lightgbm` | Advanced boosting algorithms     |
| `joblib`           | Model saving and loading         |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/no_code_ml_pipeline.git
cd no_code_ml_pipeline
```

### 2. Install dependencies
Ensure you have Python 3.7+ and install required libraries:
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn plotly xgboost lightgbm joblib
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📷 Screenshots

| Data Upload | Data Visualization |
|-------------|--------------------|
| ![upload](assets/upload.png) | ![viz](assets/visualization.png) |

| Model Training | Evaluation |
|----------------|------------|
| ![train](assets/training.png) | ![eval](assets/evaluation.png) |

---

## 📝 Project Structure

```plaintext
📂 no_code_ml_pipeline
├── app.py                             # Main Streamlit app
├── data/                              # Uploaded datasets
├── models/                            # Saved models
├── assets/                            # Screenshots/images
├── README.md
├── requirements.txt
```

---

## 💡 Future Enhancements

- Support for NLP and image data
- Advanced model tuning (GridSearchCV)
- Model explainability (SHAP, LIME)
- Deployment on platforms like Heroku, Streamlit Cloud

---

## 🤝 Contributing

Contributions are welcome! Fork the repo, make your changes, and create a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- Streamlit Team
- scikit-learn Contributors
- Open-source ML community
