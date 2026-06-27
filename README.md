# 🌾 Crop Yield Prediction and Crop Recommendation System using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Project Overview

Agriculture is one of the most important sectors of the economy. Predicting crop yield accurately helps farmers, researchers, and policymakers make better decisions regarding crop planning and resource management.

This project uses **Machine Learning** techniques to predict crop yield based on agricultural data such as:

- State
- District
- Crop
- Season
- Area (Hectares)

Additionally, the project includes a **Crop Recommendation System** that recommends suitable crops based on soil and environmental parameters.

The application is developed using **Python**, **Scikit-Learn**, and **Streamlit**, providing an interactive dashboard for real-time predictions.

---

# ✨ Features

- 🌾 Crop Yield Prediction
- 🌱 Crop Recommendation
- 📊 Interactive Streamlit Dashboard
- 🤖 Machine Learning-based Prediction
- 📈 Multiple Regression Algorithms
- 📋 Easy User Interface
- 💾 Saved ML Models
- 📂 CSV Dataset Support

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| XGBoost | Gradient Boosting |
| Joblib | Model Saving |
| Streamlit | Web Dashboard |
| Matplotlib | Data Visualization |

---

# 📂 Project Structure

```
crop-yield-pro/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── crop_production.csv
│   └── Crop_recommendation.csv
│
├── models/
│   ├── best_model.pkl
│   └── crop_recommendation.pkl
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── train_recommendation.py
│   └── predict_recommendation.py
│
└── README.md
```

---

# 📊 Machine Learning Algorithms

The project compares multiple regression algorithms.

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor ⭐
- XGBoost Regressor

The best-performing model is automatically selected and saved as:

```
models/best_model.pkl
```

---

# 📈 Dataset

The project uses two datasets:

### Crop Production Dataset

Features:

- State Name
- District Name
- Crop Year
- Season
- Crop
- Area
- Production

### Crop Recommendation Dataset

Features:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall
- Crop

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/prathameshranadive07/crop-yield-pro.git
```

Move into the project folder

```bash
cd crop-yield-pro
```

Create a virtual environment

```bash
python3 -m venv venv
```

Activate the environment

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python src/train.py
```

Train Crop Recommendation Model

```bash
python src/train_recommendation.py
```

---

# 🌐 Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open at:

```
http://localhost:8501
```

---

# 📷 Application Screenshots

> Add screenshots here after deployment.

Example:

```
screenshots/
│
├── home.png
├── prediction.png
├── recommendation.png
└── dashboard.png
```

---

# 📊 Model Evaluation

Performance Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The model with the highest R² score is selected as the final prediction model.

---

# 🔮 Future Enhancements

- 🌦 Live Weather API Integration
- 📍 GPS-Based Location Detection
- ☁ Cloud Deployment
- 📱 Mobile Application
- 💰 Crop Price Prediction
- 🌱 Fertilizer Recommendation
- 🛰 Satellite Image Integration
- 🤖 Deep Learning Models

---

# 👨‍💻 Author

**Group No 36**

Bachelor of Engineering (Computer Engineering)

Final Year Project

GitHub:
https://github.com/prathameshranadive07

---

# 📜 License

This project is licensed under the MIT License.

---

## ⭐ If you found this project useful, please consider giving it a Star on GitHub!
