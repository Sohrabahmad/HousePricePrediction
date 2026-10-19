# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview

This project uses **Linear Regression**, a supervised machine learning algorithm, to predict house sale prices based on property features such as lot area, building type, construction year, and basement area.

The project demonstrates data cleaning, categorical feature encoding, train-test splitting, model training, and evaluation. It also compares model performance before and after applying feature scaling.

## 🎯 Objectives

- Explore and understand the housing dataset.
- Handle missing values and clean the data.
- Encode categorical variables using one-hot encoding.
- Train a Linear Regression model.
- Evaluate predictions using regression metrics.
- Apply StandardScaler and compare results.

## 📂 Dataset

The project uses `HousePricePrediction.csv`.

- **Original records:** 2,919
- **Original columns:** 13
- **Columns after preprocessing:** 34, including the target column

### Features

| Feature | Description |
|---|---|
| `Id` | Unique property identifier; removed during preprocessing |
| `MSSubClass` | Building class |
| `MSZoning` | General zoning classification |
| `LotArea` | Lot size in square feet |
| `LotConfig` | Lot configuration |
| `BldgType` | Type of dwelling |
| `OverallCond` | Overall condition rating |
| `YearBuilt` | Original construction year |
| `YearRemodAdd` | Remodel or addition year |
| `Exterior1st` | Exterior covering material |
| `BsmtFinSF2` | Type 2 finished basement area |
| `TotalBsmtSF` | Total basement area |
| `SalePrice` | Target variable representing the sale price |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## ⚙️ Project Workflow

1. Load the dataset using Pandas.
2. Explore the data using `head()`, `info()`, and `describe()`.
3. Remove the `Id` column.
4. Fill missing `SalePrice` values with the mean.
5. Drop remaining rows containing missing values.
6. Apply one-hot encoding to categorical columns.
7. Separate input features and target variable.
8. Split the dataset into training and testing sets.
9. Train a Linear Regression model.
10. Evaluate the model using regression metrics.
11. Apply StandardScaler and compare the results.

## 🧹 Data Preprocessing

### Handling Missing Values

The notebook fills missing target values with the mean sale price and drops rows containing other missing values.

```python
df.drop(['Id'], axis=1, inplace=True)

df['SalePrice'] = df['SalePrice'].fillna(
    df['SalePrice'].mean()
)

df = df.dropna()
```

### Encoding Categorical Variables

One-hot encoding converts categorical variables into numeric features.

```python
cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

df = pd.get_dummies(
    df,
    columns=cols,
    drop_first=True
)
```

## 🤖 Model Training

The dataset is divided into training and testing sets using an 80:20 split.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop(['SalePrice'], axis=1)
Y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

## 📊 Model Evaluation

The model is evaluated using:

- **R² Score:** Measures the proportion of target variance explained by the model.
- **MAE:** Mean Absolute Error, the average absolute prediction error.
- **RMSE:** Root Mean Squared Error, which penalizes larger errors more heavily.
- **MAPE:** Mean Absolute Percentage Error.

### Results

Using the notebook's preprocessing and train-test split:

| Metric | Linear Regression | Linear Regression with StandardScaler |
|---|---:|---:|
| R² Score | 0.3741 | 0.3741 |
| MAE | 30,829.94 | 30,829.94 |
| RMSE | 41,138.56 | 41,138.56 |
| MAPE | 0.1874 | 0.1874 |

The model achieves an R² score of approximately **0.374**, meaning it explains about 37.4% of the target variance on the test set under this preprocessing approach.

Feature scaling does not materially change the performance of ordinary Linear Regression on this dataset.

> **Important data-quality note:** The notebook fills missing `SalePrice` values with the mean before splitting the dataset. This can introduce misleading evaluation results because missing target values are not genuine observed sale prices. A more reliable approach is to exclude rows with missing `SalePrice` before training and evaluate only on records with known sale prices.

## 📁 Project Structure

```text
House-Price-Prediction/
├── house_price_predictor.ipynb
├── HousePricePrediction.csv
└── README.md
```

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd House-Price-Prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open `house_price_predictor.ipynb` and execute the cells sequentially. Ensure `HousePricePrediction.csv` is in the notebook's working directory.

## 📚 Learning Outcomes

- Data exploration and preprocessing
- Missing-value handling
- Categorical feature encoding
- Supervised machine learning
- Linear Regression
- Train-test splitting
- Regression evaluation metrics
- Feature scaling
- Model performance interpretation

## ⚠️ Disclaimer

This project is intended for educational purposes. Predictions are estimates based on the selected dataset and features, and should not be treated as professional property valuations.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (AI & ML)
