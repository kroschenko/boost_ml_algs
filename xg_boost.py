# Импорт необходимых библиотек
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Загрузка данных (в данном случае используем встроенный набор данных по жилью в Бостоне)
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["target"] = housing.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42)

# Инициализация и обучение модели XGBoost
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
