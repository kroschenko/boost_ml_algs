import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Загрузка данных
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание базовой модели (в данном случае, дерева решений)
base_model = DecisionTreeRegressor(max_depth=3)

# Создание модели AdaBoostRegressor
adaboost_model = AdaBoostRegressor(base_model, n_estimators=50, learning_rate=0.1, random_state=42)

# Обучение модели
adaboost_model.fit(X_train, y_train)

# Прогнозирование на тестовом наборе
y_pred = adaboost_model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Визуализация важности признаков
feature_importance = adaboost_model.feature_importances_
feature_names = housing.feature_names
sorted_idx = np.argsort(feature_importance).astype(int)
plt.barh(range(len(feature_names)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_names)), np.array(feature_names)[sorted_idx])
plt.xlabel("Важность признака")
plt.title("Важность признаков в модели AdaBoostRegressor")
plt.show()
