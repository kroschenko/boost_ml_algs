# Импортируем необходимые библиотеки
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing

# Загружаем данные
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем регрессор CatBoost
model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, loss_function='RMSE', random_seed=42)

# Обучаем модель на обучающем наборе
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=50)

# Делаем прогнозы на тестовом наборе
y_pred = model.predict(X_test)

# Оцениваем качество модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Можно также посмотреть на важность признаков
feature_importance = model.get_feature_importance()
feature_names = housing.feature_names
for feature, importance in zip(feature_names, feature_importance):
    print(f"{feature}: {importance}")

# Пример прогноза для новых данных
new_data = np.array([[0.03, 0.0, 13.0, 0.0, 0.4, 6.0, 29.6, 3.8, 5.0, 264.0, 13.0, 390.0, 3.8]])
prediction = model.predict(new_data)
print(f"Прогноз для новых данных: {prediction}")
