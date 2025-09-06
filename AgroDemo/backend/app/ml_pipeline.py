import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from app.ml_engine import ModelManager
from datetime import datetime

class MLPipeline:
    def __init__(self):
        self.model_manager = ModelManager()

    def train_siembra(self, X, y):
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }
        self.model_manager.save_model(
            nombre="siembra",
            version="1.0",
            tipo_modelo="RandomForestRegressor",
            model_obj=model,
            metricas=metrics,
            fecha_entrenamiento=datetime.now()
        )
        return metrics

    def train_variedades(self, X, y):
        model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="weighted")
        }
        self.model_manager.save_model(
            nombre="variedades",
            version="1.0",
            tipo_modelo="XGBClassifier",
            model_obj=model,
            metricas=metrics,
            fecha_entrenamiento=datetime.now()
        )
        return metrics

    def train_rendimiento(self, X, y):
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "rmse": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }
        self.model_manager.save_model(
            nombre="rendimiento",
            version="1.0",
            tipo_modelo="GradientBoostingRegressor",
            model_obj=model,
            metricas=metrics,
            fecha_entrenamiento=datetime.now()
        )
        return metrics

    def retrain_all(self, datasets):
        results = {}
        if "siembra" in datasets:
            results["siembra"] = self.train_siembra(datasets["siembra"]["X"], datasets["siembra"]["y"])
        if "variedades" in datasets:
            results["variedades"] = self.train_variedades(datasets["variedades"]["X"], datasets["variedades"]["y"])
        if "rendimiento" in datasets:
            results["rendimiento"] = self.train_rendimiento(datasets["rendimiento"]["X"], datasets["rendimiento"]["y"])
        return results
