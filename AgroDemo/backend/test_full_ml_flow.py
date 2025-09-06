# test_full_ml_flow.py
"""
Flujo ML completo:
1. Entrena el modelo de siembra
2. Lo guarda en DB + archivo
3. Lo recarga desde DB
4. Hace una predicción de prueba
"""

import numpy as np
import pandas as pd
from app.ml_pipeline import MLPipeline
from app.ml_engine import ModelManager

# 1. Datos de ejemplo (14 features porque tu pipeline los espera)
X = pd.DataFrame(np.random.rand(100, 14))
y = np.random.randint(250, 300, size=100)

print("Entrenando modelo de siembra...")
pipeline = MLPipeline()
metrics = pipeline.train_siembra(X, y)
print("✅ Métricas de entrenamiento:", metrics)

# 2. Recargar el modelo desde DB
print("\nRecargando modelo desde DB...")
manager = ModelManager()
loaded_model = manager.get_model("siembra", version="1.0")
print("✅ Modelo recargado:", type(loaded_model))

# 3. Hacer predicción de prueba
print("\nRealizando predicción de prueba...")
X_test = np.random.rand(1, 14)  # 1 fila con 14 features
pred = loaded_model.predict(X_test)
print(f"✅ Predicción para {X_test.tolist()}: {pred.tolist()}")

print("\n🚀 Flujo ML completo ejecutado correctamente.")
