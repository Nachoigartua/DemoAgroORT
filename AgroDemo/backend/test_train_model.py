from app.ml_pipeline import MLPipeline
import pandas as pd
import numpy as np

# Datos de ejemplo para entrenamiento de modelo de siembra
X = pd.DataFrame(np.random.rand(100, 14))
y = np.random.randint(250, 300, size=100)

pipeline = MLPipeline()
metrics = pipeline.train_siembra(X, y)
print("Métricas de entrenamiento siembra:", metrics)
