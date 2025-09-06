import joblib
import os
from app.models_db import ModeloML
from app.database import SessionLocal

MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

class ModelManager:
    def __init__(self):
        self.db = SessionLocal()

    def get_model(self, nombre, version=None):
        query = self.db.query(ModeloML).filter_by(nombre=nombre, activo=True)
        if version:
            query = query.filter_by(version=version)
        model_entry = query.first()
        if not model_entry:
            return None
        model_file = os.path.join(MODELS_PATH, f"{model_entry.nombre}_{model_entry.version}.pkl")
        if not os.path.exists(model_file):
            with open(model_file, "wb") as f:
                f.write(model_entry.archivo_modelo)
        return joblib.load(model_file)

    def save_model(self, nombre, version, tipo_modelo, model_obj, metricas, fecha_entrenamiento):
        model_file = os.path.join(MODELS_PATH, f"{nombre}_{version}.pkl")
        joblib.dump(model_obj, model_file)
        with open(model_file, "rb") as f:
            archivo_modelo = f.read()
        nuevo = ModeloML(
            nombre=nombre,
            version=version,
            tipo_modelo=tipo_modelo,
            archivo_modelo=archivo_modelo,
            metricas_performance=metricas,
            fecha_entrenamiento=fecha_entrenamiento,
            activo=True
        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.close()

    def deactivate_model(self, nombre, version):
        model_entry = self.db.query(ModeloML).filter_by(nombre=nombre, version=version).first()
        if model_entry:
            model_entry.activo = False
            self.db.commit()
        self.db.close()
