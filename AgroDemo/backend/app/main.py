from fastapi import FastAPI, Depends, HTTPException, Request
from app.auth import verify_token
from app.database import init_db, SessionLocal
from app.models_db import Prediccion
from app.ml_engine import ModelManager
from app.ml_pipeline import MLPipeline
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/protected")
async def protected(user=Depends(verify_token)):
    return {"message": "Acceso autorizado", "user": user}

# --- Endpoints para predicciones ---
class PrediccionCreate(BaseModel):
    lote_id: str
    tipo_prediccion: str
    resultado: dict

class PrediccionOut(BaseModel):
    id: str
    lote_id: str
    tipo_prediccion: str
    resultado: dict
    fecha_creacion: Optional[str]

@app.post("/api/v1/predicciones", response_model=PrediccionOut)
def crear_prediccion(prediccion: PrediccionCreate):
    db = SessionLocal()
    nueva = Prediccion(
        id=uuid.uuid4(),
        lote_id=uuid.UUID(prediccion.lote_id),
        tipo_prediccion=prediccion.tipo_prediccion,
        resultado=prediccion.resultado
    )
    db.add(nueva)
    db.commit()
    db.refresh(nueva)
    db.close()
    return PrediccionOut(
        id=str(nueva.id),
        lote_id=str(nueva.lote_id),
        tipo_prediccion=nueva.tipo_prediccion,
        resultado=nueva.resultado,
        fecha_creacion=str(nueva.fecha_creacion)
    )

@app.get("/api/v1/predicciones", response_model=List[PrediccionOut])
def listar_predicciones():
    db = SessionLocal()
    predicciones = db.query(Prediccion).order_by(Prediccion.fecha_creacion.desc()).limit(20).all()
    db.close()
    return [
        PrediccionOut(
            id=str(p.id),
            lote_id=str(p.lote_id),
            tipo_prediccion=p.tipo_prediccion,
            resultado=p.resultado,
            fecha_creacion=str(p.fecha_creacion)
        ) for p in predicciones
    ]

# --- Modelos para los módulos principales ---
class SiembraRequest(BaseModel):
    lote_id: str
    cliente_id: str
    cultivo: str
    campana: str
    fecha_consulta: datetime

class RecomendacionSiembraResponse(BaseModel):
    lote_id: str
    cultivo: str
    recomendacion_principal: dict
    alternativas: List[dict]

@app.post("/api/v1/recomendaciones/siembra", response_model=RecomendacionSiembraResponse)
async def get_recomendaciones_siembra(request: SiembraRequest):
    # Conexión con modelo ML real
    model_manager = ModelManager()
    model = model_manager.get_model("siembra", version="1.0")
    print("Modelo cargado:", model)
    # Simulación de features SOLO numéricas (strings codificados manualmente)
    features = [
        34.5, -58.4, 22.0, 20.0, 18.0,
        100.0, 90.0, 80.0,
        1.0,          # "franco" → 1
        6.5, 3.2,
        2.0,          # "maiz" → 2
        4500.0, 260.0
    ]
    print("Features shape:", len(features))
    if not model:
        raise HTTPException(status_code=500, detail="Modelo de siembra no disponible")
    try:
        pred = model.predict([features])[0]
    except Exception as e:
        print("Error en predicción:", e)
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")
    confianza = 0.87  # En producción, usar output del modelo/probabilidad
    justificacion = "Basado en clima y suelo histórico"
    factores = ["Clima", "Suelo", "Cultivo anterior"]
    return {
        "lote_id": request.lote_id,
        "cultivo": request.cultivo,
        "recomendacion_principal": {
            "fecha_optima": str(pred),
            "ventana": [str(pred-2), str(pred+2)],
            "confianza": confianza,
            "justificacion": justificacion,
            "factores": factores
        },
        "alternativas": [
            {
                "fecha": str(pred+7),
                "pros": ["Mayor humedad esperada"],
                "contras": ["Riesgo de heladas tardías"],
                "confianza": 0.72
            }
        ]
    }

class VariedadRequest(BaseModel):
    lote_id: str
    cliente_id: str
    cultivo: str
    campana: str

class RecomendacionVariedadResponse(BaseModel):
    lote_id: str
    variedad_principal: str
    alternativas: List[dict]

@app.post("/api/v1/recomendaciones/variedades", response_model=RecomendacionVariedadResponse)
async def get_recomendaciones_variedades(request: VariedadRequest):
    # Lógica mock
    return {
        "lote_id": request.lote_id,
        "variedad_principal": "Variedad A",
        "alternativas": [
            {"variedad": "Variedad B", "pros": ["Resistente a plagas"], "contras": ["Menor rendimiento"]}
        ]
    }

@app.post("/api/v1/ml/retrain/siembra")
async def retrain_siembra():
    """Retrain modelo de siembra y retorna métricas."""
    # Simulación de datos de entrenamiento (en producción, cargar de DB)
    import numpy as np
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    pipeline = MLPipeline()
    metrics = pipeline.train_siembra(X, y)
    return {"mensaje": "Modelo siembra reentrenado", "metricas": metrics}

@app.post("/api/v1/ml/retrain/rendimiento")
async def retrain_rendimiento():
    """Retrain modelo de rendimiento y retorna métricas."""
    import numpy as np
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    pipeline = MLPipeline()
    metrics = pipeline.train_rendimiento(X, y)
    return {"mensaje": "Modelo rendimiento reentrenado", "metricas": metrics}

class ClimaRequest(BaseModel):
    latitud: float
    longitud: float
    fecha: datetime

class ClimaResponse(BaseModel):
    predicciones: dict
    alertas: List[str]

@app.post("/api/v1/predicciones/clima", response_model=ClimaResponse)
async def get_predicciones_clima(request: ClimaRequest):
    # Lógica mock
    return {
        "predicciones": {"precipitacion": 20.5, "temperatura_max": 28, "temperatura_min": 12},
        "alertas": ["Posible evento extremo"]
    }

class FertilizacionRequest(BaseModel):
    lote_id: str
    cultivo: str
    campana: str

class FertilizacionResponse(BaseModel):
    plan_principal: dict
    alternativas: List[dict]

@app.post("/api/v1/recomendaciones/fertilizacion", response_model=FertilizacionResponse)
async def get_recomendaciones_fertilizacion(request: FertilizacionRequest):
    # Lógica mock
    return {
        "plan_principal": {"producto": "Urea", "dosis": 120, "momento": "Pre-siembra"},
        "alternativas": [
            {"producto": "Fosfato", "dosis": 80, "momento": "Emergencia"}
        ]
    }

class RendimientoRequest(BaseModel):
    lote_id: str
    cultivo: str
    campana: str

class RendimientoResponse(BaseModel):
    prediccion: float
    rango_confianza: List[float]
    factores_influyentes: List[str]

@app.post("/api/v1/predicciones/rendimiento", response_model=RendimientoResponse)
async def get_predicciones_rendimiento(request: RendimientoRequest):
    # Lógica mock
    return {
        "prediccion": 4500.0,
        "rango_confianza": [4000.0, 5000.0],
        "factores_influyentes": ["Clima", "Fertilización", "Suelo"]
    }

class CosechaRequest(BaseModel):
    lote_id: str
    cultivo: str
    campana: str

class CosechaResponse(BaseModel):
    fecha_optima: str
    ventana_recomendada: List[str]
    calidad_predicha: str

@app.post("/api/v1/recomendaciones/cosecha", response_model=CosechaResponse)
async def get_recomendaciones_cosecha(request: CosechaRequest):
    # Lógica mock
    return {
        "fecha_optima": "2025-12-01",
        "ventana_recomendada": ["2025-11-28", "2025-12-05"],
        "calidad_predicha": "Excelente"
    }
