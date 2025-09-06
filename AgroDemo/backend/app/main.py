from fastapi import FastAPI, Depends, HTTPException
from app.auth import verify_token
from app.database import init_db, SessionLocal
from app.models_db import Prediccion
from pydantic import BaseModel
from typing import List, Optional
import uuid

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
