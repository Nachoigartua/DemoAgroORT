from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class SiembraRequest(BaseModel):
    lote_id: str
    cliente_id: str
    cultivo: str
    campana: str
    fecha_consulta: datetime

class RecomendacionResponse(BaseModel):
    lote_id: str
    tipo_recomendacion: str
    recomendacion_principal: dict
    alternativas: List[dict]
    nivel_confianza: float
    factores_considerados: List[str]
    costos_estimados: dict
    fecha_generacion: datetime
