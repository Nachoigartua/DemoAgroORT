from sqlalchemy import Column, String, Float, Date, TIMESTAMP, Boolean, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, BYTEA
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()

class Prediccion(Base):
    __tablename__ = 'predicciones'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lote_id = Column(UUID(as_uuid=True))
    tipo_prediccion = Column(String(50))
    resultado = Column(JSON)
    fecha_creacion = Column(TIMESTAMP)

class ClimaHistorico(Base):
    __tablename__ = 'clima_historico'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    latitud = Column(Float)
    longitud = Column(Float)
    fecha = Column(Date)
    temp_max = Column(Float)
    temp_min = Column(Float)
    precipitacion = Column(Float)

class CaracteristicaSuelo(Base):
    __tablename__ = 'caracteristicas_suelo'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lote_id = Column(UUID(as_uuid=True))
    profundidad_cm = Column(Float)
    ph = Column(Float)
    materia_organica = Column(Float)
    nitrogeno = Column(Float)
    fosforo = Column(Float)
    potasio = Column(Float)
    textura = Column(String(50))
    capacidad_campo = Column(Float)

class ModeloML(Base):
    __tablename__ = 'modelos_ml'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nombre = Column(String(100))
    version = Column(String(20))
    tipo_modelo = Column(String(50))
    archivo_modelo = Column(BYTEA)
    metricas_performance = Column(JSON)
    fecha_entrenamiento = Column(TIMESTAMP)
    activo = Column(Boolean, default=True)
