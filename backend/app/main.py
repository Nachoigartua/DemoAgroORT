from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .security import check_api_key, rate_limit
from .models import SiembraRequest, VariedadRequest, ClimaRequest, FertilizacionRequest, AgroquimicosRequest, RendimientoRequest, CosechaRequest, AplicarRequest
from .cache import make_cache_key, cache_get, cache_set, TTL_BY_TYPE
from .metrics import metrics
from . import db
from .predictors.siembra import SiembraPredictor
from .predictors.rendimiento import RendimientoPredictor
from .predictors.variedades import VariedadesPredictor
import time, json

app = FastAPI(title="ML Agrícola API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

def auth_and_limit(request: Request, x_api_key: str | None):
    check_api_key(x_api_key)
    rate_limit(request)

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/metrics")
def metrics_endpoint(): return metrics.snapshot()

@app.get("/")
def root(): return {"service":"ml-agrico-backend","version":"0.2.0"}

@app.get("/api/v1/catalogo/lotes")
def catalogo_lotes(x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    conn = db.get_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,nombre,area_ha,cultivo,latitud,longitud FROM lotes ORDER BY id")
            rows = cur.fetchall()
    conn.close()
    return {"items": rows}

@app.get("/api/v1/catalogo/campanas")
def catalogo_campanas(x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    return {"items":["2024-25","2025-26"]}

@app.get("/api/v1/catalogo/analisis")
def catalogo_analisis(x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    return {"items":["siembra","variedades","clima","fertilizacion","rendimiento","cosecha"]}

@app.post("/api/v1/recomendaciones/siembra")
def recomendaciones_siembra(req: SiembraRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    payload = req.model_dump()
    key = make_cache_key("siembra", payload)
    if (c:=cache_get(key)): return c
    start = time.time()
    pred = SiembraPredictor().predict(payload)
    metrics.track("siembra", time.time()-start)
    result = {
        "tipo":"siembra","lote_id":req.lote_id,"cultivo":req.cultivo,
        "fecha_validez_desde": pred["recomendacion_principal"]["ventana"][0],
        "fecha_validez_hasta": pred["recomendacion_principal"]["ventana"][1],
        "recomendacion_principal": pred["recomendacion_principal"],
        "alternativas": pred["alternativas"],
        "nivel_confianza": pred["recomendacion_principal"]["confianza"],
        "modelo_version": SiembraPredictor.MODEL_VERSION
    }
    cache_set(key,result,TTL_BY_TYPE["siembra"])
    return result

@app.post("/api/v1/recomendaciones/variedades")
def recomendaciones_variedades(req: VariedadRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    payload = req.model_dump()
    key = make_cache_key("variedades", payload)
    if (c:=cache_get(key)): return c
    start = time.time()
    pred = VariedadesPredictor().predict(payload)
    metrics.track("variedades", time.time()-start)
    result = {"tipo":"variedades","cultivo":req.cultivo,
              "recomendacion_principal": pred["recomendacion_principal"],
              "alternativas": pred["alternativas"],
              "modelo_version": VariedadesPredictor.MODEL_VERSION}
    cache_set(key,result,TTL_BY_TYPE["variedades"])
    return result

@app.post("/api/v1/predicciones/clima")
def prediccion_clima(req: ClimaRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    conn = db.get_conn()
    lat = req.coords.latitud; lon = req.coords.longitud
    with conn:
        with conn.cursor() as cur:
            cur.execute(\"\"\"\
                SELECT
                  AVG((temperatura_max+temperatura_min)/2.0) AS temp_media,
                  AVG(precipitacion) AS precip,
                  AVG(humedad_relativa) AS humedad,
                  AVG(velocidad_viento) AS viento,
                  AVG(radiacion_solar) AS radiacion
                FROM clima_historico
                WHERE round(latitud::numeric,2)=round(%s::numeric,2)
                  AND round(longitud::numeric,2)=round(%s::numeric,2)
                  AND fecha >= CURRENT_DATE - INTERVAL '7 days'
            \"\"\", (lat, lon))
            row = cur.fetchone()
    conn.close()
    if not row or row["temp_media"] is None:
        return {"tipo":"clima","periodo":req.periodo,"coords":req.coords,
                "mensual":{"precipitacion_mm":80.0,"temp_media_c":18.0,"humedad":65.0,"viento_kmh":12.0,"horas_sol":8.0}}
    horas_sol = float(row["radiacion"] or 6.0)
    return {"tipo":"clima","periodo":req.periodo,"coords":req.coords,
            "mensual":{"precipitacion_mm":round(float(row["precip"] or 0),1),
                       "temp_media_c":round(float(row["temp_media"] or 0),1),
                       "humedad":round(float(row["humedad"] or 0),1),
                       "viento_kmh":round(float(row["viento"] or 0),1),
                       "horas_sol":round(horas_sol,1)}}

@app.post("/api/v1/optimizacion/fertilizacion")
def optimizacion_fertilizacion(req: FertilizacionRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    conn = db.get_conn()
    mo = 2.0
    with conn:
        with conn.cursor() as cur:
            if req.lote_id:
                cur.execute("SELECT materia_organica FROM caracteristicas_suelo WHERE lote_id=%s ORDER BY id DESC LIMIT 1", (req.lote_id,))
                r = cur.fetchone()
                if r and r.get("materia_organica"): mo = float(r["materia_organica"])
    conn.close()
    base = {"trigo": 120, "maiz": 140, "soja": 30, "cebada": 100}.get(req.cultivo, 100)
    if mo > 2.5: base = int(base * 0.9)
    if (req.objetivo or "") == "alto": base += 10
    plan = {"producto":"Urea","dosis_kg_ha": base, "aplicaciones":[{"momento":"pre-siembra","porcentaje":70},{"momento":"macollaje","porcentaje":30}]}
    alts = [{"producto":"DAP","dosis_kg_ha": int(base*0.8)}, {"producto":"MAP","dosis_kg_ha": int(base*0.85)}]
    costos = {"semillas":85,"fertilizacion":round(base*1.2,0),"proteccion":45}
    costos["total"]=costos["semillas"]+costos["fertilizacion"]+costos["proteccion"]
    return {"plan_principal": plan, "alternativas": alts, "costos": costos}

@app.post("/api/v1/predicciones/rendimiento")
def prediccion_rendimiento(req: RendimientoRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    conn = db.get_conn()
    mo=2.0; temp=18.0; pcp=80.0; lat=0.0; lon=0.0
    with conn:
        with conn.cursor() as cur:
            if req.lote_id:
                cur.execute("SELECT latitud,longitud FROM lotes WHERE id=%s", (req.lote_id,))
                lr = cur.fetchone()
                if lr: lat,lon = lr["latitud"], lr["longitud"]
                cur.execute("SELECT materia_organica FROM caracteristicas_suelo WHERE lote_id=%s ORDER BY id DESC LIMIT 1", (req.lote_id,))
                sr = cur.fetchone()
                if sr and sr.get("materia_organica"): mo = float(sr["materia_organica"])
            cur.execute(\"\"\"\
                SELECT
                  AVG((temperatura_max+temperatura_min)/2.0) AS temp_media,
                  AVG(precipitacion) AS precip
                FROM clima_historico
                WHERE round(latitud::numeric,2)=round(%s::numeric,2)
                  AND round(longitud::numeric,2)=round(%s::numeric,2)
                  AND fecha >= CURRENT_DATE - INTERVAL '7 days'
            \"\"\", (lat, lon))
            cr = cur.fetchone()
            if cr:
                if cr.get("temp_media") is not None: temp = float(cr["temp_media"])
                if cr.get("precip") is not None: pcp = float(cr["precip"])
    conn.close()
    start = time.time()
    pred = RendimientoPredictor().predict(req.model_dump(), suelo_mo=mo, temp_media=temp, pp_mm=pcp)
    metrics.track("rendimiento", time.time()-start)
    return {"tipo":"rendimiento","lote_id":req.lote_id,"cultivo":req.cultivo,"resultado":pred}

@app.post("/api/v1/manejo/agroquimicos")
def manejo_agroquimicos(req: AgroquimicosRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    return {"cronograma":[{"semana":1,"producto":"Fungicida X"},{"semana":3,"producto":"Insecticida Y"}],
            "condiciones_ideales":["Viento < 15 km/h","Sin lluvia 24h"]}

@app.post("/api/v1/optimizacion/cosecha")
def optimizacion_cosecha(req: CosechaRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    vent = {"trigo":["2025-01-10","2025-01-20"],"maiz":["2025-03-01","2025-03-15"],"soja":["2025-04-10","2025-04-25"],"cebada":["2024-12-01","2024-12-15"]}.get(req.cultivo, ["2025-01-10","2025-01-20"])
    return {"ventana_recomendada": vent, "riesgos": ["lluvias tardías"]}

@app.post("/api/v1/acciones/aplicar")
def aplicar_recomendacion(req: AplicarRequest, x_api_key: str | None = Header(default=None), request: Request = None):
    auth_and_limit(request, x_api_key)
    return {"status":"ok"}
