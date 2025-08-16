# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from .inference import predict_bytes, MODEL_VERSION

import base64, binascii, json
from pathlib import Path

# -------- App --------
app = FastAPI(
    title="Vision API",
    version=MODEL_VERSION,  # o "1.0.0" si prefieres fijo
    description="Clasificador de insectos (Bee, Ant, Butterfly, Ladybug).",
)

# Raíz: redirige a Swagger
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=307)

@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Endpoints de predicción --------
class ImageB64(BaseModel):
    image_b64: str  # imagen en base64 (sin saltos de línea)

@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Imagen a clasificar")):
    try:
        content = await file.read()
        if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
            raise HTTPException(status_code=400, detail="Formato no soportado (usa JPEG o PNG).")
        return predict_bytes(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {e}")

@app.post("/predict_json")
def predict_json(body: ImageB64):
    try:
        content = base64.b64decode(body.image_b64.encode("utf-8"), validate=True)
        return predict_bytes(content)
    except binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"base64 inválido: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {e}")

# -------- Metadata (ruta segura al classes.json) --------
ROOT_DIR = Path(__file__).resolve().parent.parent  # .../ (raíz del repo)
CLASSES_PATH = ROOT_DIR / "models" / "classes.json"

@app.get("/metadata")
def metadata():
    data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    classes = [v for k, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    return {"model_version": MODEL_VERSION, "num_classes": len(classes), "classes": classes}
