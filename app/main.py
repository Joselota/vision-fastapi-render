# app/main.py
from fastapi import Request, FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from .inference import predict_bytes, MODEL_VERSION
from pydantic import BaseModel
import base64, binascii


class ImageB64(BaseModel):
    image_b64: str  # imagen en base64 (sin saltos de línea)

app = FastAPI(
    title="Vision API",
    version="1.0.0",            
    description="Clasificador de insectos (Bee, Ant, Butterfly, Ladybug).",
    docs_url="/docs",               
    redoc_url=None
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", include_in_schema=False)
def root():
    # Redirige la raíz a Swagger
    return RedirectResponse(url="/docs", status_code=307)

@app.get("/health")
def health():
    return {"status": "ok"}

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
        return predict_bytes(content)  # reutilizamos tu lógica de inferencia
    except binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"base64 inválido: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {e}")
    
@app.get("/metadata")
def metadata():
    d = json.loads(open("models/classes.json","r",encoding="utf-8").read())
    classes = [v for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))]
    return {"model_version": MODEL_VERSION, "num_classes": len(classes), "classes": classes}
