# Despliegue de un modelo de ML como API (FastAPI + TensorFlow)

Clasificador de insectos (Bee, Ant, Butterfly, Ladybug) expuesto como servicio web.
Incluye dos formas de inferencia (archivo y JSON/base64), healthcheck, metadatos del
modelo y un cliente externo para pruebas.

> **API pública en Render:**  
> https://vision-api-ml.onrender.com  
> **Documentación Swagger:**  
> https://vision-api-ml.onrender.com/docs

> ⚠️ Nota (plan gratuito de Render): la **primera** petición puede tardar unos
> segundos porque la instancia “despierta”. Recomendado llamar primero a
> `/health` y luego a `/predict`.


👀 Demo rápido
### Salud del servicio
curl -sS https://vision-api-ml.onrender.com/health

### Metadata del modelo (versión y clases)
curl -sS https://vision-api-ml.onrender.com/metadata

### Predicción (multipart/form-data)
curl -sS -X POST "https://vision-api-ml.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/ruta/a/imagen.jpg"

### Predicción (JSON: imagen en base64)
base64 -i /ruta/a/imagen.jpg | tr -d '\n' > img.b64

curl -sS -X POST "https://vision-api-ml.onrender.com/predict_json" \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\":\"$(cat img.b64)\"}"


## 1) Estructura del repositorio
```
.
├─ app/
│  ├─ main.py              # FastAPI (/health, /predict, /predict_json, /metadata)
│  └─ inference.py         # Carga del modelo y preprocesamiento/predicción
├─ models/
│  ├─ modelo.h5            # Modelo entrenado (TensorFlow/Keras)
│  └─ classes.json         # Mapeo índice → nombre de clase
├─ client.py               # Cliente externo con 3 requests de ejemplo
├─ requirements.txt        # Dependencias para Linux/Render
├─ runtime.txt             # Versión de Python para Render (p.ej. 3.11.9)
└─ README.md
```


## 2) Modelo

* Formato: models/modelo.h5 (Keras).
* Salida: vector de probabilidades sobre 4 clases (según models/classes.json):
  - Bee, Ant, Butterfly, Ladybug.
* Preprocesamiento: redimensionado a 224×224 y normalización acorde al entrenamiento (ya aplicado dentro de app/inference.py).

🚀 Endpoints
| Método | Ruta            | Descripción                                     |
| -----: | --------------- | ----------------------------------------------- |
|    GET | `/health`       | Healthcheck sencillo (`{"status":"ok"}`)        |
|    GET | `/metadata`     | Versión y clases del modelo                     |
|   POST | `/predict`      | Predicción vía **multipart/form-data** (`file`) |
|   POST | `/predict_json` | Predicción vía **JSON** (imagen en 
`base64`)    |


Esquemas de entrada

1. POST /predict (multipart/form-data)
Campo file: imagen JPEG/PNG.
2. POST /predict_json (application/json)
{
  "image_b64": "<cadena base64 sin saltos de línea>"
}

Esquema de salida (ambos endpoints)
{
  "label": "Ladybug",
  "proba": {
    "Bee": 0.0126,
    "Ant": 0.0116,
    "Butterfly": 0.0216,
    "Ladybug": 0.9549
  },
  "model_version": "1.0.0"
}

## 3) Instalación y ejecución local

Requisitos: Python 3.11 (recomendado 3.11.9). En Windows, usar WSL2 facilita TensorFlow.
### a) crear y activar entorno
python3.11 -m venv .venv
source .venv/bin/activate                  # en Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel

### b) instalar dependencias
pip install -r requirements.txt

### c) levantar servidor
uvicorn app.main:app --reload
 abrir: http://127.0.0.1:8000/docs



## 4) Endpoints
| Método | Ruta            | Descripción                                      |
| -----: | --------------- | ------------------------------------------------ |
|    GET | `/health`       | Healthcheck (`{"status":"ok"}`)                  |
|    GET | `/metadata`     | Versión y clases del modelo                      |
|   POST | `/predict`      | Predicción vía **multipart/form-data** (`file`)  |
|   POST | `/predict_json` | Predicción vía **JSON** con imagen en **base64** |

Esquema predict_json
{
  "image_b64": "<cadena base64 de la imagen>"
}

Respuesta típica: 
{
  "label": "Ant",
  "proba": {
    "Bee": 0.01,
    "Ant": 0.98,
    "Butterfly": 0.00,
    "Ladybug": 0.01
  },
  "model_version": "1.0.0"
}

## 5) Cómo probar (curl / Postman)

curl — multipart

curl -sS -X POST https://vision-api-ml.onrender.com/predict \
  -H "accept: application/json" \
  -F "file=@ruta/a/tu_imagen.jpg"

curl — JSON/base64 con Python estándar

python - <<'PY'
import base64, json, pathlib, urllib.request
img = "ruta/a/tu_imagen.jpg"
b64 = base64.b64encode(pathlib.Path(img).read_bytes()).decode()
req = urllib.request.Request(
    "https://vision-api-ml.onrender.com/predict_json",
    data=json.dumps({"image_b64": b64}).encode(),
    headers={"Content-Type": "application/json"}
)
print(urllib.request.urlopen(req, timeout=60).read().decode())
PY

Postman

1. POST https://vision-api-ml.onrender.com/predict
2. Body → form-data
   - Key: file (Type: File), Value: tu .jpg/.png
3. No agregues manualmente Content-Type; Postman lo pone.
4. Send → 200 con el JSON de predicción.


## 6) Cliente externo (client.py) — 3 peticiones

Hace 3 requests distintos (2× /predict + 1× /predict_json) y guarda un resumen en resultados_cliente.json.

    python client.py IMG1.jpg IMG2.jpg IMG3.jpg \
    --url https://vision-api-ml.onrender.com \
    --out resultados_cliente.json

Salida esperada (ejemplo real)
    --- Request #1 — /predict (multipart) ---
    Enviando: 632185ec7f925c3d.jpg | 104652 B | image/jpeg | sha16=747798eae00f2154
    Status: 200 | 14528.7 ms
    {
    "label": "Butterfly",
    "proba": { "Bee": 0.1013, "Ant": 0.3185, "Butterfly": 0.3660, "Ladybug": 0.2141 },
    "model_version": "1.0.0"
    }

    --- Request #2 — /predict (multipart) ---
    Enviando: 4827d395bbb69e38.jpg | 173863 B | image/jpeg | sha16=1b9959fd7982db36
    Status: 200 | 1532.9 ms
    {
    "label": "Ant",
    "proba": { "Bee": 0.0003, "Ant": 0.9981, "Butterfly": 0.0002, "Ladybug": 0.0014 },
    "model_version": "1.0.0"
    }

    --- Request #3 — /predict_json (JSON base64) ---
    Enviando: a41d46815d597df0.jpg | 299154 B | image/jpeg | sha16=53dfed114df10513
    (longitud base64: 398872 chars)
    Status: 200 | 1731.8 ms
    {
    "label": "Ladybug",
    "proba": { "Bee": 0.0304, "Ant": 0.2734, "Butterfly": 0.0036, "Ladybug": 0.6926 },
    "model_version": "1.0.0"
    }
    El archivo resultados_cliente.json queda en el directorio actual con el resumen de las 3 llamadas (inputs, tiempos, outputs).

    ### Cómo ejecutarlo
    python client.py IMG1.jpg IMG2.jpg IMG3.jpg \
    --url https://vision-api-ml.onrender.com \
    --out resultados_cliente.json
    Si pasas 1 o 2 imágenes, el script repite la primera para completar 3 solicitudes.

## 7) Despliegue en Render

* URL del servicio: https://vision-api-ml.onrender.com
* Documentación: https://vision-api-ml.onrender.com/docs

Config clave:
* Start command: gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT app.main:app --timeout 120
* Python: runtime.txt (p.ej. 3.11.9)
* Auto-Deploy: ON (o Manual → Deploy latest commit)

* Este repo incluye requirements.txt y runtime.txt (3.11.9).
* Comandos de Render:
   - Build: pip install -r requirements.txt
   - Start: gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT app.main:app --timeout 120

* Variables de entorno usadas por la app:
  - MODEL_PATH = models/modelo.h5
  - CLASSES_PATH = models/classes.json
  - MODEL_VERSION = 1.0.0

En plan free hay cold starts (la primera petición tras inactividad puede tardar algunos segundos).

## 8) Uso con Postman
    1) POST /predict (multipart/form-data)

    * Método POST, URL: https://vision-api-ml.onrender.com/predict
    * Pestaña Body → selecciona form-data.
    * En la tabla agrega una fila:
    - Key: file
    - Type: File (no Text)
    - Value: elige tu imagen .jpg/.png
    * No agregues Content-Type manualmente (Postman lo pone solo).
    * Envía.
    Respuesta esperada (ejemplo):
            {
            "label": "Ant",
            "proba": {
                "Bee": 0.0663,
                "Ant": 0.5208,
                "Butterfly": 0.2745,
                "Ladybug": 0.1384
            },
            "model_version": "1.0.0"
            }
    2) POST /predict_json (JSON con imagen en base64)

    * Método POST, URL: https://vision-api-ml.onrender.com/predict_json
    * Body → raw → selecciona JSON.
    * Pega un JSON como este (reemplaza el valor por tu base64 en una sola línea):
        { "image_b64": "AAA...TU_BASE64...BBB" }
    * En mac/linux puedes generar el base64 así:
        base64 -i /ruta/a/imagen.jpg | tr -d '\n' > img.b64
        Copia el contenido de img.b64 y pégalo en image_b64.
        Nota: Suele fallar cuando el base64 tiene saltos de línea; por eso el tr -d '\n'.


## 9) Validaciones y manejo de errores

* 400 – Formato no soportado: si el archivo no es JPG/PNG.
* 400 – base64 inválido: si el JSON trae base64 malformado.
* 422 – ValidationError: esquema de entrada incorrecto.
* 500 – Errores internos inesperados (ver logs).

Mensajes de error devuelven detail legible para facilitar el diagnóstico.

## 10) Notas técnicas
* TensorFlow/keras probados en Python 3.11 (Render usa runtime.txt para fijar versión).
* Si modelo.h5 superara 100 MB, usa Git LFS para subirlo.

## 11) Licencia
Proyecto académico — Magíster. Autor: Ingrid Solís González.
Este proyecto es educativo y forma parte de una tarea de despliegue de servicios de ML. Úsalo como referencia bajo el contexto académico correspondiente.
