# Despliegue de un modelo de ML como API (FastAPI + TensorFlow)

Servicio web que expone un modelo de clasificación de insectos como API REST usando FastAPI.
La API está desplegada en Render y entrega predicciones a partir de una imagen (JPEG/PNG), ya sea vía multipart/form-data o JSON (imagen en base64).

URL pública: https://vision-api-ml.onrender.com/

👀 Demo rápido
# Salud del servicio
curl -sS https://vision-api-ml.onrender.com/health

# Metadata del modelo (versión y clases)
curl -sS https://vision-api-ml.onrender.com/metadata

# Predicción (multipart/form-data)
curl -sS -X POST "https://vision-api-ml.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/ruta/a/imagen.jpg"

# Predicción (JSON: imagen en base64)
base64 -i /ruta/a/imagen.jpg | tr -d '\n' > img.b64
curl -sS -X POST "https://vision-api-ml.onrender.com/predict_json" \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\":\"$(cat img.b64)\"}"


📦 Estructura del repositorio
.
├─ app/
│  ├─ main.py            # FastAPI (endpoints /health, /predict, /predict_json, /metadata)
│  └─ inference.py       # Carga del modelo y lógica de preprocesamiento/predicción
├─ models/
│  ├─ modelo.h5          # Modelo entrenado (TensorFlow/Keras)
│  └─ classes.json       # Mapeo índice → nombre de clase
├─ client.py             # Cliente externo con 3 requests de ejemplo
├─ requirements.txt      # Dependencias para Linux/Render
├─ runtime.txt           # Versión de Python para Render (3.11.9)
└─ README.md


🧠 Modelo

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

🛠️ Instalación local

Requisitos:
* Python 3.11 (recomendado 3.11.9)
* macOS / Linux / WSL2 (Windows con WSL sufre menos con TensorFlow)
    # 1) crear y activar entorno
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip

    # 2) instalar dependencias
    pip install -r requirements.txt

Ejecutar el servidor localmente
    uvicorn app.main:app --reload
    # Abrir http://127.0.0.1:8000/docs

🧪 Cliente externo (3 requests)
    python client.py --url http://127.0.0.1:8000 img1.jpg img2.jpg img3.jpg
    # o contra Render:
    python client.py --url https://vision-api-ml.onrender.com img1.jpg img2.jpg img3.jpg
Si pasas 1 o 2 imágenes, el script reutiliza la primera para completar 3 llamadas.

☁️ Despliegue en Render
* Este repo incluye requirements.txt y runtime.txt (3.11.9).
* Comandos de Render:
   - Build: pip install -r requirements.txt
   - Start: gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT app.main:app --timeout 120

* Variables de entorno usadas por la app:
  - MODEL_PATH = models/modelo.h5
  - CLASSES_PATH = models/classes.json
  - MODEL_VERSION = 1.0.0

En plan free hay cold starts (la primera petición tras inactividad puede tardar algunos segundos).

# 🧪 Uso con Postman
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


🔒 Validaciones y manejo de errores

* 400 – Formato no soportado: si el archivo no es JPG/PNG.
* 400 – base64 inválido: si el JSON trae base64 malformado.
* 422 – ValidationError: esquema de entrada incorrecto.
* 500 – Errores internos inesperados (ver logs).

Mensajes de error devuelven detail legible para facilitar el diagnóstico.

🧩 Notas técnicas
* TensorFlow/keras probados en Python 3.11 (Render usa runtime.txt para fijar versión).
* Si modelo.h5 superara 100 MB, usa Git LFS para subirlo.

📄 Licencia
Este proyecto es educativo y forma parte de una tarea de despliegue de servicios de ML. Úsalo como referencia bajo el contexto académico correspondiente.
