# client.py  (sin dependencias extra)
# Uso:
#   python client.py img1.jpg img2.jpg img3.jpg \
#     --url https://vision-api-ml.onrender.com --out resultados_cliente.json
#
# Si pasas 1 o 2 imágenes, repite la primera para completar 3 requests.

import argparse, base64, hashlib, json, mimetypes, os, time
from pathlib import Path
from urllib import request as urlreq

def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def file_info(path: Path):
    b = path.read_bytes()
    return {
        "filename": path.name,
        "size_bytes": len(b),
        "mime": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "sha256_16": sha16(b),
        "bytes": b,  # solo en memoria para enviar
    }

def post_multipart(base_url: str, finfo: dict, timeout=60.0):
    """POST /predict con multipart/form-data (clave: file) usando urllib"""
    boundary = "----PyClientBoundary%08x" % int(time.time() * 1000)
    CRLF = b"\r\n"
    # Construir cuerpo a mano
    lines = []
    lines.append(b"--" + boundary.encode())
    lines.append(
        b'Content-Disposition: form-data; name="file"; filename="' +
        finfo["filename"].encode("utf-8") + b'"'
    )
    lines.append(b"Content-Type: " + finfo["mime"].encode("utf-8"))
    lines.append(b"")  # línea en blanco antes del binario
    body_head = CRLF.join(lines) + CRLF
    body_tail = CRLF + b"--" + boundary.encode() + b"--" + CRLF
    body = body_head + finfo["bytes"] + body_tail

    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    req = urlreq.Request(f"{base_url}/predict", data=body, headers=headers, method="POST")
    t0 = time.perf_counter()
    with urlreq.urlopen(req, timeout=timeout) as resp:
        dt = (time.perf_counter() - t0) * 1000
        raw = resp.read()
        return resp.getcode(), raw, dt

def post_json_b64(base_url: str, finfo: dict, timeout=60.0):
    """POST /predict_json con JSON (imagen en base64) usando urllib"""
    b64 = base64.b64encode(finfo["bytes"]).decode("utf-8")
    payload = json.dumps({"image_b64": b64}).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
    }
    req = urlreq.Request(f"{base_url}/predict_json", data=payload, headers=headers, method="POST")
    t0 = time.perf_counter()
    with urlreq.urlopen(req, timeout=timeout) as resp:
        dt = (time.perf_counter() - t0) * 1000
        raw = resp.read()
        return resp.getcode(), raw, dt, len(b64)

def get_json(url: str, timeout=30.0):
    req = urlreq.Request(url, headers={"Accept": "application/json"})
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://vision-api-ml.onrender.com", help="Base URL de la API")
    ap.add_argument("--out", default=None, help="Archivo para guardar resumen JSON")
    ap.add_argument("images", nargs="+", help="Rutas a imágenes (jpg/png)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    if not paths:
        print("Debes pasar al menos 1 imagen."); return
    # Asegura 3
    paths = paths + [paths[0]] * (3 - len(paths))

    print(f"Base URL: {args.url}")
    try:
        code, h = get_json(f"{args.url}/health")
        print("Health:", code, h)
    except Exception as e:
        print("Aviso: no pude consultar /health:", e)

    try:
        code, md = get_json(f"{args.url}/metadata")
        print("Metadata:", code, md)
    except Exception as e:
        print("Aviso: no pude consultar /metadata:", e)

    summary = []

    # 1) multipart
    finfo1 = file_info(paths[0])
    print("\n--- Request #1 — /predict (multipart) ---")
    print(f"Enviando: {finfo1['filename']} | {finfo1['size_bytes']} B | {finfo1['mime']} | sha16={finfo1['sha256_16']}")
    try:
        code, raw, ms = post_multipart(args.url, finfo1)
        print(f"Status: {code} | {ms:.1f} ms")
        j = json.loads(raw.decode("utf-8"))
        print(json.dumps(j, ensure_ascii=False, indent=2))
    except Exception as e:
        j = {"error": str(e)}
        print("Error:", e)
    summary.append({"req": 1, "endpoint": "/predict", "elapsed_ms": ms if 'ms' in locals() else None,
                    "input": {k: v for k, v in finfo1.items() if k != "bytes"}, "output": j})

    # 2) multipart (otra imagen)
    finfo2 = file_info(paths[1])
    print("\n--- Request #2 — /predict (multipart) ---")
    print(f"Enviando: {finfo2['filename']} | {finfo2['size_bytes']} B | {finfo2['mime']} | sha16={finfo2['sha256_16']}")
    try:
        code, raw, ms = post_multipart(args.url, finfo2)
        print(f"Status: {code} | {ms:.1f} ms")
        j = json.loads(raw.decode("utf-8"))
        print(json.dumps(j, ensure_ascii=False, indent=2))
    except Exception as e:
        j = {"error": str(e)}
        print("Error:", e)
    summary.append({"req": 2, "endpoint": "/predict", "elapsed_ms": ms if 'ms' in locals() else None,
                    "input": {k: v for k, v in finfo2.items() if k != "bytes"}, "output": j})

    # 3) JSON/base64
    finfo3 = file_info(paths[2])
    print("\n--- Request #3 — /predict_json (JSON base64) ---")
    print(f"Enviando: {finfo3['filename']} | {finfo3['size_bytes']} B | {finfo3['mime']} | sha16={finfo3['sha256_16']}")
    try:
        code, raw, ms, b64len = post_json_b64(args.url, finfo3)
        print(f"(longitud base64: {b64len} chars)")
        print(f"Status: {code} | {ms:.1f} ms")
        j = json.loads(raw.decode("utf-8"))
        print(json.dumps(j, ensure_ascii=False, indent=2))
    except Exception as e:
        j = {"error": str(e)}
        print("Error:", e)
        b64len = None
    summary.append({"req": 3, "endpoint": "/predict_json", "elapsed_ms": ms if 'ms' in locals() else None,
                    "b64_len": b64len, "input": {k: v for k, v in finfo3.items() if k != "bytes"}, "output": j})

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResumen guardado en: {args.out}")

if __name__ == "__main__":
    main()
