# client.py
import argparse, base64, httpx, json, os, sys
from pathlib import Path

def send_multipart(client, url, image_path):
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = client.post(f"{url}/predict", files=files, timeout=60)
    return r

def send_json(client, url, image_path):
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    payload = {"image_b64": b64}
    r = client.post(f"{url}/predict_json", json=payload, timeout=60)
    return r

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL de la API")
    p.add_argument("images", nargs="+", help="Rutas a imágenes (jpg/png)")
    args = p.parse_args()

    if len(args.images) < 1:
        print("Proporciona al menos 1 imagen", file=sys.stderr); sys.exit(1)
    imgs = args.images + [args.images[0]]*(3-len(args.images))  # asegura 3

    print(f"Usando API: {args.url}")
    with httpx.Client() as client:
        # 1) multipart
        r1 = send_multipart(client, args.url, imgs[0])
        # 2) multipart
        r2 = send_multipart(client, args.url, imgs[1])
        # 3) JSON/base64
        r3 = send_json(client, args.url, imgs[2])

    for i, r in enumerate([r1, r2, r3], start=1):
        print(f"\n--- Request #{i} ---")
        print("Status:", r.status_code)
        try:
            print(json.dumps(r.json(), ensure_ascii=False, indent=2))
        except Exception:
            print(r.text)

if __name__ == "__main__":
    main()
