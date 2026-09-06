import io

import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}

print("Carregando o modelo de limpeza de lineart (só demora na primeira vez)...")
pipe = pipeline(
    "image-to-image",
    model="p1atdev/MangaLineExtraction-hf",
    trust_remote_code=True,
    device=0 if torch.cuda.is_available() else -1,
)
print("Modelo carregado. Servidor pronto.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def to_pil_image(saida):
    """Normaliza o retorno do pipeline para um PIL.Image, seja qual for o formato exato."""
    if isinstance(saida, list):
        saida = saida[0]
    if isinstance(saida, dict):
        saida = saida.get("image") or saida.get("pixel_values") or next(iter(saida.values()))
    if isinstance(saida, Image.Image):
        return saida
    import numpy as np

    return Image.fromarray(np.asarray(saida).astype("uint8"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/limpar", methods=["POST"])
def limpar():
    if "imagem" not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada."}), 400

    arquivo = request.files["imagem"]
    if arquivo.filename == "" or not allowed_file(arquivo.filename):
        return jsonify({"erro": "Arquivo inválido. Use PNG, JPG, WEBP ou BMP."}), 400

    try:
        imagem = Image.open(arquivo.stream).convert("RGB")
    except Exception:
        return jsonify({"erro": "Não foi possível abrir essa imagem."}), 400

    try:
        resultado = to_pil_image(pipe(imagem))
    except Exception as e:
        return jsonify({"erro": f"Falha ao processar a imagem: {e}"}), 500

    buffer = io.BytesIO()
    resultado.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
