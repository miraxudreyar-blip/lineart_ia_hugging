import io

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

app = Flask(__name__)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}
REPO_NAME = "p1atdev/MangaLineExtraction-hf"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LADO = 900  # limite de pixels no maior lado, para não estourar o tempo em CPU fraca

print("Carregando o modelo de limpeza de lineart (só demora na primeira vez)...")
model = AutoModel.from_pretrained(REPO_NAME, trust_remote_code=True).to(DEVICE).eval()
processor = AutoImageProcessor.from_pretrained(REPO_NAME, trust_remote_code=True)
print(f"Modelo carregado em {DEVICE}. Servidor pronto.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def redimensionar_se_grande(imagem: Image.Image) -> Image.Image:
    largura, altura = imagem.size
    maior_lado = max(largura, altura)
    if maior_lado <= MAX_LADO:
        return imagem
    escala = MAX_LADO / maior_lado
    novo_tamanho = (int(largura * escala), int(altura * escala))
    return imagem.resize(novo_tamanho, Image.LANCZOS)


def limpar_lineart(imagem: Image.Image) -> Image.Image:
    imagem = redimensionar_se_grande(imagem)
    inputs = processor(imagem, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values)

    saida = outputs.pixel_values[0]
    if isinstance(saida, torch.Tensor):
        saida = saida.cpu().numpy()

    arr = np.asarray(saida)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # (C, H, W) -> (H, W[, C])
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]

    arr = np.clip(arr, 0, 255).astype("uint8")
    modo = "L" if arr.ndim == 2 else "RGB"
    return Image.fromarray(arr, mode=modo)


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
        resultado = limpar_lineart(imagem)
    except Exception as e:
        return jsonify({"erro": f"Falha ao processar a imagem: {e}"}), 500

    buffer = io.BytesIO()
    resultado.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
