# Traço Limpo — cleanup de lineart local

Site simples (Flask) para testar limpeza automática de lineart num boceto rascunhado.
Usa o modelo **MangaLineExtraction** (licença MIT), carregado via Hugging Face
`transformers`. Tudo roda na sua própria máquina — a única vez que ele acessa a
internet é para baixar os pesos do modelo (uma única vez, na primeira execução).

## 1. Requisitos

- Python 3.9 ou mais recente
- ~2 GB livres (pesos do modelo + dependências)
- GPU é opcional — funciona em CPU, só um pouco mais lento por imagem

## 2. Instalação

```bash
cd lineart-cleanup
python -m venv venv
source venv/bin/activate        # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Rodar

```bash
python app.py
```

Na primeira execução ele vai baixar o modelo do Hugging Face automaticamente
(pode demorar um pouco dependendo da sua internet). Depois disso abre o navegador em:

```
http://localhost:5000
```

Arraste um boceto (PNG/JPG/WEBP/BMP) e o resultado limpo aparece do lado.

## 4. Sobre o modelo

- Repositório original: https://github.com/ljsabc/MangaLineExtraction_PyTorch
- Versão usada aqui (pronta para `transformers`): https://huggingface.co/p1atdev/MangaLineExtraction-hf
- Paper: "Deep Extraction of Manga Structural Lines" (SIGGRAPH 2017), Li, Liu, Wong
- Licença: MIT — uso livre, inclusive não-comercial como este projeto

## 5. Se quiser trocar de modelo depois

O código está isolado em `app.py`, na variável `pipe`. Para testar outro modelo de
limpeza de lineart (por exemplo o SketchKeras portado para PyTorch), basta trocar
o carregamento do pipeline por outro jeito de rodar aquele modelo — o resto do
site (upload, preview, download) continua igual.

## 6. Limitações conhecidas

- Traços muito caóticos ou com múltiplas interpretações possíveis ainda confundem
  o modelo — ele foi treinado majoritariamente com estilo mangá/anime, então
  funciona melhor nesse tipo de traço.
- Imagens muito grandes podem demorar mais para processar em CPU.
