from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path
import pandas as pd
import torch

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
TOKEN = "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"

processor = AutoProcessor.from_pretrained(MODEL_ID, token=TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="cuda",
    token=TOKEN,
    torch_dtype=torch.bfloat16,
    attn_implementation="flex_attention",
)

folder = Path(r"E:\GIL\memes")
excel_salida = Path(r"E:\GIL\qwen35_scoring.xlsx")

extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
resultados = []

prompt = (
    "Look at the meme carefully, including any visible text. "
    "Is this meme sexist? Answer only YES or NO."
)

for ruta_imagen in folder.iterdir():
    if ruta_imagen.suffix.lower() not in extensiones:
        continue

    print(f"Procesando: {ruta_imagen.name}")

    image = Image.open(ruta_imagen).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

    respuesta = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    resultados.append({
        "archivo": ruta_imagen.name,
        "ruta": str(ruta_imagen),
        "respuesta_modelo": respuesta
    })
df = pd.DataFrame(resultados)
df.to_excel(excel_salida, index=False)

print(f"Archivo guardado en: {excel_salida}")