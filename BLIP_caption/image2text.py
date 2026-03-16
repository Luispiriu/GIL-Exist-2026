from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import pandas as pd

# modelo
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cuda")
model.eval()

# rutas
folder = Path(r"C:\GIL\BLIP_caption\memes")
excel_salida = Path(r"C:\GIL\BLIP_caption\captions.xlsx")

extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
resultados = []

# prompt
prompt = "a meme showing"
# procesar imágenes
for ruta_imagen in folder.iterdir():
    if ruta_imagen.suffix.lower() not in extensiones:
        continue

    print(f"Procesando: {ruta_imagen.name}")

    raw_image = Image.open(ruta_imagen).convert("RGB")

    inputs = processor(
        raw_image,
        prompt,
        return_tensors="pt"
    ).to("cuda")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    resultados.append({
        "archivo": ruta_imagen.name,
        "ruta": str(ruta_imagen),
        "caption_blip": caption
    })

df = pd.DataFrame(resultados)
df.to_excel(excel_salida, index=False)

print(f"Archivo guardado en: {excel_salida}")