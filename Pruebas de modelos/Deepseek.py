from openai import OpenAI
from pathlib import Path
import pandas as pd
import base64

API_KEY = "TU_NUEVA_API_KEY"   # reemplázala
MODEL = "deepseek-chat"        # o "deepseek-reasoner"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

folder = Path(r"E:\GIL\memes")
excel_salida = Path(r"E:\GIL\deepseek_api_yesno.xlsx")

extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
resultados = []

prompt = "Is this meme sexist? Answer only YES or NO."

for ruta_imagen in folder.iterdir():
    if ruta_imagen.suffix.lower() not in extensiones:
        continue

    print(f"Procesando: {ruta_imagen.name}")

    image_bytes = ruta_imagen.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{ruta_imagen.suffix.lower().replace('.', '')};base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=5
    )

    respuesta = response.choices[0].message.content.strip()

    resultados.append({
        "archivo": ruta_imagen.name,
        "ruta": str(ruta_imagen),
        "respuesta_modelo": respuesta
    })

df = pd.DataFrame(resultados)
df.to_excel(excel_salida, index=False)

print(f"Archivo guardado en: {excel_salida}")