from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import pandas as pd
import torch

# modelo
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# rutas
folder = Path(r"C:\GIL\QWEN_3.5_4B\memes")
excel_salida = Path(r"C:\GIL\QWEN_3.5_4B\captions.xlsx")

extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
resultados = []

# prompt
prompt = "Is this meme sexist? Answer only YES or NO."

# procesar imágenes
for ruta_imagen in folder.iterdir():
    if ruta_imagen.suffix.lower() not in extensiones:
        continue

    print(f"Procesando: {ruta_imagen.name}")

    messages = [{"role": "user", "content": f"[Image: {ruta_imagen}] {prompt}"}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=10)

    respuesta = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    resultados.append({
        "archivo": ruta_imagen.name,
        "ruta": str(ruta_imagen),
        "respuesta_qwen": respuesta
    })

df = pd.DataFrame(resultados)
df.to_excel(excel_salida, index=False)

print(f"Archivo guardado en: {excel_salida}")