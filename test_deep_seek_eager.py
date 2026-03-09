from transformers import AutoTokenizer, AutoModel
import torch
import os
from pathlib import Path
import pandas as pd
import io
from contextlib import redirect_stdout, redirect_stderr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_id = "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

prompt_meme = "<image>\nFree OCR."
carpeta_memes = Path(r"E:\GIL\memes")
output_path = Path(r"E:\GIL\outcome")
output_path.mkdir(parents=True, exist_ok=True)

excel_salida = Path(r"E:\GIL\ocr_memes_deepseek.xlsx")

extensiones_validas = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
resultados = []

for ruta_imagen in carpeta_memes.iterdir():
    if ruta_imagen.suffix.lower() not in extensiones_validas:
        continue

    print(f"Procesando: {ruta_imagen.name}")

    buffer_out = io.StringIO()
    buffer_err = io.StringIO()

    try:
        with redirect_stdout(buffer_out), redirect_stderr(buffer_err):
            model.infer(
                tokenizer,
                prompt=prompt_meme,
                image_file=str(ruta_imagen),
                output_path=str(output_path),
                base_size=640,
                image_size=640,
                crop_mode=False,
                save_results=False,
                test_compress=False
            )

        texto_ocr = (buffer_out.getvalue() + "\n" + buffer_err.getvalue()).strip()

        resultados.append({
            "archivo": ruta_imagen.name,
            "ruta": str(ruta_imagen),
            "texto_ocr": texto_ocr
        })

    except Exception as e:
        resultados.append({
            "archivo": ruta_imagen.name,
            "ruta": str(ruta_imagen),
            "texto_ocr": None,
            "error": str(e)
        })

df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel(excel_salida, index=False)

print(f"Excel guardado en: {excel_salida}")
print(df_resultados[["archivo", "texto_ocr"]])