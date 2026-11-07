import re
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter

SCALE = 100.0

def parse_prompt(text: str) -> dict:
    params = {}
    for line in text.splitlines():
        m = re.match(r"\s*([a-zA-Z_]+)\s*:\s*([+-]?\d+(?:\.\d+)?)", line)
        if m:
            params[m.group(1).lower()] = float(m.group(2))
    return params

def apply_params(img: Image.Image, p: dict) -> Image.Image:
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    if p.get("denoise", 0) > 0:
        s = p["denoise"] / SCALE * 2.0
        arr = gaussian_filter(arr, sigma=(s, s, 0))

    if "exposure" in p:
        arr *= 2.0 ** p["exposure"]

    arr = np.clip(arr, 0, 255)
    lum = arr.mean(axis=2, keepdims=True)

    if "shadows" in p:
        mask = np.clip((128 - lum) / 128, 0, 1)
        arr += mask * (p["shadows"] / SCALE) * 80
    if "highlights" in p:
        mask = np.clip((lum - 128) / 128, 0, 1)
        arr += mask * (p["highlights"] / SCALE) * 80
    if "blacks" in p:
        mask = np.clip((64 - lum) / 64, 0, 1)
        arr += mask * (p["blacks"] / SCALE) * 60
    if "whites" in p:
        mask = np.clip((lum - 192) / 64, 0, 1)
        arr += mask * (p["whites"] / SCALE) * 60

    arr = np.clip(arr, 0, 255)

    if "temperature" in p:
        t = p["temperature"] / 1000.0
        arr[..., 0] += t * 30
        arr[..., 2] -= t * 30
    if "tint" in p:
        g = p["tint"] / SCALE
        arr[..., 1] -= g * 20
        arr[..., 0] += g * 10
        arr[..., 2] += g * 10

    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    if "contrast" in p:
        img = ImageEnhance.Contrast(img).enhance(max(0, 1 + p["contrast"] / SCALE))
    if "saturation" in p:
        img = ImageEnhance.Color(img).enhance(max(0, 1 + p["saturation"] / SCALE))

    if "vibrance" in p:
        arr = np.asarray(img).astype(np.float32)
        mx = arr.max(axis=2, keepdims=True)
        mn = arr.min(axis=2, keepdims=True)
        sat = (mx - mn) / (mx + 1e-6)
        boost = (p["vibrance"] / SCALE) * (1 - sat) * 60
        gray = arr.mean(axis=2, keepdims=True)
        arr = arr + boost * (arr - gray)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if "clarity" in p:
        arr = np.asarray(img).astype(np.float32)
        blurred = gaussian_filter(arr, sigma=(8, 8, 0))
        arr = arr + (p["clarity"] / SCALE) * (arr - blurred)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if p.get("sharpen", 0) > 0:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2, percent=int(p["sharpen"]), threshold=3))

    if "vignette" in p:
        arr = np.asarray(img).astype(np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        mask = np.clip(dist * 1.4, 0, 1)[..., None]
        arr = arr * (1 + (p["vignette"] / SCALE) * mask)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return img

st.set_page_config(page_title="Photo Prompt App", layout="wide")
st.title("📷 Photo Prompt App")

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("Загрузи фото", type=["jpg", "jpeg", "png", "webp"])
    prompt_text = st.text_area(
        "Вставь блок параметров",
        height=300,
        placeholder="exposure: +0.3\nshadows: +40\nvibrance: +20\n..."
    )
    run = st.button("Применить", type="primary")

if file:
    original = Image.open(file).convert("RGB")
    with col1:
        st.image(original, caption="До", use_container_width=True)

    if run and prompt_text.strip():
        params = parse_prompt(prompt_text)
        if not params:
            st.error("Не распознал ни одного параметра. Формат: ключ: значение")
        else:
            with st.spinner("Обрабатываю..."):
                result = apply_params(original, params)
            with col2:
                st.image(result, caption="После", use_container_width=True)
                buf = io.BytesIO()
                result.save(buf, format="JPEG", quality=95)
                st.download_button(
                    "💾 Скачать результат",
                    buf.getvalue(),
                    file_name="result.jpg",
                    mime="image/jpeg",
                )
            st.success(f"Применено параметров: {len(params)}")
            with st.expander("Что распознано"):
                st.json(params)
