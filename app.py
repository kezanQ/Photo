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
    arr = np.asarray(img).astype(np.float32) / 255.0

    # 1. Шумоподавление
    if p.get("denoise", 0) > 0:
        s = p["denoise"] / SCALE * 2.0
        arr = gaussian_filter(arr, sigma=(s, s, 0))

    # 2. Экспозиция
    if "exposure" in p:
        arr *= 2.0 ** p["exposure"]

    arr = np.clip(arr, 0, 1)

    # 3. Контраст и насыщенность
    temp_img = Image.fromarray((arr * 255).astype(np.uint8))
    if "contrast" in p:
        temp_img = ImageEnhance.Contrast(temp_img).enhance(max(0, 1 + p["contrast"] / SCALE))
    if "saturation" in p:
        temp_img = ImageEnhance.Color(temp_img).enhance(max(0, 1 + p["saturation"] / SCALE))
    
    arr = np.asarray(temp_img).astype(np.float32) / 255.0

    # 4. Тени и света (через яркость Y, чтобы не портить цвета)
    if any(k in p for k in ["shadows", "highlights", "blacks", "whites"]):
        # Перевод в YCbCr (Яркость, Синий, Красный) - есть в чистом numpy
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = b - y
        cr = r - y

        if "shadows" in p:
            mask = np.clip((0.5 - y) / 0.5, 0, 1)
            y += mask * (p["shadows"] / SCALE) * 0.3
        if "highlights" in p:
            mask = np.clip((y - 0.5) / 0.5, 0, 1)
            y += mask * (p["highlights"] / SCALE) * 0.3
        if "blacks" in p:
            mask = np.clip((0.2 - y) / 0.2, 0, 1)
            y += mask * (p["blacks"] / SCALE) * 0.2
        if "whites" in p:
            mask = np.clip((y - 0.8) / 0.2, 0, 1)
            y += mask * (p["whites"] / SCALE) * 0.2

        y = np.clip(y, 0, 1)
        # Сборка обратно
        arr = np.stack([y + cr, y - 0.344136 * cb - 0.714136 * cr, y + cb], axis=-1)
        arr = np.clip(arr, 0, 1)

    # 5. Температура и оттенок
    if "temperature" in p:
        t = p["temperature"] / 1000.0
        arr[..., 0] += t * 0.1
        arr[..., 2] -= t * 0.1
    if "tint" in p:
        g = p["tint"] / SCALE
        arr[..., 1] -= g * 0.05
        arr[..., 0] += g * 0.02
        arr[..., 2] += g * 0.02

    arr = np.clip(arr, 0, 1)

    # 6. Vibrance
    if "vibrance" in p:
        maxc = arr.max(axis=2, keepdims=True)
        minc = arr.min(axis=2, keepdims=True)
        sat = (maxc - minc) / (maxc + 1e-6)
        boost = (p["vibrance"] / SCALE) * (1 - sat) * 0.3
        gray = arr.mean(axis=2, keepdims=True)
        arr = arr + boost * (arr - gray)
        arr = np.clip(arr, 0, 1)

    # 7. Clarity
    if "clarity" in p:
        blurred = gaussian_filter(arr, sigma=(8, 8, 0))
        arr = arr + (p["clarity"] / SCALE) * (arr - blurred)
        arr = np.clip(arr, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))

    # 8. Резкость
    if p.get("sharpen", 0) > 0:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2, percent=int(p["sharpen"]), threshold=3))

    # 9. Виньетка
    if "vignette" in p:
        arr = np.asarray(img).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        mask = np.clip(dist * 1.4, 0, 1)[..., None]
        arr = arr * (1 + (p["vignette"] / SCALE) * mask)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

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
