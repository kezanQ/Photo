import re
import io
import cv2
import numpy as np
import urllib.request
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from streamlit_drawable_canvas import st_canvas

SCALE = 100.0

@st.cache_resource
def load_face_cascade():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty(): raise Exception("Empty")
        return cascade
    except:
        filename = "haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, filename)
        return cv2.CascadeClassifier(filename)

face_cascade = load_face_cascade()

# --- НОВЫЙ УМНЫЙ АВТОАНАЛИЗ ---
def auto_enhance_params(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    
    # Считаем яркость
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Анализ гистограммы
    p_dark = np.percentile(lum, 10)   # Самые темные 10% пикселей
    p_bright = np.percentile(lum, 90) # Самые светлые 10% пикселей
    
    params = {}
    
    # Если на фото есть яркие источники света (как гирлянда)
    if p_bright > 0.7:
        params["highlights"] = -40  # Приглушаем их, чтобы не слепили
        params["exposure"] = -0.1   # Чуть-чуть убавляем общую яркость
    
    # Если много темных провалов
    if p_dark < 0.2:
        params["shadows"] = 30      # Аккуратно поднимаем тени
        params["blacks"] = 10       # И чуть-чуть черную точку
        
    # Защита от «ядовитых» цветов на ночных фото
    # Убавляем общую насыщенность, чтобы убрать кислотность
    params["saturation"] = -10
    # Но добавляем вибранс (он умный и не трогает кожу)
    params["vibrance"] = 10
    
    # Добавляем объема
    params["clarity"] = 15
    params["sharpen"] = 20
    
    return params

def parse_prompt(text: str) -> dict:
    params = {}
    for line in text.splitlines():
        m = re.match(r"\s*([a-zA-Z_]+)\s*:\s*([+-]?\d+(?:\.\d+)?)", line)
        if m:
            params[m.group(1).lower()] = float(m.group(2))
    return params

def apply_params(img: Image.Image, p: dict, do_denoise=False, do_faces=False) -> Image.Image:
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    if do_denoise or p.get("denoise", 0) > 0:
        h = int(p.get("denoise", 10))
        arr = cv2.fastNlMeansDenoisingColored(arr.astype(np.uint8), None, h, h, 7, 21)
        arr = arr.astype(np.float32)

    arr = arr / 255.0
    if "exposure" in p:
        arr *= 2.0 ** p["exposure"]
    arr = np.clip(arr, 0, 1)

    temp_img = Image.fromarray((arr * 255).astype(np.uint8))
    if "contrast" in p:
        temp_img = ImageEnhance.Contrast(temp_img).enhance(max(0, 1 + p["contrast"] / SCALE))
    if "saturation" in p:
        temp_img = ImageEnhance.Color(temp_img).enhance(max(0, 1 + p["saturation"] / SCALE))
    
    arr = np.asarray(temp_img).astype(np.float32) / 255.0

    if any(k in p for k in ["shadows", "highlights", "blacks", "whites"]):
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
        arr = np.stack([y + cr, y - 0.344136 * cb - 0.714136 * cr, y + cb], axis=-1)
        arr = np.clip(arr, 0, 1)

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

    if "vibrance" in p:
        maxc = arr.max(axis=2, keepdims=True)
        minc = arr.min(axis=2, keepdims=True)
        sat = (maxc - minc) / (maxc + 1e-6)
        boost = (p["vibrance"] / SCALE) * (1 - sat) * 0.3
        gray = arr.mean(axis=2, keepdims=True)
        arr = arr + boost * (arr - gray)
        arr = np.clip(arr, 0, 1)

    if "clarity" in p:
        blurred = gaussian_filter(arr, sigma=(8, 8, 0))
        arr = arr + (p["clarity"] / SCALE) * (arr - blurred)
        arr = np.clip(arr, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))

    if p.get("sharpen", 0) > 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(p["sharpen"]), threshold=3))

    if "vignette" in p:
        arr = np.asarray(img).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        mask = np.clip(dist * 1.4, 0, 1)[..., None]
        arr = arr * (1 + (p["vignette"] / SCALE) * mask)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    if do_faces:
        img_arr = np.asarray(img).astype(np.float32)
        h_orig, w_orig = img_arr.shape[:2]
        scale = min(1.0, 1000.0 / max(h_orig, w_orig))
        if scale < 1.0:
            img_small = cv2.resize(img_arr, (int(w_orig*scale), int(h_orig*scale)))
        else:
            img_small = img_arr

        gray = cv2.cvtColor(img_small.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            face_mask_small = np.zeros(gray.shape, dtype=np.float32)
            for (x, y, w_face, h_face) in faces:
                pad_w = int(w_face * 0.2)
                pad_h = int(h_face * 0.2)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(gray.shape[1], x + w_face + pad_w), min(gray.shape[0], y + h_face + pad_h)
                cv2.rectangle(face_mask_small, (x1, y1), (x2, y2), 1.0, -1)
            
            face_mask_small = gaussian_filter(face_mask_small, sigma=10)
            face_mask = cv2.resize(face_mask_small, (w_orig, h_orig))
            face_mask = face_mask[:, :, None]
            
            face_soft = cv2.GaussianBlur(img_arr, (15, 15), 0)
            face_bright = img_arr * 1.15
            
            blended = img_arr * (1 - face_mask) + face_bright * face_mask * 0.7 + face_soft * face_mask * 0.3
            img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    return img

st.set_page_config(page_title="Photo Prompt App", layout="wide")
st.title("📷 Photo Prompt App Pro")

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("Загрузи фото", type=["jpg", "jpeg", "png", "webp"])
    
    auto_mode = st.checkbox("✨ Улучшить автоматически (Умный анализ)", value=True)
    
    prompt_text = st.text_area(
        "Или вставь свой блок параметров",
        height=150,
        placeholder="exposure: +0.3\nshadows: +40\n..."
    )
    
    st.subheader("Умные функции")
    do_denoise = st.checkbox("🧹 Убрать шум (OpenCV AI)")
    do_faces = st.checkbox("👤 Улучшить лица")
    
    mode = st.radio("Режим работы:", ["Применить ко всему фото", "Рисовать маску (точечно)"], horizontal=True)
    
    canvas_result = None
    if file and mode == "Рисовать маску (точечно)":
        original = Image.open(file).convert("RGB")
        w, h = original.size
        max_size = 800
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            w, h = int(w * ratio), int(h * ratio)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.5)",
            stroke_width=40,
            background_image=original.resize((w, h)),
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="freedraw",
            key="canvas",
        )

    run = st.button("Применить", type="primary")

if file:
    original = Image.open(file).convert("RGB")
    
    if run:
        params = {}
        
        if auto_mode:
            params = auto_enhance_params(original)
            st.info(f"🤖 Умный анализ применил: {params}")
        elif prompt_text.strip():
            params = parse_prompt(prompt_text)
            
        if not params and not do_denoise and not do_faces:
            st.error("Нет параметров и не выбраны умные функции!")
        else:
            with st.spinner("Обрабатываю..."):
                if mode == "Рисовать маску (точечно)" and canvas_result is not None and canvas_result.image_data is not None:
                    mask_raw = canvas_result.image_data[:, :, 3]
                    if mask_raw.sum() == 0:
                        st.warning("Область не закрашена! Применяю ко всему фото.")
                        result = apply_params(original, params, do_denoise, do_faces)
                    else:
                        mask_img = Image.fromarray((mask_raw * 255).astype(np.uint8))
                        mask_img = mask_img.resize(original.size, Image.Resampling.LANCZOS)
                        mask_arr = np.asarray(mask_img).astype(np.float32) / 255.0
                        mask_arr = gaussian_filter(mask_arr, sigma=5)
                        
                        full_result = apply_params(original, params, do_denoise, do_faces)
                        full_arr = np.asarray(full_result).astype(np.float32)
                        orig_arr = np.asarray(original).astype(np.float32)
                        
                        mask_3d = np.stack([mask_arr]*3, axis=-1)
                        blended = orig_arr * (1 - mask_3d) + full_arr * mask_3d
                        result = Image.fromarray(blended.astype(np.uint8))
                else:
                    result = apply_params(original, params, do_denoise, do_faces)
            
            with col2:
                st.image(result, caption="После", use_container_width=True)
                buf = io.BytesIO()
                result.save(buf, format="JPEG", quality=95)
                st.download_button("💾 Скачать результат", buf.getvalue(), file_name="result.jpg", mime="image/jpeg")
