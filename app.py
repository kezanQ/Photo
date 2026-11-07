import re
import io
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from streamlit_drawable_canvas import st_canvas

SCALE = 100.0

# Инициализация MediaPipe для лиц
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

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

    # 1. Шумоподавление (OpenCV)
    if do_denoise or p.get("denoise", 0) > 0:
        h = int(p.get("denoise", 10)) # сила шума
        arr = cv2.fastNlMeansDenoisingColored(arr.astype(np.uint8), None, h, h, 7, 21)
        arr = arr.astype(np.float32)

    # 2. Экспозиция
    arr = arr / 255.0
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

    # 4. Тени и света (YCbCr)
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
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(p["sharpen"]), threshold=3))

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

    # 10. Улучшение лиц (MediaPipe)
    if do_faces:
        rgb_img = np.asarray(img)
        results = face_detection.process(rgb_img)
        
        if results.detections:
            img_arr = np.asarray(img).astype(np.float32)
            h, w = img_arr.shape[:2]
            
            # Создаем пустую маску для лиц
            face_mask = np.zeros((h, w), dtype=np.float32)
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Рисуем белый прямоугольник на маске (с запасом)
                cv2.rectangle(face_mask, (max(0, x-20), max(0, y-20)), 
                              (min(w, x+width+20), min(h, y+height+20)), 1.0, -1)
            
            # Размываем маску, чтобы края были плавными
            face_mask = gaussian_filter(face_mask, sigma=15)
            face_mask = face_mask[:, :, None]
            
            # Делаем лица чуть светлее и мягче (блюр для кожи)
            face_soft = cv2.GaussianBlur(img_arr, (15, 15), 0)
            face_bright = img_arr * 1.15 # +15% яркости на лице
            
            # Смешиваем: (оригинал * (1 - маска)) + (новое_лицо * маска)
            blended = img_arr * (1 - face_mask) + face_bright * face_mask * 0.7 + face_soft * face_mask * 0.3
            img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    return img

st.set_page_config(page_title="Photo Prompt App", layout="wide")
st.title("📷 Photo Prompt App Pro")

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("Загрузи фото", type=["jpg", "jpeg", "png", "webp"])
    
    prompt_text = st.text_area(
        "Вставь блок параметров",
        height=150,
        placeholder="exposure: +0.3\nshadows: +40\nvibrance: +20\n..."
    )
    
    # Новые чекбоксы
    st.subheader("Умные функции")
    do_denoise = st.checkbox("🧹 Убрать шум (OpenCV AI)")
    do_faces = st.checkbox("👤 Улучшить лица (свет + мягкая кожа)")
    
    mode = st.radio("Режим работы:", ["Применить ко всему фото", "Рисовать маску (точечно)"], horizontal=True)
    
    canvas_result = None
    if file and mode == "Рисовать маску (точечно)":
        st.info("Закрась пальцем область, к которой применить эффект.")
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
    
    if run and prompt_text.strip():
        params = parse_prompt(prompt_text)
        if not params and not do_denoise and not do_faces:
            st.error("Нет параметров и не выбраны умные функции!")
        else:
            with st.spinner("Обрабатываю (нейросети работают)..."):
                if mode == "Рисовать маску (точечно)" and canvas_result is not None and canvas_result.image_data is not None:
                    mask_raw = canvas_result.image_data[:, :, 3]
                    if mask_raw.sum() == 0:
                        st.warning("Ты не закрасил(а) ни одной области! Применяю ко всему фото.")
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
