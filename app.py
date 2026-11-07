import re
import io
import cv2
import base64
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from streamlit_drawable_canvas import st_canvas

SCALE = 100.0

# --- Получаем ключ из Secrets ---
api_key = st.secrets.get("OPENAI_API_KEY", None)

@st.cache_resource
def load_face_cascade():
    try:
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(path)
        return cascade
    except:
        return None

face_cascade = load_face_cascade()

def auto_enhance_params(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    p_dark = np.percentile(lum, 10)
    p_bright = np.percentile(lum, 90)
    
    params = {}
    if p_bright > 0.7:
        params["highlights"] = -40
        params["exposure"] = -0.1
    if p_dark < 0.2:
        params["shadows"] = 30
        params["blacks"] = 10
    params["saturation"] = -10
    params["vibrance"] = 10
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
    if "exposure" in p: arr *= 2.0 ** p["exposure"]
    arr = np.clip(arr, 0, 1)

    temp_img = Image.fromarray((arr * 255).astype(np.uint8))
    if "contrast" in p: temp_img = ImageEnhance.Contrast(temp_img).enhance(max(0, 1 + p["contrast"] / SCALE))
    if "saturation" in p: temp_img = ImageEnhance.Color(temp_img).enhance(max(0, 1 + p["saturation"] / SCALE))
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

    if do_faces and face_cascade is not None:
        try:
            img_arr = np.asarray(img).astype(np.float32)
            h_orig, w_orig = img_arr.shape[:2]
            scale = min(1.0, 800.0 / max(h_orig, w_orig))
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
        except:
            pass

    return img

def gptunnel_edit_image(img: Image.Image, prompt: str) -> Image.Image:
    """Отправляет фото в GPTunnel (через Chat Completions, как Gemini)."""
    # Конвертируем фото в base64
    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Формат запроса для Gemini (Nano Banana) через текстовый чат
    payload = {
        "model": "nano-banana-v1", # Пробуем этот ID
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    try:
        with st.spinner("Нейросеть Nano Banana колдует..."):
            # Отправляем стандартный POST-запрос
            response = requests.post(
                "https://gptunnel.ru/v1/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                st.error(f"Ошибка {response.status_code}: {response.text}")
                return img
                
            data = response.json()
            # В ответе Gemini обычно возвращает текст, но если это картинка — она будет в base64
            content = data["choices"][0]["message"]["content"]
            
            # Ищем base64 картинку в ответе (иногда они оборачивают её в markdown)
            if "base64," in content:
                b64_str = content.split("base64,")[1].split(")")[0].split('"')[0]
                return Image.open(io.BytesIO(base64.b64decode(b64_str)))
            else:
                st.warning("Нейросеть вернула текст вместо картинки. Попробуй промт: 'Отредактируй это фото и верни результат'")
                return img
                
    except Exception as e:
        st.error(f"Ошибка запроса: {e}")
        return img

st.set_page_config(page_title="Photo Prompt App AI", layout="wide")
st.title("📷 Photo Prompt App (AI)")

if not api_key:
    st.warning("⚠️ API-ключ не найден в Secrets. Работает только базовый режим.")

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("Загрузи фото", type=["jpg", "jpeg", "png", "webp"])
    app_mode = st.radio("Режим работы:", ["Быстрый (математика)", "ИИ (GPTunnel)"])

    if app_mode == "Быстрый (математика)":
        auto_mode = st.checkbox("✨ Улучшить автоматически", value=True)
        prompt_text = st.text_area("Свои параметры", height=100, placeholder="shadows: +30")
        do_denoise = st.checkbox("🧹 Убрать шум")
        do_faces = st.checkbox("👤 Улучшить лица")
    else:
        st.info("💡 Опиши текстом, что хочешь изменить.")
        ai_prompt = st.text_area("Промт для нейросети:", "Сделай фотографию профессиональной и качественной. Убери шум, сделай цвета насыщеннее. Верни только готовое изображение.")

    run = st.button("Применить", type="primary")

if file:
    original = Image.open(file).convert("RGB")
    
    if run:
        if app_mode == "ИИ (GPTunnel)":
            if not api_key:
                st.error("Нужен API-ключ.")
            else:
                result = gptunnel_edit_image(original, ai_prompt)
                with col2:
                    st.image(result, caption="После (Нейросеть)", use_container_width=True)
                    buf = io.BytesIO()
                    result.save(buf, format="JPEG", quality=95)
                    st.download_button("💾 Скачать результат", buf.getvalue(), file_name="ai_result.jpg", mime="image/jpeg")
        else:
            params = {}
            if auto_mode:
                params = auto_enhance_params(original)
                st.info(f"🤖 Умный анализ: {params}")
            elif prompt_text.strip():
                params = parse_prompt(prompt_text)
                
            if not params and not do_denoise and not do_faces:
                st.error("Нет параметров.")
            else:
                with st.spinner("Обрабатываю..."):
                    result = apply_params(original, params, do_denoise, do_faces)
                
                with col2:
                    st.image(result, caption="После (Математика)", use_container_width=True)
                    buf = io.BytesIO()
                    result.save(buf, format="JPEG", quality=95)
                    st.download_button("💾 Скачать результат", buf.getvalue(), file_name="result.jpg", mime="image/jpeg")
