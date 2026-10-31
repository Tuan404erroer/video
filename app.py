from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
from pathlib import Path
import cv2
import numpy as np
import subprocess
import gc
import toml
import sys
import threading
import importlib.util
import psutil
import shutil
import torch
from pathlib import Path
import os

app = Flask(__name__)

# Chuyển Hugging Face cache sang ổ D để tránh đầy ổ C
os.environ["HF_HOME"] = "d:/hf_cache"
os.environ["TORCH_HOME"] = "d:/torch_cache"
os.environ["TRANSFORMERS_CACHE"] = "d:/hf_cache/transformers"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEVICE_LABEL = "Khóa 1:1 không cần GPU; AI AnimateDiff + ControlNet tự kiểm tra khi chạy"

# Fix triệt để lỗi FileNotFoundError: Ép dùng đường dẫn tuyệt đối (Absolute Path)
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
FRAMES_DIR = BASE_DIR / "frames"
STATIC_DIR = BASE_DIR / "static"

for d in [OUTPUT_DIR, FRAMES_DIR, STATIC_DIR]:
    d.mkdir(exist_ok=True, parents=True)

FRAME_SIZE = (512, 704)  # width=512, height=704 (tỷ lệ dọc)
# Test mode: True = nhanh để test, False = chất lượng full
TEST_MODE = True
TEST_FRAMES = 30 if TEST_MODE else 30  # 30 frames = 2.5 giây @ 12fps
AI_STRENGTH = 0.25 if not TEST_MODE else 0.85  # MAX để cho animation (0.75 → 0.85)
OUTPUT_FPS = 12
ANIMATEDIFF_SEGMENT = 16
SEGMENT_OVERLAP = 4
AI_GUIDANCE_SCALE = 5.0       # Giảm để AI tự do tạo motion nhiều hơn (6.5 → 5.0)

# Bật/tắt ControlNet Tile
# RTX 2050 4GB: tắt ControlNet để tiết kiệm ~1.5GB VRAM (bật lại nếu có GPU lớn hơn)
USE_CONTROLNET = False  # False = dùng AnimateDiffVideoToVideoPipeline thuần (ít RAM hơn)
USE_IMAGE_TO_VIDEO = False  # Chuyển lại AnimateDiff (Tối ưu VRAM cho ổ D)

CONSISTENCY_NEGATIVE_PROMPT = (
    "static image, freeze frame, still photo, no movement, frozen, motionless, "
    "completely still, stopped, paused, inanimate, rigid, stiff, "
    "different person, different face, changed identity, changed clothes, "
    "changed background, distorted face, deformed hands, extra fingers, "
    "flicker, inconsistent details, low quality, blurry, watermark, "
    "horror, creepy, scary, deformed body, melting, morphing, "
    "bad anatomy, poorly drawn hands, poorly drawn face, mutation, "
    "ugly, bad proportions, extra limbs, cloned face, disfigured, "
    "missing arms, missing legs, extra arms, extra legs, fused fingers, "
    "too many fingers, long neck, cross-eyed, "
    "multiple panels, split screen, divided frame, comic panels, "
    "grid layout, collage, montage, multiple scenes, separate frames, "
    "identical frames, repeated frames, no camera motion, no depth"
)

global_pipe = None
gemini_client = None
progress_lock = threading.Lock()
render_lock = threading.Lock()
progress_state = {
    "running": False,
    "done": False,
    "mode": None,
    "current": 0,
    "total": 0,
    "message": "Chưa render",
    "error": None,
    "result_video": None,
}

# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def set_progress(**updates):
    with progress_lock:
        progress_state.update(updates)

def get_progress_snapshot():
    with progress_lock:
        return dict(progress_state)

# ---------------------------------------------------------------------------
# CUDA pre-check
# ---------------------------------------------------------------------------

def get_torch_cuda_runtime_hint():
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return False, "Chưa cài Torch nên không chạy được AI redraw."
    torch_dir = Path(spec.origin).parent
    lib_dir = torch_dir / "lib"
    cuda_markers = (
        "torch_cuda.dll", "c10_cuda.dll",
        "torch_cuda.so", "libtorch_cuda.so", "libc10_cuda.so",
    )
    if not any((lib_dir / marker).exists() for marker in cuda_markers):
        return (
            False,
            "Torch hiện tại là bản CPU. AI redraw sẽ cực lâu, nên app đã dừng sớm. "
            "Muốn dùng chế độ này cần cài Torch CUDA và có GPU NVIDIA."
        )
    return True, "Đã thấy Torch CUDA runtime, đang kiểm tra GPU..."

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        from google import genai
        secrets = toml.load(".streamlit/secrets.toml")
        gemini_client = genai.Client(api_key=secrets["GOOGLE_API_KEY"])
    return gemini_client

def get_stable_motion_prompt(image_path):
    """
    Gemini phân tích ảnh và tạo prompt tập trung vào:
    1. CAMERA MOTION - bay vào thế giới trong ảnh
    2. OBJECT MOTION - vật thể trong ảnh tiếp tục hành động tự nhiên
    3. ENVIRONMENT MOTION - môi trường sống động
    """
    model_name = "gemini-2.0-flash"  # gemini-2.0-flash-lite đã bị deprecated
    prompt = (
        "You are a cinematic video director. Analyze this image and write a prompt "
        "that describes a SHORT VIDEO CLIP continuing from this exact moment.\n\n"

        "CRITICAL RULES:\n"
        "1. The video MUST have VISIBLE MOVEMENT. A static result is a FAILURE.\n"
        "2. Describe what happens NEXT - the continuation of this frozen moment.\n"
        "3. Include BOTH camera motion AND object/environment motion.\n\n"

        "Step 1 - WHAT MOVES in this scene?\n"
        "- People: breathing, blinking, slight head turn, hand gesture, walking\n"
        "- Objects: curtains swaying, papers rustling, steam rising, clock ticking\n"
        "- Nature: leaves rustling, water flowing, clouds drifting, grass swaying\n"
        "- Light: sunbeams shifting, shadows moving, reflections dancing\n"
        "- Particles: dust floating, rain drops, snow falling, smoke curling\n\n"

        "Step 2 - CAMERA MOTION (pick one):\n"
        "- 'camera slowly dollying forward through the space' (rooms/interiors)\n"
        "- 'camera tracking left/right revealing more of the scene' (wide scenes)\n"
        "- 'camera gently pushing in closer' (portraits/details)\n"
        "- 'camera rising up with slight tilt down' (landscapes)\n"
        "- 'camera orbiting slowly around the subject' (objects/people)\n\n"

        "Step 3 - DEPTH and PARALLAX:\n"
        "- Foreground objects shift faster than background\n"
        "- Include depth cues: 'parallax motion, 3D depth, cinematic perspective'\n\n"

        "OUTPUT FORMAT: Write exactly ONE line. Combine:\n"
        "[scene description], [what objects/people are DOING], "
        "[camera motion], [environmental motion], [depth/quality keywords]\n\n"

        "EXAMPLE: 'cozy living room with warm sunlight, curtains gently swaying in breeze, "
        "dust particles floating in light beams, camera slowly dollying forward through room, "
        "parallax depth with furniture shifting perspective, cinematic soft lighting, "
        "subtle shadows moving across floor, photorealistic, high detail'\n\n"

        "IMPORTANT: Use action verbs (swaying, floating, moving, shifting, flowing, "
        "drifting, dancing, rustling, flickering) NOT static adjectives."
    )
    try:
        img = Image.open(image_path)
        client = get_gemini_client()
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, img]
        )
        text = response.text.strip().splitlines()[0].strip().lstrip("0123456789.-* ")
        if text:
            print(f"Gemini prompt: {text}")
            return text
    except Exception as e:
        print(f"Gemini thất bại, dùng fallback: {e}")
    return (
        "same scene continuing naturally, camera slowly dollying forward, "
        "objects gently moving, curtains swaying in soft breeze, "
        "dust particles floating in light, subtle shadows shifting, "
        "atmospheric depth, parallax motion, cinematic lighting, photorealistic"
    )

# ---------------------------------------------------------------------------
# ControlNet Tile helper
# ---------------------------------------------------------------------------

def make_canny_control_image(original_pil: Image.Image) -> Image.Image:
    """
    Tạo Canny edge map từ ảnh gốc.
    Canny giữ shape/contour của objects, kết hợp với Tile giữ texture.
    """
    import cv2
    import numpy as np
    
    img_array = np.array(original_pil)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_rgb)
def make_tile_control_image(original_pil: Image.Image) -> Image.Image:
    """
    ControlNet Tile dùng chính ảnh gốc làm conditioning.
    Nó ép mỗi frame giữ đúng spatial layout và màu sắc của ảnh gốc.
    """
    return original_pil.copy()

# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def get_image_to_video_pipeline():
    """
    SVD Image-to-Video Pipeline - sinh video TRỰC TIẾP từ ảnh gốc
    Không cần prompt, không tạo scene mới, chỉ animate ảnh có sẵn
    """
    global global_pipe
    if global_pipe is not None:
        return global_pipe

    import torch
    from diffusers import StableVideoDiffusionPipeline
    
    print("▶ Nạp SVD Image-to-Video pipeline...")
    
    # SVD cơ bản: 14 frames (nhẹ hơn SVD-XT 25 frames)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",  # Bỏ "-xt" = nhẹ hơn
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # AGGRESSIVE memory optimization cho RTX 2050 4GB
    pipe.enable_model_cpu_offload()  # Chuyển model lên CPU khi không dùng
    pipe.enable_sequential_cpu_offload()  # Toàn bộ lên CPU
    
    # Clear toàn bộ VRAM trước
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Set low memory mode
    pipe.unet.set_default_attn_processor()  # Disable fancy attention
    
    global_pipe = pipe
    print("✅ SVD Image-to-Video pipeline sẵn sàng (Ultra CPU offload)")
    return pipe

def get_animatediff_pipeline():
    """
    Nếu USE_CONTROLNET=True:
        AnimateDiffPipeline + ControlNetModel (tile) + IP-Adapter
        → spatial layout bị neo, nhân vật không drift, animation mượt
    Nếu USE_CONTROLNET=False:
        AnimateDiffPipeline + IP-Adapter (fallback về cũ)
    """
    global global_pipe
    if global_pipe is not None:
        return global_pipe

    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler

    # Kiểm tra RAM trước khi load
    import psutil
    ram_free_gb = psutil.virtual_memory().available / 1024**3
    print(f"  RAM trống trước load: {ram_free_gb:.1f}GB")
    if ram_free_gb < 2.5:
        raise MemoryError(
            f"Không đủ RAM: chỉ còn {ram_free_gb:.1f}GB, cần ít nhất 2.5GB. "
            "Hãy đóng bớt các ứng dụng khác."
        )

    print("▶ Nạp Motion Adapter v1-5-3...")
    # Giải phóng RAM + VRAM trước khi load model lớn
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if USE_CONTROLNET:
        from diffusers import ControlNetModel, AnimateDiffVideoToVideoControlNetPipeline

        print("▶ Nạp ControlNet Tile (single - tiết kiệm VRAM)...")
        controlnet_tile = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch_dtype,
        )

        print("▶ Nạp SD 1.5 + AnimateDiff + ControlNet Tile pipeline...")
        pipe = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            controlnet=controlnet_tile,
            torch_dtype=torch_dtype,
            variant="fp16",         # Tải thẳng file fp16 (ít RAM gấp đôi)
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        print("✓ ControlNet Tile đã gắn vào pipeline.")
    else:
        from diffusers import AnimateDiffVideoToVideoPipeline

        print("▶ Nạp SD 1.5 + AnimateDiff VideoToVideo pipeline...")
        pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=torch_dtype,
            variant="fp16",         # Tải thẳng file fp16 (ít RAM gấp đôi)
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        timestep_spacing="trailing",
    )

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers OK")
    except Exception:
        pass

    # Offload lên CPU - DÙNG enable_model_cpu_offload thay vì sequential để tránh conflict IP-Adapter
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()  # Thay sequential bằng model offload
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        pipe.to("cpu")
        print("⚠ Chạy CPU — sẽ chậm.")

    # Giải phóng RAM sau khi offload xong
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # IP-Adapter — load SAU khi offload để giữ RAM thấp
    print("▶ Nạp IP-Adapter sd15...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
    )
    pipe.set_ip_adapter_scale(0.15)

    global_pipe = pipe
    print("✓ Pipeline sẵn sàng.")
    return global_pipe

# ---------------------------------------------------------------------------
# Frame / image utilities
# ---------------------------------------------------------------------------

def clear_frames():
    """Xóa frames cũ + output cũ + cleanup VRAM để chuẩn bị video mới"""
    # Đảm bảo thư mục frames tồn tại (có thể bị xóa bởi cleanup)
    FRAMES_DIR.mkdir(exist_ok=True)
    
    # Xóa frames cũ
    for f in FRAMES_DIR.glob("*"):
        try:
            f.unlink()
        except:
            pass
    
    # XÓA VIDEO CŨ để tránh cache
    old_video = Path("static/output.mp4")
    if old_video.exists():
        try:
            old_video.unlink()
            print("  ✓ Xóa video cũ")
        except:
            pass
    
    # Cleanup VRAM cho video mới        
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    gc.collect()

def fit_to_model_frame(image):
    """
    Resize ảnh về 512×704 ưu tiên hình chữ nhật dọc.
    Nếu ảnh ngang → crop thành vuông rồi pad thành dọc.
    Nếu ảnh dọc → pad trực tiếp thành 512×704.
    """
    original_w, original_h = image.size
    
    # Nếu ảnh ngang hoặc vuông → crop thành vuông (lấy phần giữa)
    if original_w >= original_h:
        size = min(original_w, original_h)
        left = (original_w - size) // 2
        top = (original_h - size) // 2
        image = image.crop((left, top, left + size, top + size))
    
    # Giờ pad thành 512×704 (ưu tiên dọc)
    return ImageOps.pad(
        image, FRAME_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )

def cleanup_all_resources():
    """Giải phóng toàn bộ VRAM, RAM và xóa temp files"""
    global global_pipe
    
    print("🧹 Bắt đầu cleanup toàn bộ...")
    
    # 1. Xóa pipeline khỏi memory (safe)
    try:
        if global_pipe is not None:
            del global_pipe
        global_pipe = None
    except:
        global_pipe = None
    
    # 2. Clear VRAM (chỉ khi có CUDA)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    
    # 3. Force garbage collection  
    gc.collect()
    
    # 4. Xóa ALL temp files sau khi video xong để tránh đầy ổ đĩa
    try:
        outputs_dir = BASE_DIR / "outputs"
        if outputs_dir.exists():
            for file in outputs_dir.glob("*"):
                if file.name not in ["result.png", "output.mp4"]:  # Giữ output cuối
                    try:
                        file.unlink()
                        print(f"  Xóa: {file.name}")
                    except:
                        pass

    except Exception as e:
        print(f"  Lỗi cleanup files: {e}")
    
    # 5. Show memory status
    try:
        mem = psutil.virtual_memory()
        print(f"  RAM available: {mem.available // (1024**3):.1f}GB/{mem.total // (1024**3):.1f}GB")
        
        if torch.cuda.is_available():
            vram_free = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            print(f"  VRAM free: {vram_free // (1024**2):.0f}MB")
    except:
        print("  Không đọc được memory status")
    
    print("✅ Cleanup xong!")


def make_temporal_motion_frame(original_pil, frame_index, total_frames):
    """
    Camera drift cho chế độ '1:1 camera êm' (locked mode, không AI).
    Tăng motion so với trước: zoom 4%, pan 5px, xoay 0.2°
    """
    original_np = np.array(original_pil)
    h, w = original_np.shape[:2]
    progress = frame_index / max(total_frames - 1, 1)
    smooth = progress * progress * (3 - 2 * progress)
    zoom = 1.0 + 0.04 * smooth        # 4% zoom (trước: 1.8%)
    dx = 5.0 * smooth                  # Pan phải 5px (trước: 1.5px)
    dy = -3.0 * smooth                 # Pan lên 3px (trước: -2px)
    angle = 0.2 * smooth               # Xoay nhẹ 0.2° (trước: 0°)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    warped = cv2.warpAffine(
        original_np, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(warped)


def make_cinematic_motion_frame(original_pil, frame_index, total_frames,
                                 zoom_max=0.08, dx_max=10.0, dy_max=-6.0,
                                 angle_max=0.4):
    """
    Camera motion MẠNH cho AI init_video.
    AnimateDiff cần thấy chuyển động rõ ràng trong input để tạo ra video động.
    Hỗ trợ tuỳ chỉnh hướng camera qua tham số.
    """
    original_np = np.array(original_pil)
    h, w = original_np.shape[:2]
    progress = frame_index / max(total_frames - 1, 1)
    # Hermite smooth: ease-in-out
    smooth = progress * progress * (3 - 2 * progress)

    zoom = 1.0 + zoom_max * smooth
    dx = dx_max * smooth
    dy = dy_max * smooth
    angle = angle_max * smooth

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    warped = cv2.warpAffine(
        original_np, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(warped)


# Các preset hướng camera để video không lặp lại cùng kiểu
CAMERA_MOTION_PRESETS = [
    # (zoom_max, dx_max, dy_max, angle_max, description)
    (0.08,  10.0, -6.0,  0.4,  "dolly forward + pan right"),
    (0.06, -8.0,  -4.0, -0.3,  "dolly forward + pan left"),
    (0.10,  0.0, -8.0,   0.0,  "push in straight"),
    (0.05,  12.0,  0.0,  0.5,  "tracking right + rotate"),
    (0.05, -12.0,  0.0, -0.5,  "tracking left + rotate"),
    (0.07,  6.0,   5.0,  0.2,  "dolly forward + tilt down"),
    (0.09,  4.0,  -10.0, 0.3,  "crane up + slight pan"),
]


def make_cinematic_init_video(original_image: Image.Image, num_frames: int):
    """
    Tạo init_video với progressive camera motion thay vì duplicate ảnh tĩnh.

    KEY INSIGHT: AnimateDiff video-to-video cần thấy MOTION trong input video
    để biết hướng chuyển động. Nếu input toàn ảnh tĩnh giống nhau = output cũng tĩnh.

    Mỗi lần gọi sẽ chọn NGẪU NHIÊN một hướng camera từ CAMERA_MOTION_PRESETS
    để mỗi video tạo ra đều khác nhau, không bị lặp kiểu.
    """
    import random
    preset = random.choice(CAMERA_MOTION_PRESETS)
    zoom_max, dx_max, dy_max, angle_max, desc = preset
    print(f"  🎥 Camera preset: {desc} (zoom={zoom_max}, dx={dx_max}, dy={dy_max}, angle={angle_max})")

    frames = []
    for i in range(num_frames):
        frame = make_cinematic_motion_frame(
            original_image, i, num_frames,
            zoom_max=zoom_max, dx_max=dx_max, dy_max=dy_max, angle_max=angle_max,
        )
        frames.append(frame)
    return frames


def crossfade(img_a: Image.Image, img_b: Image.Image, alpha: float) -> Image.Image:
    """Blend 2 ảnh: alpha=0 → img_a, alpha=1 → img_b."""
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    blended = a * (1 - alpha) + b * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def save_frame(image: Image.Image, frame_index: int):
    try:
        FRAMES_DIR.mkdir(exist_ok=True, parents=True)
        path = FRAMES_DIR / f"frame_{frame_index:04d}.jpg"
        image.save(path, quality=95, subsampling=0)
    except FileNotFoundError as e:
        import os
        print(f"LỖI KỲ LẠ KHI LƯU FRAME {frame_index}: {e}")
        print(f" - Thư mục hiện tại (CWD): {os.getcwd()}")
        print(f" - FRAMES_DIR exists?: {FRAMES_DIR.exists()}")
        print(f" - FRAMES_DIR is_dir?: {FRAMES_DIR.is_dir()}")
        print(f" - Đang thử tạo lại bằng os.makedirs...")
        os.makedirs(str(FRAMES_DIR), exist_ok=True)
        image.save(path, quality=95, subsampling=0)
        print(f" - Đã lưu thành công sau khi thử lại!")
    except Exception as e:
        print(f"LỖI LƯU FRAME {frame_index}: {e}")
        raise e

def save_locked_motion_frames(original_image, num_frames):
    for i in range(num_frames):
        frame = make_temporal_motion_frame(original_image, i, num_frames)
        save_frame(frame, i)
        set_progress(current=i + 1, total=num_frames,
                     message=f"Đang tạo camera êm {i + 1}/{num_frames}...")

def save_still_frames(original_image, num_frames):
    for i in range(num_frames):
        save_frame(original_image, i)
        set_progress(current=i + 1, total=num_frames,
                     message=f"Đang tạo frame tĩnh {i + 1}/{num_frames}...")

def clear_ai_memory(torch_module):
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    gc.collect()

# ---------------------------------------------------------------------------
# AnimateDiff render
# ---------------------------------------------------------------------------

CROSSFADE_FRAMES = 4  # số frames dùng để blend ảnh gốc → AI output đầu video
SEGMENT_OVERLAP = 4   # frames cuối segment trước dùng crossfade vào segment sau để nối mượt

def render_with_svd_image2video(original_image: Image.Image, num_frames: int):
    """
    SVD Image-to-Video: Sinh video TRỰC TIẾP từ ảnh gốc
    - Không cần prompt  
    - Video bắt đầu chính xác từ ảnh gốc
    - Chuyển động tự nhiên từ ảnh đó
    """
    print(f"🎬 SVD Image-to-Video: {num_frames} frames")
    
    pipe = get_image_to_video_pipeline()
    
    # SVD yêu cầu ảnh 1024x576 (16:9) hoặc 576x1024 (9:16)
    # Resize image để fit SVD
    svd_image = original_image.resize((576, 1024), Image.LANCZOS)  # Portrait 9:16
    
    with torch.inference_mode():
        # SVD không cần prompt - chỉ cần ảnh!
        import random
        svd_seed = random.randint(0, 999999)
        print(f"  SVD seed: {svd_seed}")
        
        video_frames = pipe(
            image=svd_image,
            num_frames=14,  # SVD cơ bản: 14 frames 
            decode_chunk_size=2,  # Giảm xuống 2 để tiết kiệm VRAM
            num_inference_steps=25,  # Tăng steps cho motion chất lượng hơn (20 → 25)
            generator=torch.Generator(device="cpu").manual_seed(svd_seed),
            motion_bucket_id=200,  # MAX ANIMATION cho SVD (180 → 200)
            noise_aug_strength=0.12,  # Tăng sáng tạo + motion (0.08 → 0.12)
        ).frames[0]  # [0] vì batch_size=1
    
    # Resize frames về kích thước target
    resized_frames = []
    for frame in video_frames:
        frame = frame.resize(FRAME_SIZE, Image.LANCZOS) 
        resized_frames.append(frame)
    
    # Extend/trim để đủ num_frames
    while len(resized_frames) < num_frames:
        resized_frames.append(resized_frames[-1])  # duplicate cuối
    
    return resized_frames[:num_frames]

def render_with_animatediff(original_image: Image.Image, stable_prompt: str, num_frames: int):
    """
    Sinh video với segment overlap:
    - Mỗi segment 16 frames sinh độc lập (giới hạn AnimateDiff)
    - Chỗ nối giữa 2 segment dùng crossfade SEGMENT_OVERLAP frames để mượt
    - Frame 0 = ảnh gốc thật, crossfade vào AI output
    - ControlNet Tile neo spatial layout suốt video
    """
    import torch

    pipe = get_animatediff_pipeline()
    control_tile = make_tile_control_image(original_image)
    all_ai_frames = []
    offset = 0
    seg_idx = 0

    while offset < num_frames:
        seg_idx += 1
        seg_size = min(ANIMATEDIFF_SEGMENT, num_frames - offset)
        total_segs = -(-num_frames // ANIMATEDIFF_SEGMENT)  # ceil

        set_progress(
            message=f"AnimateDiff sinh segment {seg_idx}/{total_segs} ({seg_size} frames)...",
            current=offset, total=num_frames,
        )
        print(f"▶ Segment {seg_idx}/{total_segs}: {seg_size} frames")

        # Seed liên tục + temporal anchor để segment 2 không drift
        if seg_idx == 1:
            # Random seed mỗi lần tạo video
            import random
            base_seed = random.randint(0, 999999)
            generator = torch.Generator(device="cpu").manual_seed(base_seed)
            anchor_frame = original_image  # segment đầu dùng ảnh gốc làm anchor
            print(f"  Segment 1 seed: {base_seed}")
            # ✨ Tạo init video với camera motion progressive (KEY FIX!)
            # AnimateDiff cần thấy MOTION trong input để tạo video động
            init_video = make_cinematic_init_video(original_image, seg_size)
        else:
            # Segment sau: seed liên quan để consistency
            generator = torch.Generator(device="cpu").manual_seed(base_seed + 100)  
            anchor_frame = all_ai_frames[-1]  # frame cuối segment trước
            print(f"  Segment 2 seed: {base_seed + 100}")
            # Temporal continuity: context frames + camera motion tiếp tục
            context_n = min(SEGMENT_OVERLAP, len(all_ai_frames))
            context_frames = all_ai_frames[-context_n:]
            remaining = seg_size - context_n
            continuation_frames = make_cinematic_init_video(anchor_frame, remaining)
            init_video = list(context_frames) + continuation_frames
            print(f"  → Dùng {context_n} context frames + {remaining} motion continuation frames")
        
        # Prepare ControlNet conditioning frames (single Tile, không dùng Canny để tiết kiệm VRAM)
        tile_frames = [control_tile] * seg_size
        
        with torch.inference_mode():
            if USE_CONTROLNET:
                out = pipe(
                    prompt=stable_prompt,
                    negative_prompt=CONSISTENCY_NEGATIVE_PROMPT,
                    video=init_video,
                    strength=0.85, # Giảm nhẹ để giữ motion signal từ init_video (0.92 → 0.85)
                    ip_adapter_image=anchor_frame,
                    conditioning_frames=tile_frames,
                    controlnet_conditioning_scale=0.02,  # Gần 0 để AI tự do tạo chuyển động (0.05 → 0.02)
                    guidance_scale=AI_GUIDANCE_SCALE,
                    num_inference_steps=35 if not TEST_MODE else 30,  # TĂNG steps để vẽ motion tốt hơn (20→30)
                    generator=generator,
                )
            else:
                out = pipe(
                    prompt=stable_prompt,
                    negative_prompt=CONSISTENCY_NEGATIVE_PROMPT,
                    video=init_video,
                    strength=0.85,  # Giữ motion signal từ init_video
                    ip_adapter_image=anchor_frame,
                    guidance_scale=AI_GUIDANCE_SCALE,
                    num_inference_steps=35 if not TEST_MODE else 30,  # TĂNG steps để vẽ motion tốt hơn (20→30)
                    generator=generator,
                )


        seg_frames = out.frames[0]

        # Nếu segment cuối có ít frames hơn dự kiến (do VAE lỗi), duplicate frame cuối
        if seg_idx > 1 and len(seg_frames) < seg_size:
            last_frame = seg_frames[-1] if seg_frames else all_ai_frames[-1]
            while len(seg_frames) < seg_size:
                seg_frames.append(last_frame)
                print(f"  → Duplicate frame cuối để đủ {seg_size} frames")

        if seg_idx == 1:
            # Segment đầu: lấy toàn bộ
            all_ai_frames.extend(seg_frames)
        else:
            # Segment sau: crossfade mượt vùng nối (giữ nguyên tổng frame count)
            blend_n = min(SEGMENT_OVERLAP, len(all_ai_frames), len(seg_frames))
            last_prev = all_ai_frames[-1]   # frame cuối segment trước
            first_next = seg_frames[0]       # frame đầu segment sau

            # Smooth cuối segment trước: blend dần về frame đầu segment sau
            for i in range(blend_n):
                alpha = ((i + 1) / (blend_n + 1)) * 0.5  # nhẹ nhàng, max 40%
                idx = len(all_ai_frames) - blend_n + i
                all_ai_frames[idx] = crossfade(all_ai_frames[idx], first_next, alpha)

            # Smooth đầu segment sau: blend dần từ frame cuối segment trước
            for i in range(blend_n):
                alpha = (1.0 - (i + 1) / (blend_n + 1)) * 0.5  # giảm dần
                seg_frames[i] = crossfade(seg_frames[i], last_prev, alpha)

            all_ai_frames.extend(seg_frames)
            print(f"  → Crossfade {blend_n} frames ở chỗ nối segment")

        offset += seg_size
        set_progress(current=offset, total=num_frames,
                     message=f"Xong segment {seg_idx}/{total_segs} ({offset}/{num_frames} frames)")
        clear_ai_memory(torch)
        print(f"  ✓ {offset}/{num_frames} frames xong")

    # Đảm bảo đủ frames, nếu thiếu thì duplicate frame cuối cùng 3 lần để ổn định
    while len(all_ai_frames) < num_frames:
        if len(all_ai_frames) > 0:
            all_ai_frames.append(all_ai_frames[-1])  # duplicate frame cuối
        else:
            all_ai_frames.append(original_image)  # fallback về ảnh gốc
    
    # 🎯 Blend nhẹ frame đầu để giữ motion (không force cứng)
    if len(all_ai_frames) > 0:
        original_resized = original_image.resize(FRAME_SIZE, Image.LANCZOS)
        # Chỉ blend nhẹ frame 0: 40% ảnh gốc, giữ 60% AI motion
        # KHÔNG force 100% vì sẽ giết chết chuyển động ngay từ đầu
        all_ai_frames[0] = crossfade(all_ai_frames[0], original_resized, 0.4)
        print(f"  ✓ Blend frame 1 với 40% ảnh gốc (giữ motion)")
    
    return all_ai_frames[:num_frames]

# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------

def build_output_video():
    # Tạo video với timestamp để tránh browser cache
    import time
    timestamp = int(time.time())
    output_filename = f"output_{timestamp}.mp4"
    output_mp4 = STATIC_DIR / output_filename
    
    # Xóa videos cũ trong static/
    for old_video in STATIC_DIR.glob("output_*.mp4"):
        try:
            old_video.unlink()
            print(f"  Xóa video cũ: {old_video.name}")
        except:
            pass
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(OUTPUT_FPS),
        "-i", f"{FRAMES_DIR}/frame_%04d.jpg",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_mp4),
    ]
    try:
        subprocess.run(
            ffmpeg_cmd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        return f"/static/{output_filename}"
    except subprocess.CalledProcessError as e:
        print(e.stderr[-2000:])
        raise RuntimeError("FFmpeg không ghép được video.")

# ---------------------------------------------------------------------------
# Main render orchestrator
# ---------------------------------------------------------------------------

def render_uploaded_video(file_storage, render_mode):
    if not render_lock.acquire(blocking=False):
        msg = "Đang có một video khác được render. Đợi job đó xong rồi bấm tạo lại."
        set_progress(running=True, done=False, message=msg, error=msg)
        return None, msg

    set_progress(
        running=True, done=False, mode=render_mode,
        current=0, total=TEST_FRAMES,
        message="Đang đọc ảnh đầu vào...",
        error=None, result_video=None,
    )

    try:
        image = Image.open(file_storage).convert("RGB")
        image = fit_to_model_frame(image)

        temp_img_path = OUTPUT_DIR / "temp_input.jpg"
        image.save(temp_img_path)

        clear_frames()
        num_frames = TEST_FRAMES
        original_image = image.copy()

        # ── AI AnimateDiff + ControlNet hoặc SVD ──────────────────────────
        if render_mode in ["redraw", "svd"]:
            set_progress(message="Đang kiểm tra Torch CUDA...", current=0, total=num_frames)
            has_cuda, cuda_msg = get_torch_cuda_runtime_hint()
            if not has_cuda:
                set_progress(running=False, done=True, error=cuda_msg, message=cuda_msg,
                             current=0, total=num_frames)
                return None, cuda_msg

            import torch
            if not torch.cuda.is_available():
                msg = (
                    "Torch có CUDA runtime nhưng không thấy GPU CUDA khả dụng. "
                    "AI redraw sẽ rất chậm, nên app đã dừng sớm."
                )
                set_progress(running=False, done=True, error=msg, message=msg,
                             current=0, total=num_frames)
                return None, msg

            if render_mode == "svd":
                # SVD Image-to-Video: Không cần prompt, AI tự do sáng tạo tương lai
                set_progress(message="SVD đang phân tích ảnh...", current=0, total=num_frames)
                print("SVD Image-to-Video: Không cần prompt")
                stable_prompt = ""
            else:
                # Gemini phân tích chi tiết từng vật thể cho AnimateDiff
                set_progress(message="Gemini phân tích chi tiết ảnh...", current=0, total=num_frames)
                stable_prompt = get_stable_motion_prompt(temp_img_path)
                
                # FORCE THÊM motion keywords vào prompt
                stable_prompt += (
                    ", cinematic camera slowly dollying forward into the scene, "
                    "smooth continuous camera movement, objects gently swaying, "
                    "dynamic parallax depth with foreground shifting, "
                    "3D perspective change, atmospheric particles floating in air, "
                    "subtle environmental motion like wind and light shifts, "
                    "professional cinematic tracking shot, fluid motion"
                )
                
                print(f"🎯 Enhanced prompt: {stable_prompt}")
                print(f"🎯 Tạo video mới với random seed...")

            cn_status = "SVD" if render_mode == "svd" else ("ControlNet Tile" if USE_CONTROLNET else "AnimateDiff")
            set_progress(
                message=f"Đang nạp {cn_status} pipeline...",
                current=0, total=num_frames,
            )

            if render_mode == "svd":
                # SVD Image-to-Video: Bắt đầu từ ảnh gốc, tự tưởng tượng phần sau
                frames = render_with_svd_image2video(original_image, num_frames)
            else:
                # AnimateDiff Video-to-Video: Giữ chặt layout
                frames = render_with_animatediff(original_image, stable_prompt, num_frames)

            for i, frame in enumerate(frames):
                save_frame(frame, i)

            clear_ai_memory(torch)

        # ── Still ──────────────────────────────────────────────────────────
        elif render_mode == "still":
            save_still_frames(original_image, num_frames)
            gc.collect()

        # ── Locked camera drift ────────────────────────────────────────────
        else:
            save_locked_motion_frames(original_image, num_frames)
            gc.collect()

        set_progress(message="Đang ghép video MP4...", current=num_frames, total=num_frames)
        result_video = build_output_video()
        set_progress(
            running=False, done=True,
            current=num_frames, total=num_frames,
            message="Render xong!", error=None, result_video=result_video,
        )
        return result_video, None

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Lỗi render (full traceback):\n{tb}")
        msg = str(e) or f"{type(e).__name__}: (lỗi không có thông tin)"
        set_progress(running=False, done=True, message=f"Lỗi: {msg[:80]}", error=msg, result_video=None)
        return None, msg
    finally:
        render_lock.release()
        # Cleanup ngay lập tức sau khi render xong (không cần thread chờ)
        cleanup_all_resources()

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result_video = None
    error_message = None
    if request.method == "POST":
        result_video, error_message = render_uploaded_video(
            request.files["image"],
            request.form.get("render_mode", "locked"),
        )
    return render_template(
        "index.html",
        result_video=result_video,
        latent_info=None,
        error_message=error_message,
        device=DEVICE_LABEL,
    )

@app.route("/generate", methods=["POST"])
def generate():
    """
    Trả về ngay (không đợi render xong) để tránh browser timeout.
    Frontend sẽ poll /progress cho đến khi done=True rồi gọi /result.
    """
    # Đọc file và form data trước khi vào thread (request context không thread-safe)
    file_bytes = request.files["image"].read()
    render_mode = request.form.get("render_mode", "locked")

    import io
    def run():
        render_uploaded_video(io.BytesIO(file_bytes), render_mode)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({"ok": True, "started": True})


@app.route("/result", methods=["GET"])
def result():
    """Trả về kết quả sau khi render xong."""
    snap = get_progress_snapshot()
    if not snap["done"]:
        return jsonify({"ok": False, "error": "Chưa xong"})
    if snap["error"]:
        return jsonify({"ok": False, "error": snap["error"]})
    return jsonify({"ok": True, "result_video": snap["result_video"]})

@app.route("/cleanup", methods=["POST"])
def cleanup_endpoint():
    """API endpoint để cleanup thủ công"""
    cleanup_all_resources()
    return jsonify({"ok": True, "message": "Cleanup completed"})

@app.route("/progress", methods=["GET"])
def progress():
    return jsonify(get_progress_snapshot())


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)
