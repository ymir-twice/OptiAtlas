# engine/utils/img_covt.py

import base64
from io import BytesIO
from PIL import Image


def b64_to_pil(image_b64: str) -> Image.Image:
    """
    将 base64 字符串转换为 PIL.Image 对象。

    支持：
    1. 纯 base64 字符串
    2. data URL 格式，例如：data:image/png;base64,xxxx
    """
    # 如果是 data URL，去掉前缀
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes))

    return image


def pil_to_b64(image: Image.Image, image_format: str = "PNG") -> str:
    """
    将 PIL.Image 对象转换为 base64 字符串。

    Args:
        image: PIL.Image 对象
        image_format: 输出格式，例如 "PNG", "JPEG", "WEBP"

    Returns:
        base64 字符串，不包含 data:image/...;base64, 前缀
    """
    buffer = BytesIO()

    # JPEG 不支持透明通道，需要转换为 RGB
    if image_format.upper() in ["JPEG", "JPG"] and image.mode in ["RGBA", "LA", "P"]:
        image = image.convert("RGB")

    image.save(buffer, format=image_format)
    image_bytes = buffer.getvalue()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_b64

def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image