"""
Thumbnail generator for Instagram event images.

Sends the post image to OpenAI vision and asks it to identify the best crop
region for a landscape thumbnail, then uses Pillow to produce the crop.
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

import config


class _CropBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0, description="Left edge as fraction of image width.")
    y: float = Field(ge=0.0, le=1.0, description="Top edge as fraction of image height.")
    w: float = Field(ge=0.0, le=1.0, description="Crop width as fraction of image width.")
    h: float = Field(ge=0.0, le=1.0, description="Crop height as fraction of image height.")
    reason: str = Field(description="One sentence explaining the choice.")


_SYSTEM_PROMPT = (
    "You are a visual editor choosing a thumbnail crop for a university event listing. "
    "Given an image and the event title, identify the rectangular region that best "
    "represents the event — prefer areas with people, activity, or strong visual context "
    "over empty space, text banners, or logos. "
    "Return x, y, w, h as fractions (0.0–1.0) of the image dimensions, aiming for a "
    "landscape crop (wider than tall). The crop must stay within the image bounds "
    "(x+w <= 1.0, y+h <= 1.0)."
)

_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")


def generate_thumbnail(
    image_path: str,
    event_title: str,
    api_key: str,
    model: str = config.OPENAI_MODEL,
) -> Optional[str]:
    """
    Crop a thumbnail from *image_path* using AI-guided region selection.
    Saves the result to data/images/ and returns its /api/images/<name> URL,
    or None if anything fails.
    """
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode()
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        client = OpenAI(api_key=api_key)
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'Event title: "{event_title}"\nChoose the best thumbnail crop.',
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
                        },
                    ],
                },
            ],
            response_format=_CropBox,
            temperature=0,
            max_tokens=256,
        )
        crop = response.choices[0].message.parsed
        if crop is None:
            return None

        x_frac = max(0.0, min(crop.x, 1.0))
        y_frac = max(0.0, min(crop.y, 1.0))
        w_frac = max(0.05, min(crop.w, 1.0 - x_frac))
        h_frac = max(0.05, min(crop.h, 1.0 - y_frac))

        img = Image.open(image_path)
        iw, ih = img.size
        left   = int(x_frac * iw)
        top    = int(y_frac * ih)
        right  = int((x_frac + w_frac) * iw)
        bottom = int((y_frac + h_frac) * ih)
        cropped = img.crop((left, top, right, bottom))

        os.makedirs(_IMAGES_DIR, exist_ok=True)
        slug = hashlib.sha1(image_path.encode()).hexdigest()[:12]
        filename = f"thumb_{slug}.jpg"
        cropped.convert("RGB").save(os.path.join(_IMAGES_DIR, filename), "JPEG", quality=88)

        return f"/api/images/{filename}"

    except Exception as e:
        print(f"[thumbnail] generation failed for {image_path!r}: {e}")
        return None
