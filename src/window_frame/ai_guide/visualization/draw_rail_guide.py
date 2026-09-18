"""Human-readable views; raw IDs are saved separately."""
import numpy as np
from PIL import Image, ImageDraw
from ..model.window_frame_model import CLASS_NAMES

COLORS = np.array([[0, 0, 0], [0, 220, 230], [255, 220, 0], [245, 140, 40],
                   [170, 80, 240], [30, 220, 90]], dtype=np.uint8)


def draw_rail_guide(image, classes):
    source = np.asarray(image.convert("RGB"))
    colored = source.copy()
    selected = classes > 0
    colored[selected] = (0.55 * source[selected] + 0.45 * COLORS[classes[selected]]).astype(np.uint8)
    overlay = Image.fromarray(colored)
    # Keep the image overlay unobstructed; put the legend below it.
    canvas = Image.new("RGB", (max(image.width, 260), image.height + 120), "white")
    canvas.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(CLASS_NAMES[1:], 1):
        y = image.height + 5 + (i - 1) * 22
        draw.rectangle((6, y, 20, y + 14), fill=tuple(COLORS[i]))
        draw.text((28, y), f"{i}: {name}", fill="black")
    return canvas


def confidence_image(values):
    values = np.clip(values, 0, 1)
    return Image.fromarray(np.stack((255 * (1-values), 255 * values, np.zeros_like(values)), -1).astype(np.uint8))
