"""Save labeled image grids without requiring a plotting library."""
import numpy as np
from PIL import Image, ImageDraw


def to_image(tensor):
    array = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = (array * 255).round().astype(np.uint8)
    return Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array)


def compare_images(images, path):
    """images is a mapping of title to PIL image."""
    width = max(image.width for image in images.values())
    height = max(image.height for image in images.values())
    canvas = Image.new("RGB", (width * len(images), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, image) in enumerate(images.items()):
        draw.text((i * width + 5, 5), title, fill="black")
        canvas.paste(image.convert("RGB"), (i * width, 24))
    canvas.save(path)
    return canvas
