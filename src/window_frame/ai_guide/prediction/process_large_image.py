"""Process native-resolution crops, keeping full-image buffers off the GPU."""
import torch
from torch.nn import functional as F


def tile_starts(length, tile_size, overlap):
    """Cover any positive axis length, not just standard photo dimensions."""
    if length < 1:
        raise ValueError("Image dimensions must be positive")
    if tile_size < 16 or not 0 <= overlap < tile_size:
        raise ValueError("tile_size must be >=16; overlap must be >=0 and smaller than tile_size")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.inference_mode()
def process_large_image(model, original, tile_size, overlap, device):
    """Blend crop probabilities/corrections before making final edit decisions.

    original is a CPU tensor [1, 3, H, W]. GPU working memory depends on
    tile_size, not photo dimensions. CPU memory still grows with photo size.
    """
    height, width = original.shape[-2:]
    rows = tile_starts(height, tile_size, overlap)
    columns = tile_starts(width, tile_size, overlap)
    # Six class probabilities, one frame probability, three RGB corrections.
    combined = torch.zeros(1, 10, height, width, dtype=torch.float32)
    weights = torch.zeros(1, 1, height, width, dtype=torch.float32)
    ramp = torch.hann_window(tile_size, periodic=False).clamp_min(0.05)
    window = (ramp[:, None] * ramp[None, :])[None, None]
    for top in rows:
        for left in columns:
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            crop = original[:, :, top:bottom, left:right]
            crop_height, crop_width = crop.shape[-2:]
            # The network accepts rectangles and odd dimensions directly.
            # Only extremely small sides need padding for its downsampling layers.
            padded = F.pad(crop, (0, max(0, 16 - crop_width), 0, max(0, 16 - crop_height)), mode="replicate")
            prediction = model(padded.to(device))
            values = torch.cat((prediction["class_logits"].float().softmax(1),
                                prediction["mask_logits"].float().sigmoid(),
                                prediction["residual"].float()), dim=1)
            values = values[:, :, :crop_height, :crop_width].cpu()
            weight = window[:, :, :crop_height, :crop_width]
            combined[:, :, top:bottom, left:right] += values * weight
            weights[:, :, top:bottom, left:right] += weight
            del prediction, values, padded
    combined.div_(weights)
    return combined[:, :6], combined[:, 6:7], combined[:, 7:]
