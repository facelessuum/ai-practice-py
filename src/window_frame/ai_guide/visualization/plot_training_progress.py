"""Optional TensorBoard dashboard and always-available standalone SVG curves."""
from html import escape


def open_dashboard(folder, enabled=True):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise RuntimeError("Install the dashboard extra, or set tensorboard=False in TrainingConfig in utils/settings.py") from None
    return SummaryWriter(str(folder))


def record_dashboard(writer, epoch, values, prefix=""):
    if writer is None:
        return
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            record_dashboard(writer, epoch, value, name)
        elif isinstance(value, (int, float)):
            writer.add_scalar(name, value, epoch)
    writer.flush()


def plot_training_progress(history, path):
    series = {"training loss": [row["train_loss"] for row in history],
              "validation loss": [row["validation_loss"] for row in history]}
    maximum = max(1e-8, max(max(values) for values in series.values()))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="50" y="25">Loss by epoch (lower is better)</text>',
           '<path d="M50 50V350H770" fill="none" stroke="black"/>']
    for (name, values), color in zip(series.items(), ("#147ab8", "#cf5726")):
        points = " ".join(f"{50 + i * 720 / max(1,len(values)-1):.1f},{350 - v/maximum*290:.1f}" for i, v in enumerate(values))
        svg += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<text x="{50 if color == "#147ab8" else 270}" y="385" fill="{color}">{escape(name)}</text>']
    svg += [f'<text x="5" y="60">{maximum:.2f}</text>', '</svg>']
    path.write_text("\n".join(svg))
