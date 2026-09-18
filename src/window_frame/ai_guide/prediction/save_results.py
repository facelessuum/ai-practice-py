"""Export lossless results and link every artifact to its source checkpoint."""
import numpy as np
from PIL import Image
from ..model.window_frame_model import CLASS_NAMES
from ..run_history.record_run_details import write_json, file_hash
from ..visualization.compare_images import to_image, compare_images
from ..visualization.draw_rail_guide import draw_rail_guide, confidence_image


def save_results(folder, original, result, metadata):
    folder.mkdir(parents=True, exist_ok=False)
    soft = result["soft_mask"][0, 0].detach().cpu().numpy()
    probabilities = result["probabilities"][0].detach().cpu().numpy()
    confidence = probabilities.max(0)
    labels = probabilities.argmax(0).astype(np.uint8)
    gate = result["edit_gate"][0, 0].detach().cpu().numpy()
    mask = soft >= 0.5
    enhanced = to_image(result["enhanced"][0])
    original.save(folder / "input.png")
    enhanced.save(folder / "enhanced.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "frame_mask.png")
    Image.fromarray(np.round(soft * 255).astype(np.uint8)).save(folder / "soft_mask.png")
    Image.fromarray(labels).save(folder / "rail_classes.png")
    np.save(folder / "soft_mask.npy", soft)
    np.save(folder / "class_probabilities.npy", probabilities)
    np.save(folder / "edit_gate.npy", gate)
    draw_rail_guide(original, labels).save(folder / "rail_guide.png")
    confidence_image(confidence).save(folder / "class_confidence.png")
    compare_images({"Input": original, "Enhanced": enhanced}, folder / "comparison.png")
    changed = np.any(np.asarray(original) != np.asarray(enhanced), axis=-1)
    # Numerical certainty, not a calibrated probability that the whole output is correct.
    metadata.update(frame_detected=bool(mask.any()), frame_area_fraction=float(mask.mean()),
        edited_area_fraction=float(changed.mean()), allowed_edit_fraction=float((gate > 0).mean()),
        mean_frame_probability=float(soft.mean()),
        mean_class_confidence_on_frames=float(confidence[mask].mean()) if mask.any() else None,
        classes={name: dict(pixels=int((labels == i).sum()),
            mean_confidence=float(confidence[labels == i].mean()) if (labels == i).any() else None)
            for i, name in enumerate(CLASS_NAMES)},
        confidence_note="Uncalibrated pixel confidence; not an enhancement-quality score or image-level presence probability.",
        warnings=["Predicted protection can fail when classification is wrong.",
                  "Tiled inference limits scene context; resized inference can lose thin rails. Validate quality on large photos."],
        artifacts={p.name: dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in sorted(folder.iterdir()) if p.is_file()})
    write_json(folder / "metadata.json", metadata)
