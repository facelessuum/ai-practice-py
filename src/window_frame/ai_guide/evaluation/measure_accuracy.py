"""Accumulate counts, not averages of batch averages. Undefined scores are null."""
import torch
from torch.nn import functional as F
from ..model.window_frame_model import CLASS_NAMES


class AccuracyMeter:
    def __init__(self):
        self.confusion = torch.zeros(6, 6, dtype=torch.float64)
        self.counts = {k: 0.0 for k in ("intersection", "union", "pred", "truth", "frame_error",
            "input_error", "frame_values", "outside_error", "outside_values", "protected_error",
            "protected_values", "empty_images", "empty_false_positives", "boundary_match_pred",
            "boundary_match_truth", "boundary_pred", "boundary_truth", "brier", "valid_pixels")}
        self.bins = torch.zeros(10, 3, dtype=torch.float64)

    @torch.no_grad()
    def update(self, result, batch):
        labels = result["probabilities"].argmax(1)
        truth = batch["classes"]
        valid = truth != 255
        counts = torch.bincount((truth[valid] * 6 + labels[valid]).flatten(), minlength=36)
        self.confusion += counts.reshape(6, 6).cpu()
        pred, mask = result["soft_mask"] >= 0.5, batch["mask"] >= 0.5
        c = self.counts
        for name, value in [("intersection", pred & mask), ("union", pred | mask), ("pred", pred), ("truth", mask)]:
            c[name] += value.sum().item()
        empty = ~mask.flatten(1).any(1)
        c["empty_images"] += empty.sum().item()
        c["empty_false_positives"] += (pred.flatten(1).any(1) & empty).sum().item()
        def boundary(x):
            x = x.float()
            return (F.max_pool2d(x, 3, 1, 1) - (-F.max_pool2d(-x, 3, 1, 1))) > 0
        pb, tb = boundary(pred), boundary(mask)
        c["boundary_pred"] += pb.sum().item()
        c["boundary_truth"] += tb.sum().item()
        c["boundary_match_pred"] += (pb & (F.max_pool2d(tb.float(), 3, 1, 1) > 0)).sum().item()
        c["boundary_match_truth"] += (tb & (F.max_pool2d(pb.float(), 3, 1, 1) > 0)).sum().item()
        for prefix, region, difference in [
            ("frame", ((truth >= 1) & (truth <= 4)).unsqueeze(1), (result["enhanced"] - batch["target"]).abs()),
            ("outside", ~mask, (result["enhanced"] - batch["image"]).abs()),
            ("protected", (truth == 5).unsqueeze(1), (result["enhanced"] - batch["image"]).abs())]:
            region = region.expand_as(difference)
            c[prefix + "_error"] += difference[region].sum().item()
            c[prefix + "_values"] += region.sum().item()
            if prefix == "frame":
                c["input_error"] += (batch["image"] - batch["target"]).abs()[region].sum().item()
        probability = result["probabilities"].permute(0, 2, 3, 1)[valid]
        if probability.numel():
            onehot = F.one_hot(truth[valid], 6)
            c["brier"] += ((probability - onehot) ** 2).sum().item()
            c["valid_pixels"] += valid.sum().item()
            confidence, predicted = probability.max(1)
            correct = predicted == truth[valid]
            for i in range(10):
                select = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
                self.bins[i] += torch.tensor([select.sum().item(), confidence[select].sum().item(), correct[select].sum().item()])

    def compute(self):
        c = self.counts
        ratio = lambda a, b: a / b if b else None
        scores = dict(frame_iou=ratio(c["intersection"], c["union"]),
                      frame_dice=ratio(2*c["intersection"], c["pred"] + c["truth"]),
                      no_frame_false_positive_rate=ratio(c["empty_false_positives"], c["empty_images"]))
        scores["boundary_precision"] = ratio(c["boundary_match_pred"], c["boundary_pred"])
        scores["boundary_recall"] = ratio(c["boundary_match_truth"], c["boundary_truth"])
        for prefix in ("frame", "outside", "protected"):
            scores[prefix + "_mae"] = ratio(c[prefix + "_error"], c[prefix + "_values"])
        scores["unchanged_input_frame_mae"] = ratio(c["input_error"], c["frame_values"])
        scores["frame_improvement"] = ratio(c["input_error"] - c["frame_error"], c["frame_values"])
        scores["class_iou"] = {name: ratio(self.confusion[i, i].item(),
            (self.confusion[i].sum() + self.confusion[:, i].sum() - self.confusion[i, i]).item())
            for i, name in enumerate(CLASS_NAMES)}
        scores["class_brier_score"] = ratio(c["brier"], c["valid_pixels"])
        scores["confidence_calibration_error"] = ratio(sum(abs(row[1] - row[2]).item() for row in self.bins), c["valid_pixels"])
        scores["confidence_bins"] = self.bins.tolist()
        scores["confusion_matrix"] = self.confusion.tolist()
        return scores
