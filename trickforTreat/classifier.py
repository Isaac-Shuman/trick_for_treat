"""
trickforTreat/classifier.py  (PSEUDOCODE, shared module)

Posture classifier model + eval preprocessing + inference wrapper, shared by:
  - scripts/05_train_classifier.py  (build_model, build_eval_transform, save/load)
  - later: live loop (WATCH state) -> PostureClassifier.load("checkpoints/final.pt").predict(crop)
Training-only code (loops, CV, samplers, train augmentations) stays out of this file.
"""

# import timm, torch
# from torchvision import transforms as T


def build_model(clf_cfg, num_classes: int) -> "torch.nn.Module":
    """
    timm.create_model(clf_cfg.backbone, pretrained=clf_cfg.pretrained,
                      num_classes=num_classes, drop_rate=clf_cfg.dropout)
    Also expose a helper to split params into backbone vs head (model.get_classifier())
    for differential LR and freezing.
    """


def build_eval_transform(clf_cfg) -> "Callable":
    """Resize((input_size, input_size)) -> ToTensor -> Normalize(mean, std).
    Crops from detector.crop_with_fill are already square, so there is no center crop: nothing is cut off.
    assert input image is square (within 1 px), since a non-square input means the crop step was skipped."""


class PostureClassifier:
    """
    __init__(model, transform, classes, reward_thresholds, device)
    load(path, device) -> PostureClassifier
        ckpt = torch.load(path, map_location=device)
        rebuild model from ckpt["backbone"], len(ckpt["classes"]); load_state_dict(strict=True)
        rebuild transform from ckpt["input_size"], ckpt["normalize"]
    save(path, model, classes, clf_cfg, reward_thresholds, extra)
    predict(crop: PIL.Image) -> dict
        returns {"label": str, "probs": {class: float}, "latency_ms": {"pre", "fwd", "post"}}
    predict_batch(crops) -> list[dict]
    should_reward(pred, commanded: str) -> bool
        argmax plus threshold:
        pred["label"] == commanded and pred["probs"][commanded] >= reward_thresholds[commanded]
        (reward_thresholds come from 05's CV threshold sweep and are stored in the checkpoint)
        (per-frame decision only; the temporal gate aggregates these later)
    """


# DEBUGGING: `python -m trickforTreat.classifier checkpoints/final.pt <crop.jpg>` (TODO: __main__)
# prints label + probs. Sanity: probs sum to 1; classes match data/classes.json.
