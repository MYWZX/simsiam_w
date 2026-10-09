"""Text-aware horizontal mirror augmentation for image-text IQA datasets."""

from __future__ import annotations

import re
from typing import Dict, Tuple

from PIL import Image


_ENGLISH_DIRECTION_MAP = {
    "top left": "top right",
    "top right": "top left",
    "bottom left": "bottom right",
    "bottom right": "bottom left",
    "upper left": "upper right",
    "upper right": "upper left",
    "lower left": "lower right",
    "lower right": "lower left",
    "left hand": "right hand",
    "right hand": "left hand",
    "left side": "right side",
    "right side": "left side",
    "left": "right",
    "right": "left",
    "west": "east",
    "east": "west",
}
_ENGLISH_DIRECTION_RE = re.compile(
    r"\b(?:top[\s_-]+left|top[\s_-]+right|bottom[\s_-]+left|bottom[\s_-]+right|"
    r"upper[\s_-]+left|upper[\s_-]+right|lower[\s_-]+left|lower[\s_-]+right|"
    r"left[\s_-]+hand|right[\s_-]+hand|left[\s_-]+side|right[\s_-]+side|"
    r"left|right|west|east)\b",
    flags=re.IGNORECASE,
)
_CHINESE_DIRECTION_MAP = {
    "左上角": "右上角",
    "右上角": "左上角",
    "左下角": "右下角",
    "右下角": "左下角",
    "左手边": "右手边",
    "右手边": "左手边",
    "左侧": "右侧",
    "右侧": "左侧",
    "左边": "右边",
    "右边": "左边",
    "左方": "右方",
    "右方": "左方",
    "左": "右",
    "右": "左",
}
_CHINESE_DIRECTION_RE = re.compile(
    "|".join(re.escape(key) for key in sorted(_CHINESE_DIRECTION_MAP, key=len, reverse=True))
)


def _preserve_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper() and source[1:].islower():
        return target[:1].upper() + target[1:]
    return target


def swap_direction_words(text: str) -> str:
    """Swap horizontal direction words while preserving punctuation and case."""
    text = str(text)

    def replace_english(match: re.Match[str]) -> str:
        source = match.group(0)
        key = re.sub(r"[_-]+", " ", source.lower())
        target = _ENGLISH_DIRECTION_MAP[key]
        if "-" in source:
            target = target.replace(" ", "-")
        elif "_" in source:
            target = target.replace(" ", "_")
        return _preserve_case(source, target)

    text = _ENGLISH_DIRECTION_RE.sub(replace_english, text)
    return _CHINESE_DIRECTION_RE.sub(
        lambda match: _CHINESE_DIRECTION_MAP[match.group(0)], text
    )


def has_direction_words(text: str) -> bool:
    text = str(text)
    return bool(_ENGLISH_DIRECTION_RE.search(text) or _CHINESE_DIRECTION_RE.search(text))


def resolve_mirror_config(cfg: Dict | None, train: bool) -> Dict:
    """Normalize a mirror_augmentation block and keep eval deterministic."""
    raw = dict(cfg or {})
    enabled = bool(raw.get("enabled", False)) and bool(train)
    probability = float(raw.get("probability", raw.get("prob", 0.5)))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"mirror_augmentation.probability must be in [0, 1], got {probability}")
    return {
        "enabled": enabled,
        "probability": probability,
        "directional_only": bool(raw.get("directional_only", True)),
        "swap_text": bool(raw.get("swap_text", True)),
        # Existing random flips are disabled when this augmentation is enabled;
        # otherwise a text containing "left" could be paired with an unswapped flip.
        "disable_random_horizontal_flip": bool(
            raw.get("disable_random_horizontal_flip", True)
        ),
    }


def maybe_mirror_sample(
    pil: Image.Image,
    prompt: str,
    answer: str,
    mirror_cfg: Dict,
) -> Tuple[Image.Image, str, str, bool]:
    """Mirror one sample and swap horizontal words in its text when selected."""
    if not mirror_cfg.get("enabled", False):
        return pil, str(prompt), str(answer), False
    if mirror_cfg.get("directional_only", True) and not (
        has_direction_words(prompt) or has_direction_words(answer)
    ):
        return pil, str(prompt), str(answer), False
    import random

    if random.random() >= float(mirror_cfg["probability"]):
        return pil, str(prompt), str(answer), False
    prompt = swap_direction_words(prompt) if mirror_cfg.get("swap_text", True) else str(prompt)
    answer = swap_direction_words(answer) if mirror_cfg.get("swap_text", True) else str(answer)
    return pil.transpose(Image.FLIP_LEFT_RIGHT), prompt, answer, True


def mirror_pil_if_applied(pil: Image.Image, applied: bool) -> Image.Image:
    return pil.transpose(Image.FLIP_LEFT_RIGHT) if applied else pil
