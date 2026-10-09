from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from scipy.io import loadmat
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler

from utils.mirror_augmentation import (
    maybe_mirror_sample,
    mirror_pil_if_applied,
    resolve_mirror_config,
)


class PretrainImageTransform:
    """Mild transform used by cross-modal SimSiam pretraining."""

    def __init__(
        self,
        input_size: int = 224,
        train: bool = True,
        quality_safe: bool = False,
        horizontal_flip_prob: float = 0.5,
    ):
        self.input_size = int(input_size)
        self.train = bool(train)
        self.quality_safe = bool(quality_safe)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError(
                "horizontal_flip_prob must be in [0, 1], "
                f"got {self.horizontal_flip_prob}"
            )
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _random_resized_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(0.80, 1.00)
        cw, ch = max(8, int(w * scale)), max(8, int(h * scale))
        if cw < w or ch < h:
            x0 = random.randint(0, max(0, w - cw))
            y0 = random.randint(0, max(0, h - ch))
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
        return img.resize((self.input_size, self.input_size), Image.BILINEAR)

    def _jitter(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.3:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.95, 1.05))
        return img

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.convert("RGB")
        if self.train:
            if self.quality_safe:
                img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
            else:
                img = self._random_resized_crop(img)
            if random.random() < self.horizontal_flip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if not self.quality_safe:
                img = self._jitter(img)
        else:
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def with_mask(self, pil_img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply identical crop/flip geometry to an image and its spatial label."""
        img = pil_img.convert("RGB")
        mask = mask.convert("L")
        if self.train:
            w, h = img.size
            scale = 1.0 if self.quality_safe else random.uniform(0.80, 1.00)
            cw, ch = max(8, int(w * scale)), max(8, int(h * scale))
            x0 = random.randint(0, max(0, w - cw))
            y0 = random.randint(0, max(0, h - ch))
            box = (x0, y0, x0 + cw, y0 + ch)
            img = img.crop(box)

            mw, mh = mask.size
            mask_box = (
                int(round(x0 * mw / max(1, w))),
                int(round(y0 * mh / max(1, h))),
                int(round((x0 + cw) * mw / max(1, w))),
                int(round((y0 + ch) * mh / max(1, h))),
            )
            mask = mask.crop(mask_box)
            if random.random() < self.horizontal_flip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if not self.quality_safe:
                img = self._jitter(img)

        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_arr = np.asarray(img).astype(np.float32) / 255.0
        image_arr = (image_arr - self.mean) / self.std
        image_arr = np.transpose(image_arr, (2, 0, 1))
        mask_arr = np.asarray(mask).astype(np.float32) / 255.0
        return torch.from_numpy(image_arr), torch.from_numpy(mask_arr).unsqueeze(0).clamp(0.0, 1.0)


_SUPPORTED_SUPERVISION_VIEWS = {
    "augmented", "clean", "quality_safe", "three_view", "separate_clean"
}


def _build_image_views(
    pil: Image.Image,
    transform: PretrainImageTransform,
    clean_transform: PretrainImageTransform,
    quality_safe_transform: PretrainImageTransform,
    supervision_view: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Create canonical SimSiam views plus an optional clean label view."""
    if supervision_view not in _SUPPORTED_SUPERVISION_VIEWS:
        raise ValueError(
            f"Unknown supervision_view={supervision_view!r}. "
            f"Choose from: {sorted(_SUPPORTED_SUPERVISION_VIEWS)}"
        )
    if supervision_view in {"three_view", "separate_clean"}:
        return (
            quality_safe_transform(pil),
            quality_safe_transform(pil),
            clean_transform(pil),
        )
    if supervision_view == "clean":
        return clean_transform(pil), transform(pil), None
    if supervision_view == "quality_safe":
        return quality_safe_transform(pil), quality_safe_transform(pil), None
    return transform(pil), transform(pil), None


def _pretrain_mirror_setup(cfg: Dict, train: bool) -> Tuple[Dict, float]:
    mirror_cfg = resolve_mirror_config(cfg.get("mirror_augmentation", {}), train)
    flip_prob = (
        0.0
        if mirror_cfg["enabled"] and mirror_cfg["disable_random_horizontal_flip"]
        else 0.5
    )
    return mirror_cfg, flip_prob


def _read_pil(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to read image: {path}, error={e}")


def _read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc).fillna("")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path}: {last_err}")


def _norm_score(x: float, max_value: float = 5.0, min_value: float = 0.0) -> float:
    return float(np.clip((float(x) - min_value) / (max_value - min_value + 1e-8), 0.0, 1.0))


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_prompt_group(text: str) -> str:
    text = str(text).lower().strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "__empty_prompt__"


def _safe_float01(value, default_value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(np.clip(default_value, 0.0, 1.0))
    if not np.isfinite(v):
        return float(np.clip(default_value, 0.0, 1.0))
    return float(np.clip(v, 0.0, 1.0))


def _answer_confidence_heuristic(
    answer: str,
    status: str = "",
    min_words: int = 8,
    max_words: int = 120,
    bad_patterns: List[str] | None = None,
) -> float:
    status = str(status).strip().lower()
    if status not in {"", "ok"}:
        return 0.0
    answer = str(answer).strip()
    if not answer:
        return 0.0

    lower = answer.lower()
    patterns = bad_patterns or [
        "sorry",
        "cannot",
        "can't",
        "unable to",
        "as an ai",
        "no image",
        "not enough information",
    ]
    if any(p and str(p).lower() in lower for p in patterns):
        return 0.0

    words = _WORD_RE.findall(answer)
    n_words = len(words)
    if n_words <= 0:
        return 0.0

    score = 1.0
    if min_words > 0 and n_words < min_words:
        score *= max(0.0, n_words / float(min_words))
    if max_words > 0 and n_words > max_words:
        score *= max(0.2, max_words / float(n_words))
    return float(np.clip(score, 0.0, 1.0))


def _numeric_sort_key(path: Path):
    try:
        return (path.parent.as_posix(), int(path.stem))
    except Exception:
        return (path.parent.as_posix(), path.stem)


def _load_mat_vector(path: Path, key: str | None = None) -> np.ndarray:
    data = loadmat(str(path))
    if key and key in data:
        arr = data[key]
    else:
        candidates = [(k, v) for k, v in data.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
        if not candidates:
            raise RuntimeError(f"No numeric vector found in {path}")
        arr = candidates[0][1]
    return np.asarray(arr).reshape(-1).astype(np.float32)


def _load_prompts_from_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path).fillna("")
    else:
        df = _read_csv_robust(path)
    cols = list(df.columns)
    preferred = [c for c in cols if "prompt" in str(c).lower() or "text" in str(c).lower() or "提示" in str(c)]
    use_col = preferred[0] if preferred else None
    if use_col is None:
        for c in cols:
            if df[c].dtype == object:
                use_col = c
                break
    if use_col is None:
        return [""] * len(df)
    return [str(x).strip() for x in df[use_col].tolist()]


def _cycle_prompt_idx(idx: int, num_prompts: int, images_per_prompt: int = 4, cycle_by_round: bool = True) -> int:
    if num_prompts <= 0:
        return 0
    if not cycle_by_round:
        return min(idx // max(1, images_per_prompt), num_prompts - 1)
    images_per_round = max(1, num_prompts * max(1, images_per_prompt))
    return min((idx % images_per_round) // max(1, images_per_prompt), num_prompts - 1)


class AGIQA3KPretrainDataset(Dataset):
    """AGIQA-3K: image-image + image-answer + image-prompt + prompt-answer."""

    def __init__(self, cfg: Dict, train: bool = True):
        self.root = Path(cfg["root_dir"])
        self.image_root = self.root / cfg.get("image_dir", "image")
        self.csv_path = self.root / cfg.get("csv_file", "data.csv")
        self.mirror_cfg, mirror_flip_prob = _pretrain_mirror_setup(cfg, train)
        self.transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, horizontal_flip_prob=mirror_flip_prob
        )
        self.clean_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=False, horizontal_flip_prob=mirror_flip_prob
        )
        self.quality_safe_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, quality_safe=True,
            horizontal_flip_prob=mirror_flip_prob
        )
        self.supervision_view = str(cfg.get("supervision_view", "augmented")).strip().lower()
        self.quality_col = cfg.get("quality_score_column", "mos_quality")
        self.mos_max = float(cfg.get("mos_max", 5.0))
        self.use_full_prompt = bool(cfg.get("use_full_prompt", True))
        df = _read_csv_robust(self.csv_path)
        required = {"name", "prompt", "answer", "mos_align", self.quality_col}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"AGIQA-3K missing columns: {missing}")
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _path(self, name: str) -> Path:
        candidates = [self.image_root / name, self.root / name, self.root / "image" / name, self.root / "images" / name]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found: {name}")

    def _make_prompt(self, row) -> str:
        base = str(row["prompt"]).strip()
        if not self.use_full_prompt:
            return base
        extra = []
        for key in ["adj1", "adj2", "style"]:
            if key in row:
                val = str(row[key]).strip()
                if val and val.lower() != "nan" and val not in base:
                    extra.append(val)
        return ", ".join([base] + extra) if extra else base

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pil = _read_pil(self._path(str(row["name"])))
        prompt = self._make_prompt(row)
        answer = str(row["answer"]).strip()
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        quality = _norm_score(float(row[self.quality_col]), self.mos_max)
        align = _norm_score(float(row["mos_align"]), self.mos_max)
        image1, image2, image_clean = _build_image_views(
            pil,
            self.transform,
            self.clean_transform,
            self.quality_safe_transform,
            self.supervision_view,
        )
        sample = {
            "image1": image1,
            "image2": image2,
            "prompt": prompt,
            "answer": answer,
            "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32),
            "image_weight": torch.tensor(1.0, dtype=torch.float32),
            "align_score": torch.tensor(align, dtype=torch.float32),
            "align_weight": torch.tensor(align, dtype=torch.float32),
            "answer_weight": torch.tensor(1.0, dtype=torch.float32),
            "prompt_answer_weight": torch.tensor(align, dtype=torch.float32),
            "mos": torch.tensor(quality, dtype=torch.float32),
            "quality_score": torch.tensor(quality, dtype=torch.float32),
            "quality_weight": torch.tensor(1.0, dtype=torch.float32),
            "artifact_score": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_score": torch.tensor(0.0, dtype=torch.float32),
            "artifact_weight": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_weight": torch.tensor(0.0, dtype=torch.float32),
            "dataset_id": "agiqa3k",
        }
        if image_clean is not None:
            sample["image_clean"] = image_clean
        return sample


class AGIQA1KPretrainDataset(Dataset):
    """AGIQA-1K: image-image + image-answer only."""

    def __init__(self, cfg: Dict, train: bool = True):
        self.root = Path(cfg["root_dir"])
        self.image_root = self.root / cfg.get("image_dir", "file")
        self.csv_path = self.root / cfg.get("csv_file", "AIGC-1K_answer.csv")
        self.mirror_cfg, mirror_flip_prob = _pretrain_mirror_setup(cfg, train)
        self.transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, horizontal_flip_prob=mirror_flip_prob
        )
        self.clean_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=False, horizontal_flip_prob=mirror_flip_prob
        )
        self.quality_safe_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, quality_safe=True,
            horizontal_flip_prob=mirror_flip_prob
        )
        self.supervision_view = str(cfg.get("supervision_view", "augmented")).strip().lower()
        self.mos_col = cfg.get("mos_column", "MOS")
        self.mos_max = float(cfg.get("mos_max", 5.0))
        self.mos_min = float(cfg.get("mos_min", 0.0))
        df = _read_csv_robust(self.csv_path)
        required = {"name", "prompt", "answer", self.mos_col}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"AGIQA-1K missing columns: {missing}")
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _path(self, name: str) -> Path:
        candidates = [self.image_root / name, self.root / name, self.root / "file" / name, self.root / "image" / name, self.root / "images" / name]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found: {name}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pil = _read_pil(self._path(str(row["name"])))
        prompt = str(row["prompt"]).strip()
        answer = str(row["answer"]).strip()
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        quality = _norm_score(
            float(row[self.mos_col]), self.mos_max, self.mos_min
        )
        image1, image2, image_clean = _build_image_views(
            pil,
            self.transform,
            self.clean_transform,
            self.quality_safe_transform,
            self.supervision_view,
        )
        sample = {
            "image1": image1,
            "image2": image2,
            "prompt": prompt,
            "answer": answer,
            "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32),
            "image_weight": torch.tensor(1.0, dtype=torch.float32),
            "align_score": torch.tensor(0.0, dtype=torch.float32),
            "align_weight": torch.tensor(0.0, dtype=torch.float32),
            "answer_weight": torch.tensor(1.0, dtype=torch.float32),
            "prompt_answer_weight": torch.tensor(0.0, dtype=torch.float32),
            "mos": torch.tensor(quality, dtype=torch.float32),
            "quality_score": torch.tensor(quality, dtype=torch.float32),
            "quality_weight": torch.tensor(1.0, dtype=torch.float32),
            "artifact_score": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_score": torch.tensor(0.0, dtype=torch.float32),
            "artifact_weight": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_weight": torch.tensor(0.0, dtype=torch.float32),
            "dataset_id": "agiqa1k",
        }
        if image_clean is not None:
            sample["image_clean"] = image_clean
        return sample


class AIGCIQA2023PretrainDataset(Dataset):
    """AIGCIQA2023: image-image + image-prompt.

    mosz1 is used as image-image weight.
    mosz3 is used as image-prompt weight.
    There is no answer, so image-answer and prompt-answer are disabled.
    """

    def __init__(self, cfg: Dict, train: bool = True):
        self.root = Path(cfg["root_dir"])
        self.image_root = self.root / cfg.get("image_dir", "Image/allimg")
        self.mirror_cfg, mirror_flip_prob = _pretrain_mirror_setup(cfg, train)
        self.transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, horizontal_flip_prob=mirror_flip_prob
        )
        self.clean_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=False, horizontal_flip_prob=mirror_flip_prob
        )
        self.quality_safe_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, quality_safe=True,
            horizontal_flip_prob=mirror_flip_prob
        )
        self.supervision_view = str(cfg.get("supervision_view", "augmented")).strip().lower()

        mosz1_file = self.root / cfg.get("mosz1_file", "mosz1.mat")
        mosz3_file = self.root / cfg.get("mosz3_file", "mosz3.mat")
        key1 = cfg.get("mosz1_key", "MOSz")
        key3 = cfg.get("mosz3_key", "MOSz")
        self.quality = _load_mat_vector(mosz1_file, key1 if key1 else None)
        self.align = _load_mat_vector(mosz3_file, key3 if key3 else None)
        self.mos_max = float(cfg.get("mos_max", 100.0))

        exts = tuple(cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]))
        self.images = sorted([p for p in self.image_root.rglob("*") if p.is_file() and p.suffix.lower() in exts], key=_numeric_sort_key)
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_root}")
        n = min(len(self.images), len(self.quality), len(self.align))
        self.images = self.images[:n]
        self.quality = self.quality[:n]
        self.align = self.align[:n]

        prompt_file = cfg.get("prompt_file", "")
        self.prompts = _load_prompts_from_file(self.root / prompt_file) if prompt_file else []
        self.images_per_prompt = int(cfg.get("images_per_prompt", 4))
        self.num_prompts = int(cfg.get("num_prompts", len(self.prompts) if self.prompts else 100))
        self.prompt_cycle_by_round = bool(cfg.get("prompt_cycle_by_round", True))
        self.debug_prompt_mapping = bool(cfg.get("debug_prompt_mapping", False))

        if self.debug_prompt_mapping:
            print("[AIGCIQA2023Pretrain] prompt mapping preview:")
            for i in [0, 1, 2, 3, 4, 400, 401, 402, 403, 800, 801, 802, 803]:
                if i < len(self.images):
                    print(f"idx={i}, img={self.images[i].name}, prompt_idx={self._prompt_index(i)}, prompt={self._prompt_for_idx(i)[:80]}")

    def __len__(self):
        return len(self.images)

    def _prompt_index(self, idx: int) -> int:
        if not self.prompts:
            return -1
        if len(self.prompts) == len(self.images):
            return idx
        return _cycle_prompt_idx(idx, len(self.prompts), self.images_per_prompt, self.prompt_cycle_by_round)

    def _prompt_for_idx(self, idx: int) -> str:
        if not self.prompts:
            return ""
        return self.prompts[self._prompt_index(idx)]

    def __getitem__(self, idx):
        pil = _read_pil(self.images[idx])
        prompt = self._prompt_for_idx(idx)
        answer = prompt
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        q = _norm_score(float(self.quality[idx]), self.mos_max)
        a = _norm_score(float(self.align[idx]), self.mos_max)
        image1, image2, image_clean = _build_image_views(
            pil,
            self.transform,
            self.clean_transform,
            self.quality_safe_transform,
            self.supervision_view,
        )
        sample = {
            "image1": image1,
            "image2": image2,
            "prompt": prompt,
            "answer": answer,
            "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32),
            "image_weight": torch.tensor(q, dtype=torch.float32),
            "align_score": torch.tensor(a, dtype=torch.float32),
            "align_weight": torch.tensor(a, dtype=torch.float32),
            "answer_weight": torch.tensor(0.0, dtype=torch.float32),
            "prompt_answer_weight": torch.tensor(0.0, dtype=torch.float32),
            "mos": torch.tensor(q, dtype=torch.float32),
            "quality_score": torch.tensor(q, dtype=torch.float32),
            "quality_weight": torch.tensor(1.0, dtype=torch.float32),
            "artifact_score": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_score": torch.tensor(0.0, dtype=torch.float32),
            "artifact_weight": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_weight": torch.tensor(0.0, dtype=torch.float32),
            "dataset_id": "aigciqa2023",
        }
        if image_clean is not None:
            sample["image_clean"] = image_clean
        return sample


class AIGIQA20KPretrainDataset(Dataset):
    """AIGIQA-20K: image-image + image-answer + image-prompt + prompt-answer.

    Required CSV columns:
      - name
      - prompt
      - answer
      - mos

    The original comprehensive mos is used as a soft consistency weight for
    image-prompt and prompt-answer losses, as requested.
    """

    def __init__(self, cfg: Dict, train: bool = True):
        self.root = Path(cfg["root_dir"])
        self.image_root = self.root / cfg.get("image_dir", ".")
        csv_file = cfg.get("csv_file", "info_merge_with_answer_local.csv")
        self.csv_path = self.root / csv_file
        self.mirror_cfg, mirror_flip_prob = _pretrain_mirror_setup(cfg, train)
        self.transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, horizontal_flip_prob=mirror_flip_prob
        )
        self.clean_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=False, horizontal_flip_prob=mirror_flip_prob
        )
        self.quality_safe_transform = PretrainImageTransform(
            cfg.get("input_size", 224), train=train, quality_safe=True,
            horizontal_flip_prob=mirror_flip_prob
        )
        self.supervision_view = str(cfg.get("supervision_view", "augmented")).strip().lower()
        self.name_col = cfg.get("image_name_column", "name")
        self.prompt_col = cfg.get("prompt_column", "prompt")
        self.answer_col = cfg.get("answer_column", "answer")
        self.mos_col = cfg.get("mos_column", "mos")
        self.status_col = cfg.get("status_column", "answer_status")
        self.only_ok_answer = bool(cfg.get("only_ok_answer", True))
        self.skip_missing_images = bool(cfg.get("skip_missing_images", True))
        self.search_subdirs = [str(x) for x in cfg.get("search_subdirs", ["train", "val", "test"])]
        self.mos_max = float(cfg.get("mos_max", 5.0))
        self.mos_min = float(cfg.get("mos_min", 0.0))

        # Optional pseudo image-text alignment score, e.g. CLIPScore normalized to [0, 1].
        # This should NOT replace MOS blindly. Recommended modes:
        #   mos              : align_weight = mos_norm  (stable baseline)
        #   pseudo           : align_weight = pseudo_align_score  (not recommended; often noisy)
        #   mos_pseudo_mix   : align_weight = mos_norm * (1-alpha + alpha*pseudo)
        #   mos_pseudo_filter: align_weight = mos_norm if pseudo >= threshold else 0
        self.pseudo_align_col = cfg.get("pseudo_align_column", "pseudo_align_score")
        self.pseudo_align_status_col = cfg.get("pseudo_align_status_column", "pseudo_align_status")
        self.align_weight_mode = str(cfg.get("align_weight_mode", "mos")).lower()
        self.pseudo_alpha = float(cfg.get("pseudo_alpha", 0.3))
        self.pseudo_filter_threshold = float(cfg.get("pseudo_filter_threshold", 0.25))

        self.image_answer_score_col = cfg.get("image_answer_score_column", "image_answer_score")
        self.image_answer_status_col = cfg.get("image_answer_status_column", "image_answer_status")
        self.semantic_conf_col = cfg.get("semantic_confidence_column", "semantic_confidence_score")
        self.semantic_conf_status_col = cfg.get("semantic_confidence_status_column", "semantic_confidence_status")

        self.answer_conf_col = cfg.get("answer_confidence_column", "answer_confidence")
        self.answer_conf_source = str(cfg.get("answer_confidence_source", "column_or_heuristic")).lower()
        self.answer_conf_default = float(cfg.get("answer_confidence_default", 1.0))
        self.answer_conf_min = float(cfg.get("answer_confidence_min", 0.0))
        self.answer_min_words = int(cfg.get("answer_min_words", 8))
        self.answer_max_words = int(cfg.get("answer_max_words", 120))
        self.answer_bad_patterns = [str(x).lower() for x in cfg.get("answer_bad_patterns", [])]
        self.filter_low_conf_answer = bool(cfg.get("filter_low_confidence_answer", False))
        self.answer_weight_mode = str(cfg.get("answer_weight_mode", "constant")).lower()

        self.prompt_answer_score_col = cfg.get("prompt_answer_score_column", "prompt_answer_consistency_score")
        self.prompt_answer_status_col = cfg.get("prompt_answer_status_column", "prompt_answer_consistency_status")
        self.prompt_answer_score_source = str(cfg.get("prompt_answer_score_source", "pseudo_align")).lower()
        self.prompt_answer_weight_mode = str(cfg.get("prompt_answer_weight_mode", "align_weight")).lower()
        self.pa_positive_threshold = float(cfg.get("pa_positive_threshold", 0.70))
        self.pa_negative_threshold = float(cfg.get("pa_negative_threshold", 0.30))

        self.split_group_source = str(cfg.get("split_group_source", "prompt")).lower()
        self.split_group_col = cfg.get("split_group_column", "split_group")

        df = _read_csv_robust(self.csv_path)
        required = {self.name_col, self.prompt_col, self.answer_col, self.mos_col}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"AIGIQA-20K CSV missing columns: {missing}. CSV={self.csv_path}")
        if self.align_weight_mode in {"pseudo", "mos_pseudo_mix", "mos_pseudo_filter"} and self.pseudo_align_col not in df.columns:
            raise RuntimeError(
                f"align_weight_mode={self.align_weight_mode} requires column "
                f"'{self.pseudo_align_col}' in CSV. CSV={self.csv_path}"
            )
        if self.align_weight_mode == "semantic_confidence" and self.semantic_conf_col not in df.columns:
            raise RuntimeError(
                f"align_weight_mode=semantic_confidence requires column "
                f"'{self.semantic_conf_col}' in CSV. CSV={self.csv_path}"
            )
        if self.only_ok_answer and self.status_col in df.columns:
            df = df[df[self.status_col].astype(str).str.lower().eq("ok")].copy()
        df[self.answer_col] = df[self.answer_col].astype(str).str.strip()
        df = df[df[self.answer_col].str.len() > 0].copy()
        if self.filter_low_conf_answer:
            before = len(df)
            conf = df.apply(
                lambda row: self._get_answer_confidence(row, str(row[self.answer_col]).strip()),
                axis=1,
            )
            df = df[conf >= self.answer_conf_min].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"[AIGIQA20KPretrain] skipped {removed} low-confidence answer rows.")
        if self.skip_missing_images:
            before = len(df)
            df = df[df[self.name_col].astype(str).map(lambda x: self._find_path(x.strip()) is not None)].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"[AIGIQA20KPretrain] skipped {removed} rows with missing images.")
        self.df = df.reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"AIGIQA-20K has no valid answer rows with existing images: {self.csv_path}")
        print(f"[AIGIQA20KPretrain] rows={len(self.df)}, csv={self.csv_path}")

    def get_split_groups(self) -> List[str]:
        groups = []
        for _, row in self.df.iterrows():
            if self.split_group_source == "column" and self.split_group_col in row.index:
                groups.append(_normalize_prompt_group(row[self.split_group_col]))
            elif self.split_group_source == "name":
                groups.append(_normalize_prompt_group(row[self.name_col]))
            else:
                groups.append(_normalize_prompt_group(row[self.prompt_col]))
        return groups

    def __len__(self):
        return len(self.df)

    def _candidate_paths(self, name: str) -> List[Path]:
        candidates = [self.image_root / name, self.root / name]
        for sub in self.search_subdirs:
            candidates.append(self.image_root / sub / name)
            candidates.append(self.root / sub / name)
        # Remove duplicates while preserving order.
        out, seen = [], set()
        for p in candidates:
            sp = str(p)
            if sp not in seen:
                out.append(p)
                seen.add(sp)
        return out

    def _find_path(self, name: str) -> Path | None:
        for p in self._candidate_paths(name):
            if p.exists():
                return p
        return None

    def _path(self, name: str) -> Path:
        p = self._find_path(name)
        if p is not None:
            return p
        raise FileNotFoundError(f"AIGIQA-20K image not found: {name}. Tried: {[str(p) for p in self._candidate_paths(name)[:8]]} ...")

    def _get_pseudo_align(self, row, default_value: float) -> float:
        """Read a pseudo image-prompt alignment score from the CSV.

        The returned value is always in [0, 1]. If the column is absent, invalid,
        or has a non-ok status, the method falls back to default_value, usually
        normalized MOS. This makes the code safe for old CSV files.
        """
        if self.pseudo_align_col not in row.index:
            return float(default_value)

        if self.pseudo_align_status_col in row.index:
            status = str(row.get(self.pseudo_align_status_col, "")).strip().lower()
            # Empty status is allowed for old files; non-ok values are treated as unreliable.
            if status not in {"", "ok"}:
                return float(default_value)

        try:
            v = float(row[self.pseudo_align_col])
        except Exception:
            return float(default_value)
        if not np.isfinite(v):
            return float(default_value)
        return float(np.clip(v, 0.0, 1.0))

    def _get_score_column(self, row, score_col: str, status_col: str, default_value: float) -> float:
        if score_col not in row.index:
            return float(default_value)
        if status_col in row.index:
            status = str(row.get(status_col, "")).strip().lower()
            if status not in {"", "ok"}:
                return float(default_value)
        return _safe_float01(row[score_col], default_value=default_value)

    def _get_answer_confidence(self, row, answer: str) -> float:
        if self.answer_conf_source in {"column", "column_or_heuristic"} and self.answer_conf_col in row.index:
            v = _safe_float01(row[self.answer_conf_col], default_value=np.nan)
            if np.isfinite(v):
                return v
            if self.answer_conf_source == "column":
                return float(np.clip(self.answer_conf_default, 0.0, 1.0))

        status = str(row.get(self.status_col, "")).strip().lower() if self.status_col in row.index else ""
        heuristic = _answer_confidence_heuristic(
            answer,
            status=status,
            min_words=self.answer_min_words,
            max_words=self.answer_max_words,
            bad_patterns=self.answer_bad_patterns,
        )
        if self.answer_conf_source == "constant":
            return float(np.clip(self.answer_conf_default, 0.0, 1.0))
        return heuristic

    def _build_align_weight(self, mos: float, pseudo_align: float, semantic_confidence: float = 1.0) -> float:
        """Build the image-prompt SimSiam weight for 20K pretraining.

        All inputs must be normalized into [0, 1].
        """
        mos = float(np.clip(mos, 0.0, 1.0))
        pseudo_align = float(np.clip(pseudo_align, 0.0, 1.0))
        semantic_confidence = float(np.clip(semantic_confidence, 0.0, 1.0))
        mode = self.align_weight_mode

        if mode == "mos":
            return mos
        if mode == "pseudo":
            return pseudo_align
        if mode == "semantic_confidence":
            return semantic_confidence
        if mode == "mos_pseudo_mix":
            alpha = float(np.clip(self.pseudo_alpha, 0.0, 1.0))
            return mos * ((1.0 - alpha) + alpha * pseudo_align)
        if mode == "mos_pseudo_filter":
            return mos if pseudo_align >= self.pseudo_filter_threshold else 0.0

        raise ValueError(
            f"Unknown align_weight_mode={self.align_weight_mode}. "
            "Choose from: mos, pseudo, semantic_confidence, mos_pseudo_mix, mos_pseudo_filter."
        )

    def _get_prompt_answer_score(self, row, pseudo_align: float, answer_conf: float, align_weight: float) -> float:
        source = self.prompt_answer_score_source
        if source == "column":
            if self.prompt_answer_score_col not in row.index:
                return float(np.clip(min(pseudo_align, answer_conf), 0.0, 1.0))
            if self.prompt_answer_status_col in row.index:
                status = str(row.get(self.prompt_answer_status_col, "")).strip().lower()
                if status not in {"", "ok"}:
                    return float(np.clip(min(pseudo_align, answer_conf), 0.0, 1.0))
            return _safe_float01(row[self.prompt_answer_score_col], default_value=min(pseudo_align, answer_conf))
        if source == "answer_confidence":
            return float(np.clip(answer_conf, 0.0, 1.0))
        if source == "align_weight":
            return float(np.clip(align_weight, 0.0, 1.0))
        if source == "min_pseudo_answer_conf":
            return float(np.clip(min(pseudo_align, answer_conf), 0.0, 1.0))
        return float(np.clip(pseudo_align, 0.0, 1.0))

    def _build_prompt_answer_weights(self, pa_score: float, answer_conf: float, align_weight: float, semantic_confidence: float = 1.0) -> Tuple[float, float]:
        pa_score = float(np.clip(pa_score, 0.0, 1.0))
        answer_conf = float(np.clip(answer_conf, 0.0, 1.0))
        align_weight = float(np.clip(align_weight, 0.0, 1.0))
        semantic_confidence = float(np.clip(semantic_confidence, 0.0, 1.0))
        mode = self.prompt_answer_weight_mode

        if mode == "zero":
            return 0.0, 0.0
        if mode == "align_weight":
            return align_weight, 0.0
        if mode == "answer_confidence":
            return answer_conf, 0.0
        if mode in {"pa_score", "prompt_answer_score"}:
            return pa_score * answer_conf, 0.0
        if mode in {"high_conf_pa", "threshold"}:
            pos_den = max(1e-6, 1.0 - self.pa_positive_threshold)
            neg_den = max(1e-6, self.pa_negative_threshold)
            pos_w = np.clip((pa_score - self.pa_positive_threshold) / pos_den, 0.0, 1.0) * answer_conf
            gap_w = np.clip((self.pa_negative_threshold - pa_score) / neg_den, 0.0, 1.0) * answer_conf
            return float(pos_w), float(gap_w)
        if mode in {"high_conf_semantic", "semantic_confidence_high", "semantic_threshold"}:
            pos_den = max(1e-6, 1.0 - self.pa_positive_threshold)
            neg_den = max(1e-6, self.pa_negative_threshold)
            pos_w = np.clip((semantic_confidence - self.pa_positive_threshold) / pos_den, 0.0, 1.0) * semantic_confidence
            gap_w = np.clip((self.pa_negative_threshold - semantic_confidence) / neg_den, 0.0, 1.0) * answer_conf
            return float(pos_w), float(gap_w)

        raise ValueError(
            f"Unknown prompt_answer_weight_mode={self.prompt_answer_weight_mode}. "
            "Choose from: zero, align_weight, answer_confidence, pa_score, high_conf_pa, high_conf_semantic."
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row[self.name_col]).strip()
        pil = _read_pil(self._path(name))
        prompt = str(row[self.prompt_col]).strip()
        answer = str(row[self.answer_col]).strip()
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        mos = _norm_score(float(row[self.mos_col]), self.mos_max, self.mos_min)
        pseudo_align = self._get_pseudo_align(row, default_value=mos)
        answer_conf = self._get_answer_confidence(row, answer)
        image_answer_score = self._get_score_column(row, self.image_answer_score_col, self.image_answer_status_col, default_value=answer_conf)
        pa_score = self._get_prompt_answer_score(row, pseudo_align, answer_conf, pseudo_align)
        semantic_confidence = self._get_score_column(
            row,
            self.semantic_conf_col,
            self.semantic_conf_status_col,
            default_value=min(pseudo_align, image_answer_score, pa_score, answer_conf),
        )
        align_weight = self._build_align_weight(mos=mos, pseudo_align=pseudo_align, semantic_confidence=semantic_confidence)
        prompt_answer_weight, prompt_answer_gap_weight = self._build_prompt_answer_weights(pa_score, answer_conf, align_weight, semantic_confidence)

        if self.answer_weight_mode in {"answer_confidence", "confidence"}:
            answer_weight = answer_conf
        elif self.answer_weight_mode in {"image_answer_score", "image_answer"}:
            answer_weight = image_answer_score * answer_conf
        elif self.answer_weight_mode in {"image_answer_prompt_answer", "ia_pa"}:
            answer_weight = min(image_answer_score, pa_score) * answer_conf
        elif self.answer_weight_mode in {"semantic_confidence", "semantic"}:
            answer_weight = semantic_confidence
        else:
            answer_weight = 1.0
        image1, image2, image_clean = _build_image_views(
            pil,
            self.transform,
            self.clean_transform,
            self.quality_safe_transform,
            self.supervision_view,
        )
        sample = {
            "image1": image1,
            "image2": image2,
            "prompt": prompt,
            "answer": answer,
            "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32),
            "image_weight": torch.tensor(1.0, dtype=torch.float32),
            # align_score is kept for logging/debugging. The actual image-prompt
            # loss uses align_weight in train_cm_pretrain.py.
            "align_score": torch.tensor(pseudo_align, dtype=torch.float32),
            "align_weight": torch.tensor(align_weight, dtype=torch.float32),
            "answer_weight": torch.tensor(answer_weight, dtype=torch.float32),
            "answer_confidence": torch.tensor(answer_conf, dtype=torch.float32),
            "image_answer_score": torch.tensor(image_answer_score, dtype=torch.float32),
            "semantic_confidence": torch.tensor(semantic_confidence, dtype=torch.float32),
            "prompt_answer_score": torch.tensor(pa_score, dtype=torch.float32),
            "prompt_answer_weight": torch.tensor(prompt_answer_weight, dtype=torch.float32),
            "prompt_answer_gap_weight": torch.tensor(prompt_answer_gap_weight, dtype=torch.float32),
            # MOS is used by the generic persistent quality head.
            # It is already normalized by _norm_score() into [0, 1].
            "mos": torch.tensor(mos, dtype=torch.float32),
            "quality_score": torch.tensor(mos, dtype=torch.float32),
            "quality_weight": torch.tensor(1.0, dtype=torch.float32),
            "artifact_score": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_score": torch.tensor(0.0, dtype=torch.float32),
            "artifact_weight": torch.tensor(0.0, dtype=torch.float32),
            "aesthetics_weight": torch.tensor(0.0, dtype=torch.float32),
            "pseudo_align": torch.tensor(pseudo_align, dtype=torch.float32),
            "split_group": _normalize_prompt_group(prompt),
            "dataset_id": "aigiqa20k",
        }
        if image_clean is not None:
            sample["image_clean"] = image_clean
        return sample


class RichHF18KPretrainDataset(Dataset):
    """RichHF-18K: human alignment/quality labels + generated answer confidence.

    RichHF scalar scores are already in [0, 1] and higher-is-better. The
    dataset uses:
      - image-prompt weight: human text-image alignment score
      - image-answer weight: generated answer confidence
      - prompt-answer weight: alignment * answer confidence
      - prompt-answer gap weight: (1 - alignment) * answer confidence
      - mos/rank target: configurable, default overall_score
    """

    def __init__(self, cfg: Dict, train: bool = True):
        self.train = bool(train)
        self.root = Path(cfg["root_dir"])
        split = str(cfg.get("split", "train")).lower()
        if split in {"validation", "dev"}:
            split = "val"
        self.split = split

        csv_pattern = str(cfg.get("csv_file", "{split}_with_answer_confidence.csv"))
        if split in {"all", "merged", "combined"}:
            source_splits = cfg.get("source_splits", ["train", "val", "test"])
            normalized_splits = []
            for source_split in source_splits:
                source_split = str(source_split).lower()
                normalized_splits.append("val" if source_split in {"validation", "dev"} else source_split)
            self.csv_paths = [
                self.root / csv_pattern.format(split=source_split)
                for source_split in normalized_splits
            ]
        else:
            self.csv_paths = [self.root / csv_pattern.format(split=split)]
        self.csv_path = self.csv_paths[0]
        input_size = cfg.get("input_size", 224)
        self.mirror_cfg, mirror_flip_prob = _pretrain_mirror_setup(cfg, train)
        self.transform = PretrainImageTransform(
            input_size, train=train, horizontal_flip_prob=mirror_flip_prob
        )
        self.clean_transform = PretrainImageTransform(
            input_size, train=False, horizontal_flip_prob=mirror_flip_prob
        )
        self.quality_safe_transform = PretrainImageTransform(
            input_size, train=train, quality_safe=True,
            horizontal_flip_prob=mirror_flip_prob
        )
        self.supervision_view = str(cfg.get("supervision_view", "augmented")).strip().lower()
        if self.supervision_view not in {
            "augmented", "clean", "quality_safe", "three_view", "separate_clean"
        }:
            raise ValueError(
                f"Unknown RichHF supervision_view={self.supervision_view!r}. "
                "Choose from: augmented, clean, quality_safe, three_view."
            )

        self.name_col = cfg.get("image_name_column", "name")
        self.image_path_col = cfg.get("image_path_column", "image_path")
        self.prompt_col = cfg.get("prompt_column", "prompt")
        self.answer_col = cfg.get("answer_column", "answer")
        self.status_col = cfg.get("status_column", "answer_status")
        self.answer_conf_col = cfg.get("answer_confidence_column", "answer_confidence")
        self.quality_col = cfg.get("quality_score_column", "quality_score")
        self.quality_score_mode = str(cfg.get("quality_score_mode", "column")).lower()
        self.artifact_col = cfg.get("artifact_score_column", "artifact_score")
        self.aesthetics_col = cfg.get("aesthetics_score_column", "aesthetics_score")
        self.quality_artifact_weight = float(cfg.get("quality_artifact_weight", 0.5))
        self.quality_aesthetics_weight = float(cfg.get("quality_aesthetics_weight", 0.5))
        self.align_col = cfg.get("align_score_column", "align_score")
        self.rank_target_col = cfg.get("rank_target_column", "overall_score")
        self.use_local_feedback = bool(cfg.get("use_local_feedback", False))
        self.misalignment_map_path_col = cfg.get("misalignment_map_path_column", "misalignment_map_path")
        self.misalignment_map_dir = str(cfg.get("misalignment_map_dir", "{split}/misalignment_maps"))
        self.prompt_misalignment_col = cfg.get(
            "prompt_misalignment_label_column", "prompt_misalignment_label"
        )

        self.only_ok_answer = bool(cfg.get("only_ok_answer", True))
        self.skip_missing_images = bool(cfg.get("skip_missing_images", True))
        self.filter_low_conf_answer = bool(cfg.get("filter_low_confidence_answer", False))
        self.answer_conf_min = float(cfg.get("answer_confidence_min", 0.0))
        self.answer_conf_default = float(cfg.get("answer_confidence_default", 0.0))
        self.answer_weight_mode = str(cfg.get("answer_weight_mode", "answer_confidence")).lower()
        self.align_weight_mode = str(cfg.get("align_weight_mode", "human_align")).lower()
        self.prompt_answer_weight_mode = str(cfg.get("prompt_answer_weight_mode", "align_confidence")).lower()
        self.use_prompt_answer_gap = bool(cfg.get("use_prompt_answer_gap", True))

        frames = []
        for csv_path in self.csv_paths:
            if not csv_path.exists():
                raise FileNotFoundError(f"RichHF-18K CSV not found: {csv_path}")
            frames.append(_read_csv_robust(csv_path))
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        required = {
            self.name_col,
            self.prompt_col,
            self.answer_col,
            self.align_col,
            self.rank_target_col,
        }
        if self.quality_score_mode in {"weighted_components", "artifact_aesthetics", "components"}:
            required.update({self.artifact_col, self.aesthetics_col})
        elif self.quality_score_mode == "column":
            required.add(self.quality_col)
        else:
            raise ValueError(
                f"Unknown RichHF quality_score_mode={self.quality_score_mode}. "
                "Choose from: column, weighted_components."
            )
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"RichHF-18K CSV missing columns: {missing}. CSV={self.csv_path}")
        if self.use_local_feedback and self.prompt_misalignment_col not in df.columns:
            raise RuntimeError(
                f"RichHF-18K local feedback requires column "
                f"{self.prompt_misalignment_col!r}. CSV={self.csv_path}"
            )
        if self.image_path_col not in df.columns:
            print(f"[RichHF18KPretrain] image_path column missing; will resolve from split/images/name.")

        if self.only_ok_answer and self.status_col in df.columns:
            df = df[df[self.status_col].astype(str).str.lower().eq("ok")].copy()
        df[self.answer_col] = df[self.answer_col].astype(str).str.strip()
        df = df[df[self.answer_col].str.len() > 0].copy()

        if self.filter_low_conf_answer:
            before = len(df)
            conf = df.apply(lambda row: self._get_answer_confidence(row), axis=1)
            df = df[conf >= self.answer_conf_min].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"[RichHF18KPretrain] skipped {removed} low-confidence answer rows.")

        if self.skip_missing_images:
            before = len(df)
            df = df[df.apply(lambda row: self._find_path(row) is not None, axis=1)].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"[RichHF18KPretrain] skipped {removed} rows with missing images.")

        self.df = df.reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"RichHF-18K has no valid rows: {self.csv_path}")
        self.quality_loss_weights = None
        self.quality_component_loss_weights = None
        quality_loss_cfg = cfg.get("quality_loss_reweighting", {})
        if isinstance(quality_loss_cfg, dict) and bool(quality_loss_cfg.get("enabled", False)):
            quality_loss_cfg = dict(quality_loss_cfg)
            quality_loss_cfg["_purpose"] = "loss-reweighting"
            component_weights = self.get_quality_sampling_weights(
                quality_loss_cfg, return_components=True
            )
            self.quality_loss_weights = component_weights["combined"]
            self.quality_component_loss_weights = component_weights
        csv_summary = ",".join(str(path) for path in self.csv_paths)
        print(
            f"[RichHF18KPretrain] split={self.split}, rows={len(self.df)}, "
            f"local_feedback={self.use_local_feedback}, "
            f"supervision_view={self.supervision_view}, csv={self.csv_path}"
        )

    def __len__(self):
        return len(self.df)

    def get_split_groups(self) -> List[str]:
        return [_normalize_prompt_group(x) for x in self.df[self.prompt_col].tolist()]

    def get_quality_relation_metadata(self) -> List[Dict]:
        """Return label metadata used to construct same-prompt quality pairs."""
        return [
            {
                "prompt_group": _normalize_prompt_group(row[self.prompt_col]),
                "quality": self._get_quality_score(row),
                "align": _safe_float01(row[self.align_col], default_value=0.0),
                "dataset_id": "richhf18k",
            }
            for _, row in self.df.iterrows()
        ]

    def get_quality_sampling_weights(
        self, sampling_cfg: Dict, *, return_components: bool = False
    ) -> List[float] | Dict[str, List[float]]:
        """Return label-balanced replacement-sampling weights for RichHF rows.

        The sampler changes only how often a training row is seen. It does not
        modify images, labels, views, or any loss. Artifact and aesthetics are
        balanced independently and then averaged, so a rare value in either
        component receives more exposure without letting a small corner of the
        joint distribution dominate training.
        """
        if not isinstance(sampling_cfg, dict) or not bool(sampling_cfg.get("enabled", False)):
            return [1.0] * len(self.df)

        default_edges = [0.0, 0.5, 0.6, 0.7, 0.85, 1.000001]
        edges = np.asarray(sampling_cfg.get("bin_edges", default_edges), dtype=np.float64)
        if edges.ndim != 1 or edges.size < 3 or not np.all(np.diff(edges) > 0):
            raise ValueError(
                "quality_balanced_sampling.bin_edges must be strictly increasing "
                "with at least three values."
            )
        if edges[0] > 0.0 or edges[-1] < 1.0:
            raise ValueError(
                "quality_balanced_sampling.bin_edges must cover the normalized [0, 1] range."
            )

        power = float(sampling_cfg.get("weight_power", 0.5))
        min_weight = float(sampling_cfg.get("min_weight", 0.5))
        max_weight = float(sampling_cfg.get("max_weight", 3.0))
        if power < 0.0:
            raise ValueError("quality_balanced_sampling.weight_power must be non-negative.")
        if min_weight <= 0.0 or max_weight < min_weight:
            raise ValueError(
                "quality_balanced_sampling requires 0 < min_weight <= max_weight."
            )

        artifact_values = np.asarray(
            [_safe_float01(row.get(self.artifact_col, 0.0), default_value=0.0)
             for _, row in self.df.iterrows()],
            dtype=np.float64,
        )
        aesthetics_values = np.asarray(
            [_safe_float01(row.get(self.aesthetics_col, 0.0), default_value=0.0)
             for _, row in self.df.iterrows()],
            dtype=np.float64,
        )

        def component_weights(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Interior edges define bins; the final edge is slightly above one
            # so that an exact score of 1.0 is included in the last bin.
            bin_ids = np.digitize(values, edges[1:-1], right=False)
            counts = np.bincount(bin_ids, minlength=edges.size - 1).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            raw = (float(len(values)) / counts[bin_ids]) ** power
            raw = raw / max(float(raw.mean()), 1e-12)
            raw = np.clip(raw, min_weight, max_weight)
            raw = raw / max(float(raw.mean()), 1e-12)
            return raw, counts

        artifact_balanced, artifact_counts = component_weights(artifact_values)
        aesthetics_balanced, aesthetics_counts = component_weights(aesthetics_values)

        # Mix each component with the uniform distribution independently.
        # This is important for loss reweighting: Artifact rarity must not
        # accidentally amplify the Aesthetics loss, and vice versa.
        balance_strength = float(sampling_cfg.get("balance_strength", 1.0))
        if not 0.0 <= balance_strength <= 1.0:
            raise ValueError(
                "quality_balanced_sampling.balance_strength must be in [0, 1]."
            )

        def mix_component(values: np.ndarray) -> np.ndarray:
            mixed = (1.0 - balance_strength) + balance_strength * values
            return mixed / max(float(mixed.mean()), 1e-12)

        artifact_component_weights = mix_component(artifact_balanced)
        aesthetics_component_weights = mix_component(aesthetics_balanced)
        balanced_weights = (
            0.5 * artifact_component_weights + 0.5 * aesthetics_component_weights
        )
        balanced_weights = balanced_weights / max(float(balanced_weights.mean()), 1e-12)

        purpose = str(sampling_cfg.get("_purpose", "sampling"))
        print(
            f"[RichHF18KPretrain] quality-balanced {purpose} enabled: "
            f"artifact_counts={artifact_counts.astype(int).tolist()}, "
            f"aesthetics_counts={aesthetics_counts.astype(int).tolist()}, "
            f"combined_range=({balanced_weights.min():.3f}, {balanced_weights.max():.3f}), "
            f"artifact_range=({artifact_component_weights.min():.3f}, {artifact_component_weights.max():.3f}), "
            f"aesthetics_range=({aesthetics_component_weights.min():.3f}, {aesthetics_component_weights.max():.3f}), "
            f"power={power:.3f}, balance_strength={balance_strength:.3f}"
        )
        if return_components:
            return {
                "combined": balanced_weights.astype(np.float64).tolist(),
                "artifact": artifact_component_weights.astype(np.float64).tolist(),
                "aesthetics": aesthetics_component_weights.astype(np.float64).tolist(),
            }
        return balanced_weights.astype(np.float64).tolist()


    def _candidate_paths(self, row) -> List[Path]:
        name = str(row[self.name_col]).strip()
        candidates: List[Path] = []
        if self.image_path_col in row.index:
            image_path = str(row.get(self.image_path_col, "")).strip()
            if image_path:
                candidates.append(self.root / image_path)
                candidates.append(Path(image_path))
        candidates.extend([
            self.root / self.split / "images" / name,
            self.root / "train" / "images" / name,
            self.root / "val" / "images" / name,
            self.root / "test" / "images" / name,
            self.root / name,
        ])
        out, seen = [], set()
        for p in candidates:
            sp = str(p)
            if sp not in seen:
                out.append(p)
                seen.add(sp)
        return out

    def _find_path(self, row) -> Path | None:
        for p in self._candidate_paths(row):
            if p.exists():
                return p
        return None

    def _path(self, row) -> Path:
        p = self._find_path(row)
        if p is not None:
            return p
        name = str(row[self.name_col]).strip()
        tried = [str(p) for p in self._candidate_paths(row)[:8]]
        raise FileNotFoundError(f"RichHF-18K image not found: {name}. Tried: {tried}")

    def _misalignment_map_path(self, row) -> Path | None:
        candidates: List[Path] = []
        if self.misalignment_map_path_col in row.index:
            value = str(row.get(self.misalignment_map_path_col, "")).strip()
            if value:
                candidates.extend([self.root / value, Path(value)])
        name = str(row[self.name_col]).strip()
        map_dir = self.misalignment_map_dir.format(split=self.split)
        candidates.extend([
            self.root / map_dir / name,
            self.root / "train" / "misalignment_maps" / name,
            self.root / "val" / "misalignment_maps" / name,
            self.root / "test" / "misalignment_maps" / name,
        ])
        for path in candidates:
            if path.exists():
                return path
        return None

    def _get_answer_confidence(self, row) -> float:
        if self.answer_conf_col in row.index:
            return _safe_float01(row[self.answer_conf_col], default_value=self.answer_conf_default)
        return float(np.clip(self.answer_conf_default, 0.0, 1.0))

    def _get_quality_score(self, row) -> float:
        if self.quality_score_mode == "column":
            return _safe_float01(row[self.quality_col], default_value=0.0)

        artifact = _safe_float01(row[self.artifact_col], default_value=0.0)
        aesthetics = _safe_float01(row[self.aesthetics_col], default_value=0.0)
        wa = max(0.0, self.quality_artifact_weight)
        ws = max(0.0, self.quality_aesthetics_weight)
        denom = wa + ws
        if denom <= 0:
            raise ValueError("RichHF quality component weights must sum to a positive value.")
        return float(np.clip((wa * artifact + ws * aesthetics) / denom, 0.0, 1.0))

    def _get_quality_components(self, row) -> Tuple[float, float, float, float]:
        """Return component scores and validity weights for optional supervision."""
        artifact_valid = float(self.artifact_col in row.index)
        aesthetics_valid = float(self.aesthetics_col in row.index)
        quality = self._get_quality_score(row)
        artifact = (
            _safe_float01(row[self.artifact_col], default_value=quality)
            if artifact_valid else quality
        )
        aesthetics = (
            _safe_float01(row[self.aesthetics_col], default_value=quality)
            if aesthetics_valid else quality
        )
        return artifact, aesthetics, artifact_valid, aesthetics_valid

    def _build_align_weight(self, align_score: float, quality_score: float, overall_score: float) -> float:
        align_score = float(np.clip(align_score, 0.0, 1.0))
        quality_score = float(np.clip(quality_score, 0.0, 1.0))
        overall_score = float(np.clip(overall_score, 0.0, 1.0))
        mode = self.align_weight_mode
        if mode in {"human_align", "align", "alignment"}:
            return align_score
        if mode == "overall":
            return overall_score
        if mode == "quality_align_mix":
            return 0.5 * (align_score + quality_score)
        raise ValueError(
            f"Unknown RichHF align_weight_mode={self.align_weight_mode}. "
            "Choose from: human_align, overall, quality_align_mix."
        )

    def _build_prompt_answer_weights(self, align_weight: float, answer_conf: float) -> Tuple[float, float]:
        align_weight = float(np.clip(align_weight, 0.0, 1.0))
        answer_conf = float(np.clip(answer_conf, 0.0, 1.0))
        mode = self.prompt_answer_weight_mode
        if mode == "zero":
            pos_w = 0.0
        elif mode in {"answer_confidence", "confidence"}:
            pos_w = answer_conf
        elif mode in {"align", "human_align"}:
            pos_w = align_weight
        elif mode in {"align_confidence", "align_answer_confidence"}:
            pos_w = align_weight * answer_conf
        else:
            raise ValueError(
                f"Unknown RichHF prompt_answer_weight_mode={self.prompt_answer_weight_mode}. "
                "Choose from: zero, answer_confidence, align, align_confidence."
            )
        gap_w = (1.0 - align_weight) * answer_conf if self.use_prompt_answer_gap else 0.0
        return float(pos_w), float(gap_w)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pil = _read_pil(self._path(row))
        prompt = str(row[self.prompt_col]).strip()
        answer = str(row[self.answer_col]).strip()
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        quality = self._get_quality_score(row)
        artifact, aesthetics, artifact_weight, aesthetics_weight = (
            self._get_quality_components(row)
        )
        align = _safe_float01(row[self.align_col], default_value=0.0)
        rank_target = _safe_float01(row[self.rank_target_col], default_value=quality)
        answer_conf = self._get_answer_confidence(row)
        align_weight = self._build_align_weight(align, quality, rank_target)
        prompt_answer_weight, prompt_answer_gap_weight = self._build_prompt_answer_weights(align_weight, answer_conf)
        answer_weight = answer_conf if self.answer_weight_mode in {"answer_confidence", "confidence"} else 1.0
        image_clean = None
        map_clean = None
        use_three_views = self.supervision_view in {"three_view", "separate_clean"}
        if self.use_local_feedback:
            map_path = self._misalignment_map_path(row)
            if map_path is None:
                raise FileNotFoundError(
                    f"RichHF-18K misalignment map not found for {row[self.name_col]}. "
                    "Run tools/prepare_richhf18k.py with the updated exporter first."
                )
            spatial_label = _read_pil(map_path).convert("L")
            spatial_label = mirror_pil_if_applied(spatial_label, mirror_applied)
            if use_three_views:
                # Canonical Image-Image SimSiam gets two independent augmented
                # views; human and cross-modal labels use a third clean view.
                image_clean, map_clean = self.clean_transform.with_mask(pil, spatial_label)
                image1, map1 = self.quality_safe_transform.with_mask(pil, spatial_label)
                image2, map2 = self.quality_safe_transform.with_mask(pil, spatial_label)
            elif self.train and self.supervision_view == "clean":
                # View 1 preserves the complete image and its human annotation.
                # View 2 remains stochastic and is used by image-view SimSiam.
                image1, map1 = self.clean_transform.with_mask(pil, spatial_label)
                image2, map2 = self.transform.with_mask(pil, spatial_label)
            elif self.train and self.supervision_view == "quality_safe":
                # Both views preserve the full frame and all quality evidence.
                image1, map1 = self.quality_safe_transform.with_mask(pil, spatial_label)
                image2, map2 = self.quality_safe_transform.with_mask(pil, spatial_label)
            else:
                image1, map1 = self.transform.with_mask(pil, spatial_label)
                image2, map2 = self.transform.with_mask(pil, spatial_label)
            prompt_word_labels = str(row[self.prompt_misalignment_col]).strip()
        else:
            if use_three_views:
                image_clean = self.clean_transform(pil)
                image1 = self.quality_safe_transform(pil)
                image2 = self.quality_safe_transform(pil)
            elif self.train and self.supervision_view == "clean":
                image1 = self.clean_transform(pil)
                image2 = self.transform(pil)
            elif self.train and self.supervision_view == "quality_safe":
                image1 = self.quality_safe_transform(pil)
                image2 = self.quality_safe_transform(pil)
            else:
                image1 = self.transform(pil)
                image2 = self.transform(pil)

        sample = {
            "image1": image1,
            "image2": image2,
            "prompt": prompt,
            "answer": answer,
            "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32),
            "image_weight": torch.tensor(1.0, dtype=torch.float32),
            "align_score": torch.tensor(align, dtype=torch.float32),
            "align_weight": torch.tensor(align_weight, dtype=torch.float32),
            "answer_weight": torch.tensor(answer_weight, dtype=torch.float32),
            "answer_confidence": torch.tensor(answer_conf, dtype=torch.float32),
            "prompt_answer_score": torch.tensor(prompt_answer_weight, dtype=torch.float32),
            "prompt_answer_weight": torch.tensor(prompt_answer_weight, dtype=torch.float32),
            "prompt_answer_gap_weight": torch.tensor(prompt_answer_gap_weight, dtype=torch.float32),
            "mos": torch.tensor(rank_target, dtype=torch.float32),
            "quality_score": torch.tensor(quality, dtype=torch.float32),
            "quality_weight": torch.tensor(1.0, dtype=torch.float32),
            "quality_loss_weight": torch.tensor(
                1.0 if self.quality_loss_weights is None else self.quality_loss_weights[idx],
                dtype=torch.float32,
            ),
            "artifact_loss_weight": torch.tensor(
                1.0
                if self.quality_component_loss_weights is None
                else self.quality_component_loss_weights["artifact"][idx],
                dtype=torch.float32,
            ),
            "aesthetics_loss_weight": torch.tensor(
                1.0
                if self.quality_component_loss_weights is None
                else self.quality_component_loss_weights["aesthetics"][idx],
                dtype=torch.float32,
            ),
            "artifact_score": torch.tensor(artifact, dtype=torch.float32),
            "aesthetics_score": torch.tensor(aesthetics, dtype=torch.float32),
            "artifact_weight": torch.tensor(artifact_weight, dtype=torch.float32),
            "aesthetics_weight": torch.tensor(aesthetics_weight, dtype=torch.float32),
            "split_group": _normalize_prompt_group(prompt),
            "dataset_id": "richhf18k",
        }
        if image_clean is not None:
            sample["image_clean"] = image_clean
        if self.use_local_feedback:
            sample.update({
                "misalignment_map1": map1,
                "misalignment_map2": map2,
                "misalignment_map_valid": torch.tensor(1.0, dtype=torch.float32),
                # RichHF convention: 0 means a misaligned word, 1 means aligned.
                "prompt_word_labels": prompt_word_labels,
            })
            if map_clean is not None:
                sample["misalignment_map_clean"] = map_clean
        return sample


def _build_single_dataset(cfg: Dict, train: bool = True) -> Dataset:
    typ = str(cfg.get("type", "")).lower()
    if typ == "agiqa3k":
        return AGIQA3KPretrainDataset(cfg, train=train)
    if typ == "agiqa1k":
        return AGIQA1KPretrainDataset(cfg, train=train)
    if typ == "aigciqa2023":
        return AIGCIQA2023PretrainDataset(cfg, train=train)
    if typ in {"aigiqa20k", "aigiqa-20k", "20k"}:
        return AIGIQA20KPretrainDataset(cfg, train=train)
    if typ in {"richhf18k", "richhf-18k", "richhf"}:
        return RichHF18KPretrainDataset(cfg, train=train)
    raise ValueError(f"Unsupported mixed pretrain dataset type: {typ}")


def _split_indices(total: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g).tolist()
    tr = int(total * train_ratio)
    va = int(total * val_ratio)
    return idx[:tr], idx[tr:tr + va], idx[tr + va:]


def _split_indices_by_group(groups: List[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    group_to_indices: Dict[str, List[int]] = {}
    for idx, group in enumerate(groups):
        group_to_indices.setdefault(str(group), []).append(idx)

    rng = random.Random(seed)
    unique_groups = list(group_to_indices.keys())
    rng.shuffle(unique_groups)

    total = len(groups)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for group in unique_groups:
        bucket = group_to_indices[group]
        if len(train_idx) < train_target:
            train_idx.extend(bucket)
        elif len(val_idx) < val_target:
            val_idx.extend(bucket)
        else:
            test_idx.extend(bucket)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


class SubsetWithWeights(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int], sampling_weight: float):
        self.dataset = dataset
        self.indices = indices
        self.sampling_weight = float(sampling_weight)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def get_quality_relation_metadata(self) -> List[Dict | None]:
        if not hasattr(self.dataset, "get_quality_relation_metadata"):
            return [None] * len(self.indices)
        metadata = self.dataset.get_quality_relation_metadata()
        return [metadata[index] for index in self.indices]


def _get_train_sampling_weights(
    dataset: Dataset,
    indices: List[int],
    base_weight: float,
    sampling_cfg: Dict,
) -> List[float]:
    """Combine dataset-level and optional label-balanced row weights."""
    base = float(base_weight)
    if bool(sampling_cfg.get("enabled", False)) and hasattr(dataset, "get_quality_sampling_weights"):
        all_weights = dataset.get_quality_sampling_weights(sampling_cfg)
        return [base * float(all_weights[index]) for index in indices]
    return [base] * len(indices)


class QualityRelationBatchSampler(Sampler[List[int]]):
    """Build batches with explicit same-prompt quality pairs or triplets."""

    def __init__(
        self,
        metadata: List[Dict | None],
        weights: List[float],
        batch_size: int,
        num_samples: int,
        relation_cfg: Dict,
        seed: int,
    ):
        self.metadata = metadata
        self.weights = [float(value) for value in weights]
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.triplets_per_batch = int(relation_cfg.get("triplets_per_batch", 0))
        self.near_pairs_per_batch = int(relation_cfg.get("near_pairs_per_batch", 1))
        self.far_pairs_per_batch = int(relation_cfg.get("far_pairs_per_batch", 1))
        self.near_quality_max_gap = float(relation_cfg.get("near_quality_max_gap", 0.03))
        self.far_quality_min_gap = float(relation_cfg.get("far_quality_min_gap", 0.15))
        self.align_max_gap = float(relation_cfg.get("align_max_gap", 0.05))

        required_slots = (
            3 * self.triplets_per_batch
            + 2 * (self.near_pairs_per_batch + self.far_pairs_per_batch)
        )
        if self.batch_size < required_slots:
            raise ValueError(
                f"quality_relation_siam needs batch_size >= {required_slots}, "
                f"got {self.batch_size}."
            )
        if len(self.metadata) != len(self.weights):
            raise ValueError("Quality relation metadata and sampling weights have different lengths.")

        groups: Dict[Tuple[str, str], List[int]] = {}
        for index, item in enumerate(self.metadata):
            if not item:
                continue
            key = (str(item.get("dataset_id", "")), str(item.get("prompt_group", "")))
            groups.setdefault(key, []).append(index)

        self.near_pairs: List[Tuple[int, int]] = []
        self.far_pairs: List[Tuple[int, int]] = []
        self.triplets: List[Tuple[int, int, int]] = []
        for (dataset_id, _prompt_group), indices in groups.items():
            if dataset_id != "richhf18k":
                continue
            near_by_anchor: Dict[int, List[int]] = {index: [] for index in indices}
            far_by_anchor: Dict[int, List[int]] = {index: [] for index in indices}
            for offset, left in enumerate(indices):
                left_meta = self.metadata[left]
                for right in indices[offset + 1:]:
                    right_meta = self.metadata[right]
                    quality_gap = abs(float(left_meta["quality"]) - float(right_meta["quality"]))
                    align_gap = abs(float(left_meta["align"]) - float(right_meta["align"]))
                    if align_gap > self.align_max_gap:
                        continue
                    if quality_gap <= self.near_quality_max_gap:
                        self.near_pairs.append((left, right))
                        near_by_anchor[left].append(right)
                        near_by_anchor[right].append(left)
                    if quality_gap >= self.far_quality_min_gap:
                        self.far_pairs.append((left, right))
                        far_by_anchor[left].append(right)
                        far_by_anchor[right].append(left)
            for anchor in indices:
                for near in near_by_anchor[anchor]:
                    for far in far_by_anchor[anchor]:
                        self.triplets.append((anchor, near, far))

        if self.triplets_per_batch > 0 and not self.triplets:
            raise RuntimeError("No same-prompt Anchor/Near/Far triplets satisfy quality_relation_siam.")
        if self.near_pairs_per_batch > 0 and not self.near_pairs:
            raise RuntimeError("No same-prompt near-quality pairs satisfy quality_relation_siam.")
        if self.far_pairs_per_batch > 0 and not self.far_pairs:
            raise RuntimeError("No same-prompt far-quality pairs satisfy quality_relation_siam.")
        print(
            "[QualityRelationBatchSampler] "
            f"triplets={len(self.triplets)}, near_pairs={len(self.near_pairs)}, "
            f"far_pairs={len(self.far_pairs)}, triplets_per_batch={self.triplets_per_batch}, "
            f"near_per_batch={self.near_pairs_per_batch}, "
            f"far_per_batch={self.far_pairs_per_batch}"
        )

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        population = list(range(len(self.metadata)))

        for _ in range(len(self)):
            batch: List[int] = []
            used = set()

            def add_relation(pool) -> None:
                for _attempt in range(128):
                    relation = pool[rng.randrange(len(pool))]
                    if all(index not in used for index in relation):
                        batch.extend(relation)
                        used.update(relation)
                        return
                raise RuntimeError(
                    "Could not place the requested disjoint quality relations in one batch. "
                    "Reduce triplets_per_batch/near_pairs_per_batch/far_pairs_per_batch."
                )

            for _triplet_index in range(self.triplets_per_batch):
                add_relation(self.triplets)
            for _pair_index in range(self.near_pairs_per_batch):
                add_relation(self.near_pairs)
            for _pair_index in range(self.far_pairs_per_batch):
                add_relation(self.far_pairs)

            while len(batch) < self.batch_size:
                index = rng.choices(population, weights=self.weights, k=1)[0]
                if index not in used:
                    batch.append(index)
                    used.add(index)
            rng.shuffle(batch)
            yield batch


def build_mixed_pretrain_dataloaders(cfg: Dict):
    dcfg = cfg["dataset"]
    lcfg = cfg["loader"]
    train_sets, val_sets, test_sets = [], [], []
    train_weights = []
    train_relation_metadata: List[Dict | None] = []
    seed = int(cfg.get("split_seed", cfg.get("seed", 42)))
    train_ratio = float(dcfg.get("train_ratio", 0.7))
    val_ratio = float(dcfg.get("val_ratio", 0.1))
    tcfg = cfg.get("train", {})
    local_feedback_requested = (
        float(tcfg.get("lambda_img_prompt_local", 0.0)) > 0.0
        or float(tcfg.get("lambda_prompt_answer_word", 0.0)) > 0.0
    )
    for i, item in enumerate(dcfg["datasets"]):
        item_cfg = {**item, "input_size": dcfg.get("input_size", item.get("input_size", 224))}
        item_type = str(item_cfg.get("type", "")).lower()
        item_cfg["supervision_view"] = tcfg.get(
            "human_supervision_view", item_cfg.get("supervision_view", "augmented")
        )
        item_cfg["quality_loss_reweighting"] = tcfg.get(
            "quality_loss_reweighting", {}
        )
        item_cfg["mirror_augmentation"] = tcfg.get(
            "mirror_augmentation", item_cfg.get("mirror_augmentation", {})
        )
        if item_type in {"richhf18k", "richhf-18k", "richhf"}:
            if local_feedback_requested:
                item_cfg["use_local_feedback"] = True
        split_strategy = str(item.get("split_strategy", dcfg.get("split_strategy", "random"))).lower()
        if split_strategy in {"official", "predefined", "csv_split"}:
            train_cfg = {**item_cfg, "split": item.get("train_split", "train")}
            val_cfg = {**item_cfg, "split": item.get("val_split", "val")}
            test_cfg = {**item_cfg, "split": item.get("test_split", "test")}
            ds_train = _build_single_dataset(train_cfg, train=True)
            ds_val = _build_single_dataset(val_cfg, train=False)
            ds_test = _build_single_dataset(test_cfg, train=False)
            w = float(item.get("sampling_weight", 1.0))
            tr_subset = SubsetWithWeights(ds_train, list(range(len(ds_train))), w)
            train_sets.append(tr_subset)
            train_relation_metadata.extend(tr_subset.get_quality_relation_metadata())
            val_sets.append(ds_val)
            test_sets.append(ds_test)
            train_weights.extend(
                _get_train_sampling_weights(
                    ds_train,
                    list(range(len(ds_train))),
                    w,
                    tcfg.get("quality_balanced_sampling", {}),
                )
            )
            print(
                f"[MixedPretrain] {item.get('name', item.get('type'))}: "
                f"train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}, "
                f"split={split_strategy}, sampling_weight={w}"
            )
            continue

        ds_train = _build_single_dataset(item_cfg, train=True)
        ds_eval = _build_single_dataset(item_cfg, train=False)
        if split_strategy in {"group_prompt", "prompt_group", "group"} and hasattr(ds_train, "get_split_groups"):
            tr_idx, va_idx, te_idx = _split_indices_by_group(ds_train.get_split_groups(), train_ratio, val_ratio, seed + i)
        else:
            tr_idx, va_idx, te_idx = _split_indices(len(ds_train), train_ratio, val_ratio, seed + i)
        w = float(item.get("sampling_weight", 1.0))
        tr_subset = SubsetWithWeights(ds_train, tr_idx, w)
        train_sets.append(tr_subset)
        train_relation_metadata.extend(tr_subset.get_quality_relation_metadata())
        val_sets.append(torch.utils.data.Subset(ds_eval, va_idx))
        test_sets.append(torch.utils.data.Subset(ds_eval, te_idx))
        train_weights.extend(
            _get_train_sampling_weights(
                ds_train,
                tr_idx,
                w,
                tcfg.get("quality_balanced_sampling", {}),
            )
        )
        print(
            f"[MixedPretrain] {item.get('name', item.get('type'))}: "
            f"total={len(ds_train)}, train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}, "
            f"split={split_strategy}, sampling_weight={w}"
        )

    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)
    test_dataset = ConcatDataset(test_sets)
    num_samples = int(dcfg.get("num_samples_per_epoch", len(train_weights)))
    relation_cfg = tcfg.get("quality_relation_siam", {})
    if bool(relation_cfg.get("enabled", False)):
        batch_sampler = QualityRelationBatchSampler(
            metadata=train_relation_metadata,
            weights=train_weights,
            batch_size=int(lcfg.get("batch_size", 8)),
            num_samples=num_samples,
            relation_cfg=relation_cfg,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=lcfg.get("num_workers", 4),
            pin_memory=lcfg.get("pin_memory", True),
        )
    else:
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(train_weights),
            num_samples=num_samples,
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=lcfg.get("batch_size", 8),
            sampler=sampler,
            num_workers=lcfg.get("num_workers", 4),
            pin_memory=lcfg.get("pin_memory", True),
            drop_last=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=lcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=lcfg.get("num_workers", 4),
        pin_memory=lcfg.get("pin_memory", True),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=lcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=lcfg.get("num_workers", 4),
        pin_memory=lcfg.get("pin_memory", True),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
