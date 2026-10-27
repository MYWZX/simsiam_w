# AGIQA-1K 和 AIGCIQA2023 微调数据集
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split

from utils.data_protocols import loader_rng_kwargs, fit_1k_train_scaler
from utils.perturbations import perturb_image
from utils.saliency import compute_simple_saliency
from utils.mirror_augmentation import maybe_mirror_sample, resolve_mirror_config


class IQAImageTransform:
    def __init__(
        self, input_size: int = 224, train: bool = False, horizontal_flip_prob: float = 0.5
    ):
        self.input_size = int(input_size)
        self.train = bool(train)
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
            x0 = random.randint(0, max(0, w - cw)); y0 = random.randint(0, max(0, h - ch))
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

    def to_tensor(self, pil_img: Image.Image, output_size: int | None = None) -> torch.Tensor:
        """Resize and ImageNet-normalize one already prepared PIL image."""
        img = pil_img.convert("RGB")
        size = int(output_size if output_size is not None else self.input_size)
        img = img.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def fixed_crops(
        self,
        pil_img: Image.Image,
        crop_scale: float = 0.65,
        num_crops: int = 5,
        output_size: int | None = None,
    ) -> torch.Tensor:
        """Crop at native resolution, then resize each crop for the local encoder.

        The canonical order is top-left, top-right, center, bottom-left and
        bottom-right. Three-crop mode keeps top-left/center/bottom-right and is
        useful as a cheaper ablation. No color or contrast transform is applied,
        because those operations can change the quality represented by the MOS.
        """
        img = pil_img.convert("RGB")
        scale = float(crop_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"local crop_scale must be in (0, 1], got {scale}")
        if int(num_crops) not in {1, 3, 5}:
            raise ValueError(f"local num_crops must be 1, 3 or 5, got {num_crops}")

        width, height = img.size
        crop_w = max(8, min(width, int(round(width * scale))))
        crop_h = max(8, min(height, int(round(height * scale))))
        max_x = max(0, width - crop_w)
        max_y = max(0, height - crop_h)
        boxes = [
            (0, 0),
            (max_x, 0),
            (max_x // 2, max_y // 2),
            (0, max_y),
            (max_x, max_y),
        ]
        if int(num_crops) == 1:
            boxes = [boxes[2]]
        elif int(num_crops) == 3:
            boxes = [boxes[0], boxes[2], boxes[4]]

        crops = []
        for x0, y0 in boxes:
            crop = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            crops.append(self.to_tensor(crop, output_size=output_size))
        return torch.stack(crops, dim=0)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.convert("RGB")
        if self.train:
            img = self._random_resized_crop(img)
            if random.random() < self.horizontal_flip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self._jitter(img)
        return self.to_tensor(img)


def _read_pil_and_bgr(path: Path):
    try:
        pil = Image.open(path).convert("RGB")
        rgb = np.asarray(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return pil, bgr
    except Exception:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), bgr


def _norm_score(x: float, max_value: float = 5.0, min_value: float = 0.0) -> float:
    return float(np.clip((float(x) - min_value) / (max_value - min_value + 1e-8), 0.0, 1.0))


def _numeric_sort_key(path: Path):
    try: return (path.parent.as_posix(), int(path.stem))
    except Exception: return (path.parent.as_posix(), path.stem)


def _load_mat_vector(path: Path, key: str | None = None) -> np.ndarray:
    data = loadmat(str(path))
    if key:
        if key not in data:
            raise KeyError(f"MAT score key {key!r} missing from {path}; available keys: {[k for k in data if not k.startswith('__')]}")
        arr = data[key]
    else:
        candidates = [(k, v) for k, v in data.items() if not k.startswith("__")
                      and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
                      and v.size > 0 and np.squeeze(v).ndim <= 1]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one numeric score vector in {path}; specify its MAT key explicitly")
        arr = candidates[0][1]
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number) or arr.size == 0 or np.squeeze(arr).ndim > 1:
        raise ValueError(f"Expected a nonempty numeric score vector in {path}, got shape {arr.shape}")
    result = arr.reshape(-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite MOS values in {path}")
    return result


def _load_prompts_from_file(path: Path) -> List[str]:
    if not path.is_file(): raise FileNotFoundError(f"Configured prompt file not found: {path}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path).fillna("")
    else:
        df = pd.read_csv(path).fillna("")
    cols = list(df.columns)
    preferred = [c for c in cols if "prompt" in str(c).lower() or "text" in str(c).lower() or "提示" in str(c)]
    use_col = preferred[0] if preferred else None
    if use_col is None:
        for c in cols:
            if df[c].dtype == object:
                use_col = c; break
    if use_col is None: return [""] * len(df)
    return [str(x).strip() for x in df[use_col].tolist()]


class BaseIQADataset(Dataset):
    def _setup_visual_pipeline(
        self,
        input_size: int,
        train_augmentation_mode: str = "legacy",
        quality_safe_flip_prob: float = 0.5,
        local_multicrop: Dict | None = None,
    ) -> None:
        self.mirror_cfg = resolve_mirror_config(self._mirror_augmentation, self.train)
        mirror_flip_prob = (
            0.0
            if self.mirror_cfg["enabled"] and self.mirror_cfg["disable_random_horizontal_flip"]
            else 0.5
        )
        self.train_transform = IQAImageTransform(
            input_size, train=True, horizontal_flip_prob=mirror_flip_prob
        )
        self.eval_transform = IQAImageTransform(
            input_size, train=False, horizontal_flip_prob=mirror_flip_prob
        )
        self.train_augmentation_mode = str(train_augmentation_mode).strip().lower()
        if self.train_augmentation_mode not in {"legacy", "quality_safe"}:
            raise ValueError(
                "train_augmentation_mode must be 'legacy' or 'quality_safe', "
                f"got {self.train_augmentation_mode!r}"
            )
        self.quality_safe_flip_prob = (
            0.0
            if self.mirror_cfg["enabled"] and self.mirror_cfg["disable_random_horizontal_flip"]
            else float(quality_safe_flip_prob)
        )
        if not 0.0 <= self.quality_safe_flip_prob <= 1.0:
            raise ValueError(
                "quality_safe_flip_prob must be in [0, 1], "
                f"got {self.quality_safe_flip_prob}"
            )

        multicrop = dict(local_multicrop or {})
        self.local_multicrop_enabled = bool(multicrop.get("enabled", False))
        self.local_crop_scale = float(multicrop.get("crop_scale", 0.65))
        self.local_crop_input_size = int(multicrop.get("input_size", input_size))
        self.local_train_num_crops = int(multicrop.get("train_num_crops", 5))
        self.local_eval_num_crops = int(multicrop.get("eval_num_crops", 5))

    def _make_visual_inputs(self, pil_img: Image.Image) -> Dict[str, torch.Tensor]:
        """Build a whole-image global view and optional native-resolution crops."""
        source = pil_img.convert("RGB")
        if self.train and self.train_augmentation_mode == "quality_safe":
            # Apply only a shared horizontal flip. Crop, brightness, contrast and
            # color jitter can alter the image quality while leaving MOS unchanged.
            if random.random() < self.quality_safe_flip_prob:
                source = source.transpose(Image.FLIP_LEFT_RIGHT)
            image = self.eval_transform(source)
        elif self.train:
            image = self.train_transform(source)
        else:
            image = self.eval_transform(source)

        visual = {"image": image}
        if self.local_multicrop_enabled:
            num_crops = self.local_train_num_crops if self.train else self.local_eval_num_crops
            visual["local_crops"] = self.eval_transform.fixed_crops(
                source,
                crop_scale=self.local_crop_scale,
                num_crops=num_crops,
                output_size=self.local_crop_input_size,
            )
        return visual

    def _make_perturbed_tensor(self, image_bgr: np.ndarray, strength: str) -> torch.Tensor:
        if self.train and self.use_perturb and random.random() < self.perturbation_prob:
            saliency = compute_simple_saliency(image_bgr)
            image_pert, _ = perturb_image(image_bgr, saliency, strength=strength, saliency_sampling_ratio=self.saliency_sampling_ratio, perturbation_prob=1.0)
        else:
            image_pert = image_bgr
        rgb = cv2.cvtColor(image_pert, cv2.COLOR_BGR2RGB)
        return self.eval_transform(Image.fromarray(rgb))

    def _add_perturb(self, sample: Dict, bgr: np.ndarray):
        if self.use_perturb and self.perturbation_strengths:
            for strength in self.perturbation_strengths:
                sample[f"image_{strength}"] = self._make_perturbed_tensor(bgr, strength)
        return sample


class AGIQA1KIQADataset(BaseIQADataset):
    def __init__(self, root_dir: str, image_dir: str = "file", csv_file: str = "AIGC-1K_answer.csv", input_size: int = 224, train: bool = True, mos_max: float | None = None, use_perturb: bool = True, perturbation_prob: float = 0.7, perturbation_strengths: Sequence[str] | None = None, saliency_sampling_ratio: float = 0.7, final_quality_weight: float = 1.0, final_align_weight: float = 0.0, train_augmentation_mode: str = "legacy", quality_safe_flip_prob: float = 0.5, local_multicrop: Dict | None = None, mirror_augmentation: Dict | None = None, mos_normalization: str = "legacy_full_data"):
        self.root = Path(root_dir); self.image_root = self.root / image_dir; self.csv_path = self.root / csv_file
        self.train = bool(train); self.use_perturb = bool(use_perturb); self.perturbation_prob = float(perturbation_prob); self.saliency_sampling_ratio = float(saliency_sampling_ratio)
        self.perturbation_strengths = [str(s) for s in (perturbation_strengths or [])]
        self.final_quality_weight = float(final_quality_weight); self.final_align_weight = float(final_align_weight)
        self._mirror_augmentation = mirror_augmentation or {}
        self._setup_visual_pipeline(input_size, train_augmentation_mode, quality_safe_flip_prob, local_multicrop)
        df = pd.read_csv(self.csv_path).fillna("")
        required = {"name", "prompt", "MOS"}
        missing = required - set(df.columns)
        if missing: raise RuntimeError(f"AGIQA-1K missing columns: {missing}")
        self.df = df.reset_index(drop=True)
        scores = self.df["MOS"].astype(float).values
        if mos_normalization not in {"legacy_full_data", "train_minmax"}:
            raise ValueError("Unsupported 1K MOS normalization mode")
        # Fitted by the loader after generating training indices, never on test labels.
        if mos_normalization == "train_minmax":
            self.min_mos, self.max_mos = 0.0, 1.0
        else:
            self.min_mos = float(np.min(scores)); self.max_mos = float(mos_max) if mos_max else float(np.max(scores))

    def __len__(self): return len(self.df)
    def _path(self, name):
        for p in [self.image_root/name, self.root/name, self.root/"file"/name, self.root/"image"/name, self.root/"images"/name]:
            if p.exists(): return p
        raise FileNotFoundError(f"Image not found: {name}")
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; path = self._path(str(row["name"])); pil, bgr = _read_pil_and_bgr(path)
        prompt = str(row["prompt"]).strip()
        answer = str(row.get("answer", row["prompt"])).strip()
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        if mirror_applied:
            bgr = cv2.flip(bgr, 1)
        q = _norm_score(float(row["MOS"]), self.max_mos, self.min_mos)
        a = 0.0; final = self.final_quality_weight*q + self.final_align_weight*a
        sample = {**self._make_visual_inputs(pil), "quality_score": torch.tensor(q, dtype=torch.float32), "align_score": torch.tensor(a, dtype=torch.float32), "score": torch.tensor(final, dtype=torch.float32), "quality_mask": torch.tensor(1.0), "align_mask": torch.tensor(0.0), "final_mask": torch.tensor(1.0), "prompt": prompt, "answer": answer, "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32), "path": str(path)}
        return self._add_perturb(sample, bgr)


class AIGCIQA2023IQADataset(BaseIQADataset):
    def __init__(self, root_dir: str, image_dir: str = "Image/allimg", input_size: int = 224, train: bool = True, mosz1_file: str = "mosz1.mat", mosz3_file: str = "mosz3.mat", mosz1_key: str = "MOSz1", mosz3_key: str = "MOSz3", prompt_file: str = "", images_per_prompt: int = 4, num_prompts: int = 100, prompt_cycle_by_round: bool = True, debug_prompt_mapping: bool = False, mos_max: float | None = None, use_perturb: bool = True, perturbation_prob: float = 0.7, perturbation_strengths: Sequence[str] | None = None, saliency_sampling_ratio: float = 0.7, final_quality_weight: float = 0.5, final_align_weight: float = 0.5, train_augmentation_mode: str = "legacy", quality_safe_flip_prob: float = 0.5, local_multicrop: Dict | None = None, mirror_augmentation: Dict | None = None):
        self.root = Path(root_dir); self.image_root = self.root / image_dir
        self.train = bool(train); self.use_perturb = bool(use_perturb); self.perturbation_prob = float(perturbation_prob); self.saliency_sampling_ratio = float(saliency_sampling_ratio); self.perturbation_strengths = [str(s) for s in (perturbation_strengths or [])]
        self.final_quality_weight = float(final_quality_weight); self.final_align_weight = float(final_align_weight)
        self._mirror_augmentation = mirror_augmentation or {}
        self._setup_visual_pipeline(input_size, train_augmentation_mode, quality_safe_flip_prob, local_multicrop)
        q = _load_mat_vector(self.root / mosz1_file, mosz1_key if mosz1_key else None)
        a = _load_mat_vector(self.root / mosz3_file, mosz3_key if mosz3_key else None)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        imgs = sorted([p for p in self.image_root.rglob("*") if p.is_file() and p.suffix.lower() in exts], key=_numeric_sort_key)
        if not imgs:
            raise ValueError(f"No AIGCIQA2023 images found in {self.image_root}")
        if len(imgs) != len(q) or len(imgs) != len(a):
            raise ValueError(f"AIGCIQA2023 image/MOS count mismatch: images={len(imgs)}, quality={len(q)}, align={len(a)}; positional labels must match exactly")
        self.images = imgs; self.quality = q; self.align = a
        mx = mos_max if mos_max is not None else (100.0 if max(np.nanmax(self.quality), np.nanmax(self.align)) > 10 else 5.0)
        self.mos_max = float(mx); self.prompts = _load_prompts_from_file(self.root / prompt_file) if prompt_file else []
        self.images_per_prompt = int(images_per_prompt)
        self.num_prompts = int(num_prompts if num_prompts else (len(self.prompts) if self.prompts else 100))
        self.prompt_cycle_by_round = bool(prompt_cycle_by_round)
        self.debug_prompt_mapping = bool(debug_prompt_mapping)
        if self.debug_prompt_mapping:
            print("[AIGCIQA2023IQA] prompt mapping preview:")
            for i in [0,1,2,3,4,400,401,402,403,800,801,802,803]:
                if i < len(self.images):
                    print(f"idx={i}, img={self.images[i].name}, prompt={self._prompt(i)[:80]}")
    def __len__(self): return len(self.images)
    def _prompt(self, idx):
        if not self.prompts: return ""
        if len(self.prompts) == len(self.images): return self.prompts[idx]
        num_prompts = len(self.prompts)
        images_per_prompt = max(1, self.images_per_prompt)
        if self.prompt_cycle_by_round:
            images_per_round = num_prompts * images_per_prompt
            pidx = (idx % images_per_round) // images_per_prompt
        else:
            pidx = idx // images_per_prompt
        return self.prompts[min(pidx, num_prompts - 1)]
    def __getitem__(self, idx):
        path = self.images[idx]; pil, bgr = _read_pil_and_bgr(path)
        prompt = self._prompt(idx)
        answer = prompt
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        if mirror_applied:
            bgr = cv2.flip(bgr, 1)
        q = _norm_score(float(self.quality[idx]), self.mos_max); a = _norm_score(float(self.align[idx]), self.mos_max)
        final = self.final_quality_weight*q + self.final_align_weight*a
        sample = {**self._make_visual_inputs(pil), "quality_score": torch.tensor(q, dtype=torch.float32), "align_score": torch.tensor(a, dtype=torch.float32), "score": torch.tensor(final, dtype=torch.float32), "quality_mask": torch.tensor(1.0), "align_mask": torch.tensor(1.0), "final_mask": torch.tensor(1.0), "prompt": prompt, "answer": answer, "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32), "path": str(path)}
        return self._add_perturb(sample, bgr)


def build_iqa_dataloaders_from_dataset(dataset_train: Dataset, dataset_eval: Dataset, cfg: Dict):
    dcfg, lcfg = cfg["dataset"], cfg["loader"]
    total = len(dataset_train); tr = int(total*float(dcfg.get("train_ratio", 0.7))); va = int(total*float(dcfg.get("val_ratio", 0.1))); te = total-tr-va
    split_seed = int(cfg.get("split_seed", cfg.get("seed", 42)))
    g = torch.Generator().manual_seed(split_seed); idx = torch.randperm(total, generator=g).tolist()
    if dcfg.get("type") == "agiqa1k_iqa":
        fit_1k_train_scaler(dataset_train, dataset_eval, idx[:tr], cfg)
    train_set = torch.utils.data.Subset(dataset_train, idx[:tr]); val_set = torch.utils.data.Subset(dataset_eval, idx[tr:tr+va]); test_set = torch.utils.data.Subset(dataset_eval, idx[tr+va:])
    train_loader = DataLoader(train_set, batch_size=lcfg.get("batch_size",8), shuffle=True, num_workers=lcfg.get("num_workers",4), pin_memory=lcfg.get("pin_memory",True), drop_last=True, **loader_rng_kwargs(cfg, "train"))
    val_loader = DataLoader(val_set, batch_size=lcfg.get("batch_size",8), shuffle=False, num_workers=lcfg.get("num_workers",4), pin_memory=lcfg.get("pin_memory",True), drop_last=False, **loader_rng_kwargs(cfg, "val"))
    test_loader = DataLoader(test_set, batch_size=lcfg.get("batch_size",8), shuffle=False, num_workers=lcfg.get("num_workers",4), pin_memory=lcfg.get("pin_memory",True), drop_last=False, **loader_rng_kwargs(cfg, "test"))
    return train_loader, val_loader, test_loader


def build_agiqa1k_iqa_dataloaders(cfg: Dict):
    d = cfg["dataset"]
    common = dict(root_dir=d["root_dir"], image_dir=d.get("image_dir","file"), csv_file=d.get("csv_file","AIGC-1K_answer.csv"), input_size=d.get("input_size",224), mos_max=d.get("mos_max",None), use_perturb=d.get("use_perturb",True), perturbation_prob=d.get("perturbation_prob",0.7), perturbation_strengths=d.get("perturbation_strengths",[]), saliency_sampling_ratio=d.get("saliency_sampling_ratio",0.7), final_quality_weight=d.get("final_quality_weight",1.0), final_align_weight=d.get("final_align_weight",0.0), train_augmentation_mode=d.get("train_augmentation_mode","legacy"), quality_safe_flip_prob=d.get("quality_safe_flip_prob",0.5), local_multicrop=d.get("local_multicrop",{}), mirror_augmentation=d.get("mirror_augmentation",{}))
    common["mos_normalization"] = d.get("mos_normalization", "legacy_full_data")
    return build_iqa_dataloaders_from_dataset(AGIQA1KIQADataset(train=True, **common), AGIQA1KIQADataset(train=False, **common), cfg)


def build_aigciqa2023_iqa_dataloaders(cfg: Dict):
    d = cfg["dataset"]
    common = dict(root_dir=d["root_dir"], image_dir=d.get("image_dir","Image/allimg"), input_size=d.get("input_size",224), mosz1_file=d.get("mosz1_file","mosz1.mat"), mosz3_file=d.get("mosz3_file","mosz3.mat"), mosz1_key=d.get("mosz1_key","MOSz1"), mosz3_key=d.get("mosz3_key","MOSz3"), prompt_file=d.get("prompt_file",""), images_per_prompt=d.get("images_per_prompt",4), num_prompts=d.get("num_prompts",100), prompt_cycle_by_round=d.get("prompt_cycle_by_round",True), debug_prompt_mapping=d.get("debug_prompt_mapping",False), mos_max=d.get("mos_max",None), use_perturb=d.get("use_perturb",True), perturbation_prob=d.get("perturbation_prob",0.7), perturbation_strengths=d.get("perturbation_strengths",[]), saliency_sampling_ratio=d.get("saliency_sampling_ratio",0.7), final_quality_weight=d.get("final_quality_weight",0.5), final_align_weight=d.get("final_align_weight",0.5), train_augmentation_mode=d.get("train_augmentation_mode","legacy"), quality_safe_flip_prob=d.get("quality_safe_flip_prob",0.5), local_multicrop=d.get("local_multicrop",{}), mirror_augmentation=d.get("mirror_augmentation",{}))
    return build_iqa_dataloaders_from_dataset(AIGCIQA2023IQADataset(train=True, **common), AIGCIQA2023IQADataset(train=False, **common), cfg)
