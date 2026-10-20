# AGIQA-3K 主模型微调数据集，虽然名字带 simsiam，但当前 3K 微调用它
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from utils.perturbations import perturb_image
from utils.saliency import compute_simple_saliency
from utils.mirror_augmentation import maybe_mirror_sample, resolve_mirror_config


class IQAImageTransform:
    """Lightweight image transform without relying on torchvision.

    train=True: random resized crop + flip + mild color jitter.
    train=False: deterministic resize.

    The augmentation here is intentionally mild because strong augmentation can
    change image-quality labels.
    """

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
        crop_w = max(8, int(w * scale))
        crop_h = max(8, int(h * scale))
        if crop_w < w or crop_h < h:
            x0 = random.randint(0, max(0, w - crop_w))
            y0 = random.randint(0, max(0, h - crop_h))
            img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
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
            img = self._random_resized_crop(img)
            if random.random() < self.horizontal_flip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self._jitter(img)
        else:
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)

        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


class AGIQA3KSimSiamDataset(Dataset):
    """AGIQA-3K dataset for cross-modal SimSiam + perturbation ranking.

    Expected CSV columns:
        name, prompt, adj1, adj2, style, mos_quality, mos_align, answer

    Outputs:
        image1, image2: two mild augmentations of the same generated image.
        image: deterministic/evaluation-like image view, used by validation.
        image_mild/image_moderate/...: perturbed degradation views.
        image_pert: compatibility alias for the first perturbation view.
        prompt, answer
        quality_score: normalized mos_quality
        align_score: normalized mos_align
        score: weighted composite score

    Important:
        Perturbed images are NOT SimSiam positive views. They are quality
        degradation views and should only be used by L_pert_rank.
    """

    def __init__(
        self,
        root_dir: str,
        image_dir: str = "image",
        csv_file: str = "data.csv",
        input_size: int = 224,
        mos_max: float = 5.0,
        final_quality_weight: float = 0.5,
        final_align_weight: float = 0.5,
        use_full_prompt: bool = True,
        train: bool = True,
        use_perturb: bool = True,
        perturbation_prob: float = 0.7,
        perturbation_strength: str = "mild",
        perturbation_strengths: Sequence[str] | None = None,
        saliency_sampling_ratio: float = 0.7,
        verify_images: bool = False,
        mirror_augmentation: Dict | None = None,
        train_augmentation_mode: str = "legacy",
        quality_safe_flip_prob: float = 0.5,
        local_multicrop: Dict | None = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_root = self.root_dir / image_dir
        self.csv_path = self.root_dir / csv_file
        self.input_size = int(input_size)
        self.mos_max = float(mos_max)
        self.final_quality_weight = float(final_quality_weight)
        self.final_align_weight = float(final_align_weight)
        self.use_full_prompt = bool(use_full_prompt)
        self.train = bool(train)
        self.use_perturb = bool(use_perturb)
        self.perturbation_prob = float(perturbation_prob)
        self.perturbation_strength = str(perturbation_strength)
        self.saliency_sampling_ratio = float(saliency_sampling_ratio)

        if perturbation_strengths is None:
            perturbation_strengths = [self.perturbation_strength]
        self.perturbation_strengths = [str(s) for s in perturbation_strengths if str(s)]

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path).fillna("")
        required = {"name", "prompt", "mos_quality", "mos_align", "answer"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"AGIQA-3K CSV missing columns: {missing}")

        if verify_images:
            valid_rows: List[pd.Series] = []
            bad: List[str] = []
            for _, row in df.iterrows():
                name = str(row["name"])
                try:
                    path = self._resolve_image_path(name)
                    with Image.open(path) as im:
                        im.verify()
                    valid_rows.append(row)
                except Exception:
                    bad.append(name)
            if bad:
                print(f"[WARN] Skip {len(bad)} broken images. First bad: {bad[:10]}")
            df = pd.DataFrame(valid_rows).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.mirror_cfg = resolve_mirror_config(mirror_augmentation, self.train)
        mirror_flip_prob = (
            0.0
            if self.mirror_cfg["enabled"] and self.mirror_cfg["disable_random_horizontal_flip"]
            else 0.5
        )
        self.train_transform = IQAImageTransform(
            input_size=input_size, train=True, horizontal_flip_prob=mirror_flip_prob
        )
        self.eval_transform = IQAImageTransform(
            input_size=input_size, train=False, horizontal_flip_prob=mirror_flip_prob
        )
        # Reuse the same crop implementation as 1K/2023; legacy transforms
        # above remain untouched for reproducibility of existing experiments.
        from .multi_iqa_dataset import IQAImageTransform as CropTransform
        self.crop_transform = CropTransform(input_size=input_size, train=False)
        self.train_augmentation_mode = str(train_augmentation_mode).lower()
        if self.train_augmentation_mode not in {"legacy", "quality_safe"}:
            raise ValueError("train_augmentation_mode must be legacy or quality_safe")
        self.quality_safe_flip_prob = float(quality_safe_flip_prob)
        if not 0.0 <= self.quality_safe_flip_prob <= 1.0:
            raise ValueError("quality_safe_flip_prob must be in [0,1]")
        self.local_multicrop = dict(local_multicrop or {})

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, name: str) -> Path:
        name = str(name)
        candidates = [
            self.image_root / name,
            self.root_dir / name,
            self.root_dir / "image" / name,
            self.root_dir / "images" / name,
            self.root_dir / "file" / name,
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found: {name}. Tried: {[str(p) for p in candidates]}")

    def _read_pil_and_bgr(self, name: str):
        path = self._resolve_image_path(name)
        try:
            pil = Image.open(path).convert("RGB")
            rgb = np.asarray(pil)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return pil, bgr, path
        except Exception:
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image with PIL and cv2: {path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(image_rgb)
            return pil, image_bgr, path

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
        if not extra:
            return base
        return ", ".join([base] + extra)

    def _make_perturbed_tensor(self, image_bgr: np.ndarray, strength: str) -> torch.Tensor:
        """Make one perturbed degradation view.

        During eval, perturbation is disabled by using the clean image. This keeps
        validation/test deterministic while preserving keys in the batch.
        """
        if self.train and self.use_perturb and random.random() < self.perturbation_prob:
            saliency = compute_simple_saliency(image_bgr)
            image_pert, _meta = perturb_image(
                image_bgr,
                saliency,
                strength=strength,
                saliency_sampling_ratio=self.saliency_sampling_ratio,
                perturbation_prob=1.0,
            )
        else:
            image_pert = image_bgr
        rgb = cv2.cvtColor(image_pert, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.eval_transform(pil)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        name = str(row["name"])
        pil, bgr, path = self._read_pil_and_bgr(name)

        raw_quality = float(row["mos_quality"])
        raw_align = float(row["mos_align"])
        quality_score = float(np.clip(raw_quality / self.mos_max, 0.0, 1.0))
        align_score = float(np.clip(raw_align / self.mos_max, 0.0, 1.0))
        final_score = self.final_quality_weight * quality_score + self.final_align_weight * align_score

        prompt = self._make_prompt(row)
        answer = str(row["answer"]).strip()
        pil, prompt, answer, mirror_applied = maybe_mirror_sample(
            pil, prompt, answer, self.mirror_cfg
        )
        if mirror_applied:
            bgr = cv2.flip(bgr, 1)

        if self.train and self.train_augmentation_mode == "quality_safe":
            if random.random() < self.quality_safe_flip_prob:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
                bgr = cv2.flip(bgr, 1)
            image1 = self.eval_transform(pil)
            image2 = image1
        elif self.train:
            image1 = self.train_transform(pil)
            image2 = self.train_transform(pil)
        else:
            image1 = self.eval_transform(pil)
            image2 = self.eval_transform(pil)

        sample = {
            "image": image1,
            "image1": image1,
            "image2": image2,
            "score": torch.tensor(final_score, dtype=torch.float32),
            "quality_score": torch.tensor(quality_score, dtype=torch.float32),
            "align_score": torch.tensor(align_score, dtype=torch.float32),
            "raw_quality": torch.tensor(raw_quality, dtype=torch.float32),
            "raw_align": torch.tensor(raw_align, dtype=torch.float32),
            "prompt": prompt,
            "answer": answer,
            "mirror_applied": torch.tensor(float(mirror_applied), dtype=torch.float32),
            "path": str(path),
        }

        if self.local_multicrop.get("enabled", False):
            crops = self.local_multicrop
            sample["local_crops"] = self.crop_transform.fixed_crops(
                pil, crop_scale=float(crops.get("crop_scale", .65)),
                num_crops=int(crops.get("train_num_crops" if self.train else "eval_num_crops", 5)),
                output_size=int(crops.get("input_size", self.input_size)),
            )

        if self.use_perturb and self.perturbation_strengths:
            for strength in self.perturbation_strengths:
                sample[f"image_{strength}"] = self._make_perturbed_tensor(bgr, strength)
            # Backward compatibility: old train code uses image_pert.
            first_strength = self.perturbation_strengths[0]
            sample["image_pert"] = sample[f"image_{first_strength}"]
        else:
            sample["image_pert"] = self.eval_transform(pil)

        return sample


def build_agiqa3k_simsiam_dataloaders(cfg: Dict):
    dcfg = cfg["dataset"]
    lcfg = cfg["loader"]

    base_kwargs = dict(
        root_dir=dcfg["root_dir"],
        image_dir=dcfg.get("image_dir", "image"),
        csv_file=dcfg.get("csv_file", "data.csv"),
        input_size=dcfg.get("input_size", 224),
        mos_max=dcfg.get("mos_max", 5.0),
        final_quality_weight=dcfg.get("final_quality_weight", 0.5),
        final_align_weight=dcfg.get("final_align_weight", 0.5),
        use_full_prompt=dcfg.get("use_full_prompt", True),
        use_perturb=dcfg.get("use_perturb", True),
        perturbation_prob=dcfg.get("perturbation_prob", 0.7),
        perturbation_strength=dcfg.get("perturbation_strength", "mild"),
        perturbation_strengths=dcfg.get("perturbation_strengths", None),
        saliency_sampling_ratio=dcfg.get("saliency_sampling_ratio", 0.7),
        verify_images=dcfg.get("verify_images", False),
        mirror_augmentation=dcfg.get("mirror_augmentation", {}),
        train_augmentation_mode=dcfg.get("train_augmentation_mode", "legacy"),
        quality_safe_flip_prob=dcfg.get("quality_safe_flip_prob", .5),
        local_multicrop=dcfg.get("local_multicrop", {}),
    )
    train_dataset = AGIQA3KSimSiamDataset(train=True, **base_kwargs)
    eval_dataset = AGIQA3KSimSiamDataset(train=False, **base_kwargs)

    total = len(train_dataset)
    train_ratio = float(dcfg.get("train_ratio", 0.7))
    val_ratio = float(dcfg.get("val_ratio", 0.1))
    train_len = int(total * train_ratio)
    val_len = int(total * val_ratio)
    test_len = total - train_len - val_len

    split_seed = int(cfg.get("split_seed", cfg.get("seed", 42)))
    generator = torch.Generator().manual_seed(split_seed)
    all_indices = torch.randperm(total, generator=generator).tolist()
    train_indices = all_indices[:train_len]
    val_indices = all_indices[train_len:train_len + val_len]
    test_indices = all_indices[train_len + val_len:]

    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set = torch.utils.data.Subset(eval_dataset, val_indices)
    test_set = torch.utils.data.Subset(eval_dataset, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=lcfg.get("batch_size", 8),
        shuffle=True,
        num_workers=lcfg.get("num_workers", 4),
        pin_memory=lcfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=lcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=lcfg.get("num_workers", 4),
        pin_memory=lcfg.get("pin_memory", True),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=lcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=lcfg.get("num_workers", 4),
        pin_memory=lcfg.get("pin_memory", True),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
