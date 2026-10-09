# 根据 config 选择对应 dataloader
from .agiqa3k_simsiam_dataset import build_agiqa3k_simsiam_dataloaders
from .mixed_pretrain_dataset import build_mixed_pretrain_dataloaders
from .multi_iqa_dataset import build_agiqa1k_iqa_dataloaders, build_aigciqa2023_iqa_dataloaders


def build_dataloaders(cfg):
    dataset_type = cfg["dataset"].get("type", "agiqa3k_simsiam")
    if dataset_type == "agiqa3k_simsiam":
        return build_agiqa3k_simsiam_dataloaders(cfg)
    if dataset_type == "mixed_pretrain":
        return build_mixed_pretrain_dataloaders(cfg)
    if dataset_type == "agiqa1k_iqa":
        return build_agiqa1k_iqa_dataloaders(cfg)
    if dataset_type == "aigciqa2023_iqa":
        return build_aigciqa2023_iqa_dataloaders(cfg)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
