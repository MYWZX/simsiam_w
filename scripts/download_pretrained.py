# 权重下载
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-download timm/Hugging Face pretrained weights used by the project.')
    parser.add_argument('--global-model', type=str, default='vit_base_patch16_224.augreg2_in21k_ft_in1k')
    parser.add_argument('--local-model', type=str, default='resnet50.a1_in1k')
    parser.add_argument('--cache-dir', type=str, default=None, help='Optional Hugging Face cache dir.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(cache_dir.resolve())
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir.resolve())

    try:
        import timm
    except ImportError as e:
        raise ImportError('Please install timm first: pip install timm huggingface_hub safetensors') from e

    print(f'Downloading global model: {args.global_model}')
    timm.create_model(
        args.global_model,
        pretrained=True,
        num_classes=0,
        global_pool='avg',
    )

    print(f'Downloading local model: {args.local_model}')
    timm.create_model(
        args.local_model,
        pretrained=True,
        features_only=True,
        out_indices=(4,),
    )

    print('Done. The pretrained weights are now cached locally.')


if __name__ == '__main__':
    main()
