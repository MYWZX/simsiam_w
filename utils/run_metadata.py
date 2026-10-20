"""Non-secret runtime provenance; records settings without changing determinism."""
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch


def checkpoint_fingerprint(path):
    if not path:
        return None
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Input checkpoint changed while hashing.')
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def collect_run_metadata(checkpoint, cfg):
    repo = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(repo), *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    packages = {}
    for name in ('torchvision', 'open-clip-torch', 'timm', 'numpy', 'scipy', 'pandas'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    status = git('status', '--porcelain')
    return dict(
        checkpoint=checkpoint, checkpoint_verified_after_load=False,
        code_commit=git('rev-parse', 'HEAD'), workspace_dirty=None if status is None else bool(status),
        packages=packages, python=sys.version, platform=platform.platform(), pytorch=str(torch.__version__),
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        gpus=[dict(index=i, name=torch.cuda.get_device_name(i),
                   total_memory_bytes=torch.cuda.get_device_properties(i).total_memory)
              for i in range(torch.cuda.device_count())],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        environment={key: os.environ.get(key) for key in (
            'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        seed=cfg.get('seed', 42), split_seed=cfg.get('split_seed', cfg.get('seed', 42)),
        comparison=cfg.get('train', {}).get('compare_selection_policies', False),
        note='Settings recorded, not changed; fixed seeds do not guarantee deterministic CUDA training.')


def write_run_metadata(outdir, metadata):
    path = Path(outdir) / 'run_metadata.json'
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return str(path)
