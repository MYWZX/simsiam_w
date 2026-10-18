"""Checkpoint publication and legacy pretraining bookkeeping (no ML imports)."""
import csv
import json
import os
import tempfile


def atomic_save(checkpoint, path, save):
    """Publish only after the writer has completed and the file is flushed."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix='.' + os.path.basename(path), suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def legacy_resume_records(csv_path, run_id, epoch, output_dir, specs):
    """Recover main state at resume epoch; preserve existing later auxiliary bests."""
    main = None
    auxiliary = {}
    if not os.path.isfile(csv_path):
        return main, auxiliary
    with open(csv_path, encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('event') != 'epoch' or row.get('run_id') != run_id:
                continue
            metrics = json.loads(row['metrics_json'])
            if int(row['epoch']) == epoch:
                main = dict(best=metrics['best_selection_value'],
                            best_epoch=int(row['best_epoch']),
                            no_improve=int(metrics.get('no_improve', 0)))
            for name, (_, filename) in specs.items():
                value = metrics.get(f'best_{name}_selection_value')
                if value is not None and os.path.isfile(os.path.join(output_dir, filename)):
                    auxiliary[name] = dict(value=value, epoch=int(metrics[f'best_{name}_epoch']))
    return main, auxiliary
