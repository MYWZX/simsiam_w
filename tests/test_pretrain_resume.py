import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from utils.pretrain_resume import atomic_save, legacy_resume_records


class ResumeTests(unittest.TestCase):
    def test_atomic_publication_and_failed_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pth'
            atomic_save({'epoch': 31}, path, pickle.dump)
            def interrupted_save(obj, stream):
                stream.write(b'incomplete')
                raise RuntimeError('simulated interrupted writer')
            with self.assertRaises(RuntimeError):
                atomic_save({'epoch': 32}, path, interrupted_save)
            with path.open('rb') as stream:
                self.assertEqual(pickle.load(stream)['epoch'], 31)
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])
            atomic_save({'epoch': 32}, path, pickle.dump)
            with path.open('rb') as stream:
                self.assertEqual(pickle.load(stream)['epoch'], 32)

    def test_legacy_records_main_at_resume_epoch_auxiliary_at_latest_saved_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.csv'
            (Path(tmp) / 'best_align.pth').touch()
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['event', 'run_id', 'epoch', 'best_epoch', 'metrics_json'])
                writer.writeheader()
                for epoch, value in [(31, .7), (35, .8)]:
                    writer.writerow(dict(event='epoch', run_id='run', epoch=epoch, best_epoch=31,
                                         metrics_json=json.dumps(dict(best_selection_value=.9, no_improve=epoch-31,
                                         best_align_selection_value=value, best_align_epoch=epoch))))
            main, auxiliary = legacy_resume_records(path, 'run', 31, tmp, {'align': ('metric', 'best_align.pth')})
            self.assertEqual(main, dict(best=.9, best_epoch=31, no_improve=0))
            self.assertEqual(auxiliary['align'], dict(value=.8, epoch=35))


if __name__ == '__main__':
    unittest.main()
