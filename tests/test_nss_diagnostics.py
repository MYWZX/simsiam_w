import io,json,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from scripts import pack_nss_diagnostics as diagnostics

class NSSDiagnosticsTests(unittest.TestCase):
    def test_incomplete_archives_include_logs_not_weights_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for v in diagnostics.VARIANTS:
                p=root/f'configs/nss_ablation/{v}/pretrain.yaml';p.parent.mkdir(parents=True)
                p.write_text(yaml.safe_dump({'output_dir':f'outputs/{v}','results_csv':f'outputs/{v}.csv'}))
                d=root/f'outputs/{v}';d.mkdir(parents=True)
                (d/'console.log').write_text('original failure\nTraceback: error\n')
                (d/'best_component_anchor_pretrain.pth').write_bytes(b'never-include')
                (root/f'outputs/{v}.csv').write_text('event,epoch\nepoch,1\n')
                archived=root/f'outputs/{v}.failed_earlier';archived.mkdir();(archived/'console.log').write_text('earlier error')
            with patch.object(diagnostics.subprocess,'run') as run:
                run.return_value.returncode=1
                archive,report=diagnostics.pack(root,diagnostics.VARIANTS)
            with tarfile.open(archive) as tar:
                self.assertIn('outputs/stats6/console.log',tar.getnames())
                self.assertIn('outputs/ggd.failed_earlier/console.log',tar.getnames())
                self.assertFalse(any(n.endswith(('.pth','.pt','.bin')) for n in tar.getnames()))
                data=json.load(tar.extractfile('diagnosis_summary.json'))
                self.assertFalse(data['variants']['stats6']['completion_manifest_exists'])
            self.assertEqual((root/'outputs/stats6/console.log').read_text(),'original failure\nTraceback: error\n')
            self.assertTrue((root/'outputs/ggd/best_component_anchor_pretrain.pth').exists())
            with patch.object(diagnostics,'ROOT',root),patch('sys.argv',['pack_nss_diagnostics.py']),patch('sys.stdout',io.StringIO()):diagnostics.main()

    def test_no_output_directory_and_bounded_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'configs/nss_ablation/stats6/pretrain.yaml';p.parent.mkdir(parents=True)
            p.write_text(yaml.safe_dump({'output_dir':'outputs/new','results_csv':'outputs/new.csv'}))
            payload,report=diagnostics.collect(root,['stats6'])
            self.assertFalse(report['variants']['stats6']['directory_exists'])
            self.assertFalse((root/'outputs').exists())
            d=root/'outputs/new';d.mkdir(parents=True);(d/'console.log').write_text('X'*100+'LAST')
            with patch.object(diagnostics,'MAX_FILE_BYTES',16):payload,report=diagnostics.collect(root,['stats6'])
            self.assertEqual(payload['outputs/new/console.log.tail.txt'],b'X'*12+b'LAST')
            self.assertTrue(report['files']['outputs/new/console.log.tail.txt']['tail_only'])

if __name__=='__main__':unittest.main()
