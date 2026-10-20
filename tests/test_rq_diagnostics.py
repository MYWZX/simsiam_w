"""Offline research-control checks with tiny/mock models, no dataset/download."""
import copy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

import train_cm_iqa as iqa
import train_cm_pretrain as pretrain
from models.two_stage_models import load_pretrained_backbone
from tools import run_rq_diagnostics as runner
from utils.train_utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def pretrain_args(*flags):
    with patch("sys.argv", ["train_cm_pretrain.py", "--config", "unused.yaml", *flags]):
        return pretrain.parse_args()


def iqa_args(*flags):
    with patch("sys.argv", ["train_cm_iqa.py", "--config", "unused.yaml", *flags]):
        return iqa.parse_args()


class TinyTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_quality_adapter = True
        self.backbone = nn.Module()
        self.backbone.global_encoder = nn.Linear(2, 2)
        self.backbone.local_encoder = nn.Linear(2, 2)
        self.backbone.fusion = nn.Linear(2, 2)
        self.text_to_fused = nn.Linear(2, 2)
        self.align_global_adapter = nn.Linear(2, 2)
        self.align_text_adapter = nn.Linear(2, 2)
        self.align_patch_adapter = nn.Linear(2, 2)
        self.quality_adapter = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.global_quality_head = nn.Linear(2, 1)


class ResearchControlTests(unittest.TestCase):
    def base(self):
        return load_yaml(str(ROOT / "configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml"))

    def test_default_pretrain_is_unchanged(self):
        cfg = self.base()
        expected = copy.deepcopy(cfg)
        pretrain.configure_research_controls(cfg, pretrain_args())
        self.assertEqual(cfg, expected)

    def test_ablation_changes_exactly_four_coefficients(self):
        cfg = self.base()
        expected = copy.deepcopy(cfg)
        for key in pretrain.REGULAR_SIMSIAM_WEIGHT_KEYS:
            expected["train"][key] = 0.0
        pretrain.configure_research_controls(cfg, pretrain_args("--disable-regular-simsiam"))
        self.assertEqual(cfg, expected)
        self.assertGreater(cfg["train"]["lambda_img_prompt_local"], 0)
        self.assertGreater(cfg["train"]["lambda_prompt_answer_word"], 0)
        self.assertTrue(cfg["train"]["quality_component_anchor_ordinal"]["enabled"])

    def test_fixed_budget_preserves_scheduler_and_patience(self):
        cfg = self.base()
        expected = copy.deepcopy(cfg)
        expected["train"]["fixed_training_epochs"] = 17
        pretrain.configure_research_controls(cfg, pretrain_args("--fixed-epochs", "17"))
        self.assertEqual(cfg, expected)

    def test_ablation_also_clears_effective_image_image_alias(self):
        cfg = self.base()
        cfg["train"]["lambda_quality_view_siam"] = 0.3
        expected = copy.deepcopy(cfg)
        for key in (*pretrain.REGULAR_SIMSIAM_WEIGHT_KEYS, "lambda_quality_view_siam"):
            expected["train"][key] = 0.0
        pretrain.configure_research_controls(cfg, pretrain_args("--disable-regular-simsiam"))
        self.assertEqual(cfg, expected)
        self.assertEqual(cfg["train"].get("lambda_quality_view_siam", cfg["train"]["lambda_img_siam"]), 0.0)

    def test_invalid_budget_has_no_partial_mutation(self):
        for value in ("0", "-1", "51"):
            cfg = self.base()
            expected = copy.deepcopy(cfg)
            with self.subTest(value=value), self.assertRaises(ValueError):
                pretrain.configure_research_controls(
                    cfg, pretrain_args("--fixed-epochs", value, "--disable-regular-simsiam"))
            self.assertEqual(cfg, expected)

    def test_fixed_budget_ignores_patience_and_logs_completion(self):
        cfg = {
            "seed": 42, "train": {"epochs": 5, "early_stop_patience": 1, "amp": False},
            "model": {},
        }
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = MagicMock()
        with (
            patch("sys.argv", ["train_cm_pretrain.py", "--config", "unused.yaml", "--fixed-epochs", "3"]),
            patch.object(pretrain, "load_yaml", return_value=cfg),
            patch.object(pretrain, "set_seed"),
            patch.object(pretrain, "ensure_dir"),
            patch.object(pretrain, "dump_config_snapshot", return_value="unused-snapshot.yaml"),
            patch.object(pretrain, "CSVExperimentLogger") as logger,
            patch.object(pretrain, "build_dataloaders", return_value=(None, None, None)),
            patch.object(pretrain, "CrossModalSimSiamPretrain", return_value=model),
            patch.object(pretrain, "build_optimizer", return_value=optimizer),
            patch.object(pretrain, "CosineAnnealingLR", return_value=scheduler) as build_scheduler,
            patch.object(pretrain, "train_one_epoch", return_value={"loss": 1.0}) as train,
            patch.object(pretrain, "evaluate", return_value={"loss": 1.0}),
            patch.object(pretrain.torch.cuda, "is_available", return_value=False),
            patch.object(pretrain, "atomic_save"),
        ):
            pretrain.main()
        self.assertEqual(train.call_count, 3)
        self.assertEqual(build_scheduler.call_args.kwargs["T_max"], 5)
        events = [call.kwargs["event"] for call in logger.return_value.log.call_args_list]
        self.assertNotIn("early_stop", events)
        final = logger.return_value.log.call_args_list[-1].kwargs
        self.assertEqual(final["event"], "final")
        self.assertEqual(final["metrics"]["last_epoch"], 3)
        self.assertTrue(final["metrics"]["fixed_training_budget"])

    def test_default_still_early_stops(self):
        cfg = {"seed": 42, "train": {"epochs": 5, "early_stop_patience": 1, "amp": False}, "model": {}}
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with (
            patch("sys.argv", ["train_cm_pretrain.py", "--config", "unused.yaml"]),
            patch.object(pretrain, "load_yaml", return_value=cfg),
            patch.object(pretrain, "set_seed"), patch.object(pretrain, "ensure_dir"),
            patch.object(pretrain, "dump_config_snapshot", return_value="unused-snapshot.yaml"),
            patch.object(pretrain, "CSVExperimentLogger") as logger,
            patch.object(pretrain, "build_dataloaders", return_value=(None, None, None)),
            patch.object(pretrain, "CrossModalSimSiamPretrain", return_value=model),
            patch.object(pretrain, "build_optimizer", return_value=optimizer),
            patch.object(pretrain, "CosineAnnealingLR", return_value=MagicMock()),
            patch.object(pretrain, "train_one_epoch", return_value={"loss": 1.0}) as train,
            patch.object(pretrain, "evaluate", return_value={"loss": 1.0}),
            patch.object(pretrain.torch.cuda, "is_available", return_value=False),
            patch.object(pretrain, "atomic_save"),
        ):
            pretrain.main()
        self.assertEqual(train.call_count, 2)
        self.assertIn("early_stop", [c.kwargs["event"] for c in logger.return_value.log.call_args_list])

    def test_adapter_overrides_are_isolated_in_all_datasets(self):
        for dataset in ("agiqa1k", "agiqa3k", "aigciqa2023"):
            for mode in ("replace", "parallel_score"):
                for init in ("random", "pretrained"):
                    cfg = load_yaml(str(ROOT / f"configs/{dataset}_cm_iqa_decoupled_quality_align.yaml"))
                    expected = copy.deepcopy(cfg)
                    expected["model"]["quality_adapter"]["mode"] = mode
                    expected["train"]["skip_pretrained_quality_adapter"] = init == "random"
                    iqa.configure_quality_adapter_experiment(
                        cfg, iqa_args("--quality-adapter-mode", mode, "--quality-adapter-init", init))
                    self.assertEqual(cfg, expected)

    def test_adapter_default_is_unchanged_and_disabled_rejected(self):
        cfg = {"model": {"quality_adapter": {"enabled": True}}, "train": {}}
        expected = copy.deepcopy(cfg)
        iqa.configure_quality_adapter_experiment(cfg, iqa_args())
        self.assertEqual(cfg, expected)
        with self.assertRaises(ValueError):
            iqa.configure_quality_adapter_experiment(
                cfg, iqa_args("--disable-quality-adapter", "--quality-adapter-init", "random"))
        self.assertEqual(cfg, expected)

    def test_iqa_main_passes_adapter_initialization_to_loader_and_snapshot(self):
        for initialization in ("pretrained", "random"):
            cfg = load_yaml(str(ROOT / "configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml"))
            with (
                self.subTest(initialization=initialization),
                patch("sys.argv", ["train_cm_iqa.py", "--config", "unused.yaml",
                                   "--seed", "42", "--quality-adapter-init", initialization,
                                   "--pretrained-checkpoint", "mock.pth"]),
                patch.object(iqa, "load_yaml", return_value=cfg),
                patch.object(iqa, "ensure_dir"),
                patch.object(iqa, "dump_config_snapshot", return_value="snapshot.yaml") as snapshot,
                patch.object(iqa, "CSVExperimentLogger"),
                patch.object(iqa, "build_dataloaders", return_value=(None, None, None)),
                patch.object(iqa, "build_iqa_model", return_value=nn.Linear(2, 2)),
                patch.object(iqa.os.path, "isfile", return_value=True),
                patch.object(iqa.torch.cuda, "is_available", return_value=False),
                patch.object(iqa, "load_pretrained_backbone",
                             side_effect=RuntimeError("test stops after checkpoint loading")) as loader,
            ):
                with self.assertRaisesRegex(RuntimeError, "test stops after checkpoint loading"):
                    iqa.main()
            random = initialization == "random"
            self.assertEqual(loader.call_args.kwargs["skip_quality_adapter"], random)
            self.assertEqual(snapshot.call_args.args[0]["train"]["skip_pretrained_quality_adapter"], random)
            self.assertEqual(cfg["split_seed"], 42)
            self.assertTrue(cfg["model"]["quality_adapter"]["enabled"])


class TransferTests(unittest.TestCase):
    def state(self, model, adapter_source):
        state = {}
        for key, value in model.state_dict().items():
            if key.startswith("quality_adapter."):
                suffix = key.removeprefix("quality_adapter.")
                state[adapter_source + suffix] = torch.full_like(value, 7)
            else:
                state["iqa." + key] = torch.full_like(value, 3)
        return state

    def test_random_skips_all_adapter_sources_and_bn_buffers(self):
        for source in ("quality_image_projector.", "quality_adapter.", "iqa.quality_adapter."):
            model = TinyTransfer()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            state = {"module." + k: v for k, v in self.state(model, source).items()}
            with patch("models.two_stage_models.torch.load", return_value={"model": state}):
                load_pretrained_backbone(model, "mock.pth", load_scope="representation", skip_quality_adapter=True)
            for key, value in model.state_dict().items():
                if key.startswith(("quality_adapter.", "global_quality_head.")):
                    self.assertTrue(torch.equal(value, before[key]), key)
                else:
                    self.assertTrue(torch.equal(value, torch.full_like(value, 3)), key)
            self.assertTrue(all(p.requires_grad for p in model.quality_adapter.parameters()))

    def test_default_loads_projector_including_bn_buffers(self):
        model = TinyTransfer()
        with patch("models.two_stage_models.torch.load",
                   return_value={"model": self.state(model, "quality_image_projector.")}):
            load_pretrained_backbone(model, "mock.pth", load_scope="representation")
        for key, value in model.state_dict().items():
            if key.startswith("quality_adapter."):
                self.assertTrue(torch.equal(value, torch.full_like(value, 7)), key)

    def test_missing_adapter_is_an_error_except_explicit_random(self):
        model = TinyTransfer()
        state = {"iqa.backbone.fusion.weight": torch.ones(2, 2)}
        with patch("models.two_stage_models.torch.load", return_value={"model": state}):
            with self.assertRaisesRegex(RuntimeError, "did not provide"):
                load_pretrained_backbone(model, "mock.pth", load_scope="representation")
            load_pretrained_backbone(model, "mock.pth", load_scope="representation", skip_quality_adapter=True)


class RunnerTests(unittest.TestCase):
    def test_pretrain_plan_differs_only_in_one_flag_and_output_name(self):
        jobs = runner.build_plan(runner.parse_args(["pretrain"]))
        self.assertEqual(len(jobs), 2)
        normalized = []
        for job in jobs:
            command = list(job.command)
            command[command.index("--output-dir") + 1] = "OUT"
            if "--disable-regular-simsiam" in command:
                command.remove("--disable-regular-simsiam")
            normalized.append(command)
        self.assertEqual(*normalized)

    def test_downstream_checkpoint_seed_is_independent(self):
        for dataset in ("1k", "3k", "2023"):
            jobs = runner.build_plan(runner.parse_args(["downstream", "--dataset", dataset, "--seed", "123"]))
            self.assertEqual(len(jobs), 2)
            self.assertTrue(all("seed42" in str(j.checkpoint) for j in jobs))
            self.assertTrue(all(j.command[j.command.index("--seed") + 1] == "123" for j in jobs))
            self.assertTrue(all(
                Path(j.command[j.command.index("--config") + 1]).name
                == f"{runner.DATASETS[dataset]}_cm_iqa_decoupled_quality_align.yaml"
                for j in jobs
            ))

    def test_adapter_plan_is_two_by_two_using_one_checkpoint(self):
        jobs = runner.build_plan(runner.parse_args(["adapter"]))
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({j.checkpoint for j in jobs}), 1)
        self.assertEqual(len({j.output_dir for j in jobs}), 4)
        conditions = {
            (j.command[j.command.index("--quality-adapter-mode") + 1],
             j.command[j.command.index("--quality-adapter-init") + 1])
            for j in jobs
        }
        self.assertEqual(conditions, {
            (mode, initialization) for mode in ("replace", "parallel_score")
            for initialization in ("pretrained", "random")
        })

    def test_excessive_budget_is_rejected_before_output_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not_created"
            jobs = runner.build_plan(runner.parse_args([
                "pretrain", "--fixed-epochs", "51", "--output-root", str(root)
            ]))
            with patch.object(runner.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "Fixed training epochs"):
                    runner.execute_plan(jobs)
            self.assertFalse(root.exists())
            run.assert_not_called()

    def test_dry_run_does_not_write_or_execute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not_created"
            with patch.object(runner.subprocess, "run") as run, redirect_stdout(StringIO()):
                runner.main(["pretrain", "--output-root", str(root)])
            self.assertFalse(root.exists())
            run.assert_not_called()

    def test_missing_checkpoint_stops_entire_batch_before_any_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not_created"
            jobs = runner.build_plan(runner.parse_args(["downstream", "--output-root", str(root)]))
            with patch.object(runner.subprocess, "run") as run:
                with self.assertRaises(FileNotFoundError):
                    runner.execute_plan(jobs)
            self.assertFalse(root.exists())
            run.assert_not_called()

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = runner.build_plan(runner.parse_args(["pretrain", "--output-root", temporary]))
            jobs[1].output_dir.mkdir()
            with patch.object(runner.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    runner.execute_plan(jobs)
            self.assertFalse(jobs[0].output_dir.exists())
            run.assert_not_called()

    def test_failure_stops_following_jobs_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = runner.build_plan(runner.parse_args(["pretrain", "--output-root", temporary]))
            import subprocess
            with patch.object(runner.subprocess, "run",
                              side_effect=subprocess.CalledProcessError(1, jobs[0].command)) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.execute_plan(jobs)
            self.assertEqual(run.call_count, 1)
            self.assertFalse(jobs[1].output_dir.exists())


if __name__ == "__main__":
    unittest.main()
