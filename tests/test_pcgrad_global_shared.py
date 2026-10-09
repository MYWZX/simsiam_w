"""CPU regression tests; no dataset, network, or pretrained weights needed."""
import math
import unittest

import torch
from torch import nn
from torch.amp import GradScaler

from utils.pcgrad import global_shared_pcgrad_backward


class GlobalSharedPCGradTests(unittest.TestCase):
    def run_gradients(self, qvec, avec, *, partitioned=False, amp=False, strength=1.0):
        model = nn.Module()
        qvec, avec = torch.tensor(qvec), torch.tensor(avec)
        if partitioned:
            model.shared = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in qvec])
            vector = torch.stack(list(model.shared))
        else:
            model.shared = nn.Parameter(torch.zeros_like(qvec))
            vector = model.shared
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = GradScaler("cpu", enabled=amp, init_scale=128.0)
        stats = global_shared_pcgrad_backward(
            (vector * qvec).sum(), (vector * avec).sum(), model, optimizer, scaler,
            projection_strength=strength,
        )
        if amp:
            scaler.unscale_(optimizer)
        grads = (
            torch.stack([p.grad for p in model.shared])
            if partitioned else model.shared.grad
        ).clone()
        optimizer.zero_grad(set_to_none=True)
        return grads, stats

    def test_global_agreement_not_projected_despite_local_conflict(self):
        grad, stats = self.run_gradients([2.0, -1.0], [2.0, 2.0], partitioned=True)
        torch.testing.assert_close(grad, torch.tensor([4.0, 1.0]))
        self.assertEqual(stats["conflict"], 0.0)
        self.assertGreater(stats["cosine"], 0.0)

    def test_global_conflict(self):
        grad, stats = self.run_gradients([1.0, 0.0], [-1.0, 1.0])
        torch.testing.assert_close(grad, torch.tensor([0.5, 1.5]))
        self.assertEqual(stats["conflict"], 1.0)
        self.assertAlmostEqual(stats["cosine"], -1.0 / math.sqrt(2), places=6)
        self.assertAlmostEqual(stats["update_norm_before"], 1.0)
        self.assertAlmostEqual(stats["update_norm_after"], math.sqrt(2.5))

    def test_tensor_partition_does_not_change_result(self):
        for q, a in [([2.0, -1.0], [2.0, 2.0]), ([1.0, 0.0], [-1.0, 1.0])]:
            for strength in [0.0, 0.5, 1.0]:
                flat, _ = self.run_gradients(q, a, strength=strength)
                partitioned, _ = self.run_gradients(q, a, partitioned=True, strength=strength)
                torch.testing.assert_close(flat, partitioned)

    def test_half_strength_is_half_the_correction(self):
        grad, stats = self.run_gradients([1.0, 0.0], [-1.0, 1.0], strength=0.5)
        torch.testing.assert_close(grad, torch.tensor([0.25, 1.25]))
        # Conflict is detected BEFORE projection, independent of its strength.
        self.assertEqual(stats['conflict'], 1.0)
        self.assertAlmostEqual(stats['update_norm_after'], math.sqrt(1.625))

    def test_strength_interpolates_between_sum_and_full_projection(self):
        q, a = [1.0, 2.0, -1.0], [-2.0, -1.0, 0.5]
        summed, _ = self.run_gradients(q, a, strength=0.0)
        full, full_stats = self.run_gradients(q, a)
        for strength in [0.25, 0.5, 0.75, 1.0]:
            actual, stats = self.run_gradients(q, a, strength=strength)
            torch.testing.assert_close(actual, (1.0-strength)*summed + strength*full)
            self.assertEqual(stats['conflict'], full_stats['conflict'])
            self.assertAlmostEqual(stats['cosine'], full_stats['cosine'])

    def test_nonconflicting_gradients_do_not_depend_on_strength(self):
        for strength in [0.0, 0.5, 1.0]:
            grad, _ = self.run_gradients([2.0, -1.0], [2.0, 2.0], strength=strength)
            torch.testing.assert_close(grad, torch.tensor([4.0, 1.0]))

    def test_invalid_strength_rejected(self):
        for strength in [-0.01, 1.01, float('nan'), float('inf'), -float('inf')]:
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                self.run_gradients([1.0, 0.0], [-1.0, 1.0], strength=strength)

    def test_zero_strength_uses_one_backward_and_matches_adamw(self):
        for amp in [False, True]:
            model = nn.Linear(2, 1)
            reference = nn.Linear(2, 1)
            reference.load_state_dict(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
            scaler = GradScaler('cpu', enabled=amp, init_scale=128.0)
            ref_scaler = GradScaler('cpu', enabled=amp, init_scale=128.0)
            calls = []
            hook = model.weight.register_hook(lambda grad: calls.append(1))
            x = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
            for step in range(2):
                y, ref_y = model(x), reference(x)
                stats = global_shared_pcgrad_backward(
                    y.square().sum(), -(y+0.2).square().sum(),
                    model, optimizer, scaler, projection_strength=0.0,
                )
                ref_optimizer.zero_grad(set_to_none=True)
                ref_scaler.scale(ref_y.square().sum() - (ref_y+0.2).square().sum()).backward()
                self.assertEqual(len(calls), step+1)
                for p, ref in zip(model.parameters(), reference.parameters()):
                    self.assertTrue(torch.equal(p.grad, ref.grad))
                scaler.unscale_(optimizer)
                ref_scaler.unscale_(ref_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
                scaler.step(optimizer)
                ref_scaler.step(ref_optimizer)
                scaler.update()
                ref_scaler.update()
                for p, ref in zip(model.parameters(), reference.parameters()):
                    self.assertTrue(torch.equal(p, ref))
                self.assertTrue(all(value == 0.0 for value in stats.values()))
            hook.remove()

    def test_private_and_unused_parameters(self):
        model = nn.Module()
        model.shared = nn.Parameter(torch.zeros(2))
        model.quality = nn.Parameter(torch.zeros(()))
        model.align = nn.Parameter(torch.zeros(()))
        model.unused = nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = global_shared_pcgrad_backward(
            model.shared[0] + 3 * model.quality,
            -model.shared[0] + model.shared[1] + 5 * model.align,
            model, optimizer, GradScaler("cpu", enabled=False),
        )
        torch.testing.assert_close(model.shared.grad, torch.tensor([0.5, 1.5]))
        self.assertEqual(model.quality.grad.item(), 3.0)
        self.assertEqual(model.align.grad.item(), 5.0)
        self.assertIsNone(model.unused.grad)
        self.assertEqual(stats["shared_numel"], 2)

    def test_single_task_matches_ordinary_backward(self):
        model = nn.Linear(2, 1)
        reference = nn.Linear(2, 1)
        reference.load_state_dict(model.state_dict())
        x = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = global_shared_pcgrad_backward(
            model(x).square().sum(), torch.tensor(0.0), model, optimizer,
            GradScaler("cpu", enabled=False), align_active=False,
        )
        reference(x).square().sum().backward()
        for p, ref in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(p.grad, ref.grad))
        self.assertEqual(stats["active"], 0.0)

    def test_amp_projection_and_diagnostics_match_unscaled(self):
        for strength in [0.0, 0.5, 1.0]:
            expected, baseline = self.run_gradients([1.0, 0.0], [-1.0, 1.0], strength=strength)
            actual, stats = self.run_gradients([1.0, 0.0], [-1.0, 1.0], amp=True, strength=strength)
            torch.testing.assert_close(actual, expected)
            for key in baseline:
                self.assertAlmostEqual(stats[key], baseline[key])

    def test_half_strength_preserves_private_gradients(self):
        model = nn.Module()
        model.shared = nn.Parameter(torch.zeros(2))
        model.quality = nn.Parameter(torch.zeros(()))
        model.align = nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        global_shared_pcgrad_backward(
            model.shared[0] + 3*model.quality,
            -model.shared[0] + model.shared[1] + 5*model.align,
            model, optimizer, GradScaler('cpu', enabled=False), projection_strength=0.5,
        )
        torch.testing.assert_close(model.shared.grad, torch.tensor([0.25, 1.25]))
        self.assertEqual(model.quality.grad.item(), 3.0)
        self.assertEqual(model.align.grad.item(), 5.0)

    def test_half_strength_single_task_matches_ordinary_backward(self):
        model = nn.Linear(2, 1)
        reference = nn.Linear(2, 1)
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        stats = global_shared_pcgrad_backward(
            model(x).square().sum(), torch.tensor(0.0), model, optimizer,
            GradScaler('cpu', enabled=False), align_active=False, projection_strength=0.5,
        )
        reference(x).square().sum().backward()
        for p, ref in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(p.grad, ref.grad))
        self.assertEqual(stats['active'], 0.0)

    def test_no_shared_parameters(self):
        model = nn.Module()
        model.quality = nn.Parameter(torch.zeros(()))
        model.align = nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = global_shared_pcgrad_backward(
            3 * model.quality, 5 * model.align, model, optimizer,
            GradScaler("cpu", enabled=False),
        )
        self.assertEqual(model.quality.grad.item(), 3.0)
        self.assertEqual(model.align.grad.item(), 5.0)
        self.assertEqual(stats["active"], 0.0)

    def test_exact_opposition_is_finite(self):
        grad, stats = self.run_gradients([1.0, 2.0], [-1.0, -2.0])
        torch.testing.assert_close(grad, torch.zeros(2))
        self.assertTrue(all(math.isfinite(v) for v in stats.values()))

    def test_zero_task_gradient_is_not_counted_as_active(self):
        grad, stats = self.run_gradients([0.0, 0.0], [1.0, 2.0])
        torch.testing.assert_close(grad, torch.tensor([1.0, 2.0]))
        self.assertEqual(stats["active"], 0.0)

    def test_nonfinite_gradient_reaches_scaler_and_skips_step(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = GradScaler("cpu", enabled=True, init_scale=128.0)
        stats = global_shared_pcgrad_backward(
            model.weight.sum() * float("inf"), model.weight.sum(),
            model, optimizer, scaler,
        )
        self.assertEqual(stats["nonfinite"], 1.0)
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
