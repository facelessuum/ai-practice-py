import unittest
from window_frame.ai_guide.training.check_early_stopping import EarlyStopping
from window_frame.ai_guide.training.estimate_remaining_time import RemainingTime
from window_frame.ai_guide.run_history.show_progress import format_duration


class TrainingControlTests(unittest.TestCase):
    def test_warmup_plateau_and_improvement(self):
        stopping = EarlyStopping(patience=2, min_delta=.1, start_epoch=2)
        self.assertFalse(stopping.update(0, 1))
        self.assertFalse(stopping.update(1, 1))
        self.assertIsNone(stopping.best)
        self.assertFalse(stopping.update(2, 3))
        self.assertFalse(stopping.update(3, 2.95))
        self.assertEqual(stopping.bad_epochs, 1)
        self.assertFalse(stopping.update(4, 2.8))
        self.assertEqual(stopping.bad_epochs, 0)
        self.assertFalse(stopping.update(5, 2.8))
        self.assertTrue(stopping.update(6, 2.8))

    def test_disabled_and_resume(self):
        disabled = EarlyStopping(patience=0)
        for epoch in range(10):
            self.assertFalse(disabled.update(epoch, 2))
        original = EarlyStopping(patience=2)
        original.update(0, 1)
        original.update(1, 2)
        restored = EarlyStopping(patience=2)
        self.assertTrue(restored.restore(original.state_dict()))
        self.assertTrue(restored.update(2, 2))
        self.assertFalse(EarlyStopping(patience=3).restore(original.state_dict()))
        self.assertFalse(EarlyStopping().restore(None))

    def test_invalid_settings(self):
        for patience, delta in [(-1, 0), (2, -1), (2, float('nan'))]:
            with self.assertRaises(ValueError):
                EarlyStopping(patience, delta)

    def test_planned_run_estimate_and_resume(self):
        timer = RemainingTime(epochs=4, batches_per_epoch=10)
        self.assertIsNone(timer.estimate(0, 0, 0))
        self.assertEqual(timer.estimate(0, elapsed=20, completed_batches=2), 380)
        self.assertEqual(timer.finish_epoch(120), 120)
        self.assertEqual(timer.estimate(1, elapsed=30, completed_batches=3), 330)
        # A resumed run uses the absolute epoch number, not a new four-epoch total.
        self.assertEqual(timer.estimate(3, elapsed=30, completed_batches=3), 90)
        self.assertEqual(format_duration(90061), '1d 1h 1m 1s')
