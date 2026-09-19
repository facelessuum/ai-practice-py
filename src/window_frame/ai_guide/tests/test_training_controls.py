from window_frame.ai_guide.utils.settings import Settings, DatasetConfig, ModelConfig, TrainingConfig, OutputConfig
import unittest
from window_frame.ai_guide.training.check_early_stopping import EarlyStopping
from window_frame.ai_guide.training.train_model import main
from window_frame.ai_guide.utils.prepare_batches import limit_samples
from window_frame.ai_guide.utils.save_examples import visualization_ids
from unittest.mock import patch
from window_frame.ai_guide.training.estimate_remaining_time import RemainingTime
from window_frame.ai_guide.run_history.show_progress import format_duration


class TrainingControlTests(unittest.TestCase):
    def test_visualization_selection_rotates_without_duplicates(self):
        ids = [str(i) for i in range(12)]
        first = visualization_ids(ids, 0, 42)
        second = visualization_ids(ids, 1, 42)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(second), 5)
        self.assertTrue(first.isdisjoint(second))
        self.assertEqual(first, visualization_ids(ids, 0, 42))
        self.assertEqual(visualization_ids(ids[:3], 2, 42), set(ids[:3]))
        self.assertEqual(visualization_ids([], 0, 42), set())
        self.assertEqual(ids, [str(i) for i in range(12)])

    def test_small_subsets_keep_split_boundaries(self):
        split = dict(train=list(range(100)), validation=list(range(100, 120)), test=list(range(120, 130)))
        settings = TrainingConfig(seed=42, max_train_samples=50, max_validation_samples=10)
        limited = limit_samples(split, settings)
        self.assertEqual(len(limited['train']), 50)
        self.assertEqual(len(limited['validation']), 10)
        self.assertEqual(limited['test'], split['test'])
        self.assertEqual(len(split['train']), 100)
        self.assertEqual(limited, limit_samples(split, settings))
        self.assertTrue(set(limited['train']).isdisjoint(limited['validation']))

    def test_command_line_limits(self):
        with patch('window_frame.ai_guide.training.train_model.settings', Settings()), \
             patch('sys.argv', ['train-window-frame', '--epochs', '5', '--max-train-samples', '50',
                               '--max-validation-samples', '10']), \
             patch('window_frame.ai_guide.training.train_model.run_training') as run:
            main()
        settings = run.call_args.args[0].training
        self.assertEqual(settings.epochs, 5)
        self.assertEqual(settings.max_train_samples, 50)
        self.assertEqual(settings.max_validation_samples, 10)

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
