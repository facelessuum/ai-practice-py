from window_frame.ai_guide.utils.settings import Settings, DatasetConfig, ModelConfig, TrainingConfig, OutputConfig
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from window_frame.ai_guide.run_history.show_progress import Progress
from window_frame.ai_guide.training import train_model


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_terminal_bar_and_clean_saved_log(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'run.log'
            logger = logging.Logger('progress-test')
            handler = logging.FileHandler(path)
            logger.addHandler(handler)
            terminal = Terminal()
            try:
                with Progress('Training', logger, 2, stream=terminal) as progress:
                    progress.update(1, loss=2)
                    progress.update(2, loss=1)
            finally:
                handler.close()
            self.assertIn('100.0%', terminal.getvalue())
            self.assertIn('loss 1.0000', path.read_text())
            self.assertNotIn('\r', path.read_bytes().decode())

    def test_redirected_output_and_interruption(self):
        stream = io.StringIO()
        logger = logging.Logger('redirect-test')
        logger.addHandler(logging.StreamHandler(stream))
        with self.assertRaises(ValueError):
            with Progress('Validation', logger, 3, stream=stream) as progress:
                progress.update(1)
                raise ValueError('test')
        self.assertIn('stopped', stream.getvalue())
        self.assertNotIn('\r', stream.getvalue())

    def test_missing_confirmation_fails_before_audit(self):
        config = Settings()
        config.dataset.class_meanings_confirmed = False
        with patch.object(train_model, 'audit_dataset') as audit:
            with self.assertRaisesRegex(ValueError, '--confirm-labels'):
                train_model.run_training(config)
            audit.assert_not_called()

    def test_default_training_never_prompts(self):
        with patch('sys.argv', ['train-window-frame']), \
             patch('builtins.input', side_effect=AssertionError('Training must not prompt')), \
             patch.object(train_model, 'run_training') as run:
            train_model.main()
        self.assertTrue(run.call_args.args[0].dataset.class_meanings_confirmed)

    def test_confirmation_flag_records_choice(self):
        with patch('sys.argv', ['train-window-frame', '--confirm-labels']), \
             patch.object(train_model, 'run_training') as run:
            train_model.main()
        self.assertTrue(run.call_args.args[0].dataset.class_meanings_confirmed)
