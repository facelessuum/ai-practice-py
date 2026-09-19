import unittest
from unittest.mock import patch

from window_frame.ai_guide.utils.settings import (
    DatasetConfig, ModelConfig, OutputConfig, Settings, TrainingConfig, settings,
)
from window_frame.ai_guide.training import train_model
from window_frame.ai_guide.utils.check_settings import validate_config


class SettingsTests(unittest.TestCase):
    def test_grouped_configuration_and_shared_instance(self):
        self.assertIs(train_model.settings, settings)
        config = Settings()
        self.assertIsInstance(config.dataset, DatasetConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.output, OutputConfig)
        self.assertFalse(hasattr(config, 'to_dict'))
        validate_config(config)

    def test_instances_have_separate_nested_settings(self):
        first, second = Settings(), Settings()
        first.training.epochs = 999
        first.dataset.image_type.append('other')
        self.assertEqual(second.training.epochs, 10)
        self.assertNotIn('other', second.dataset.image_type)

    def test_cli_without_overrides_uses_shared_settings(self):
        with patch('sys.argv', ['train-window-frame']), \
             patch.object(train_model, 'run_training') as run:
            train_model.main()
        self.assertIs(run.call_args.args[0], settings)

    def test_cli_label_confirmation_updates_settings(self):
        config = Settings(dataset=DatasetConfig(class_meanings_confirmed=False))
        with patch.object(train_model, 'settings', config), \
             patch('sys.argv', ['train-window-frame', '--confirm-labels']), \
             patch.object(train_model, 'run_training') as run:
            train_model.main()
        passed = run.call_args.args[0]
        self.assertTrue(passed.dataset.class_meanings_confirmed)
        self.assertIs(passed, config)
        self.assertTrue(config.dataset.class_meanings_confirmed)

    def test_cli_passes_settings_object(self):
        config = Settings()
        config.training.epochs = 12
        with patch.object(train_model, 'settings', config), \
             patch('sys.argv', ['train-window-frame', '--epochs', '3']), \
             patch.object(train_model, 'run_training') as run:
            train_model.main()
        passed = run.call_args.args[0]
        self.assertIsInstance(passed, Settings)
        self.assertEqual(passed.training.epochs, 3)
        self.assertIs(passed, config)
        self.assertEqual(config.training.epochs, 3)
