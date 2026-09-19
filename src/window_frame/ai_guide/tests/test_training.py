from window_frame.ai_guide.utils.settings import Settings, DatasetConfig, ModelConfig, TrainingConfig, OutputConfig
"""Tiny CPU end-to-end test; does not train on the user's dataset."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from window_frame.ai_guide.tests.test_dataset import make_dataset
from window_frame.ai_guide.training.train_model import run_training
from window_frame.ai_guide.training.save_and_load import load_checkpoint
from window_frame.ai_guide.utils.save_files import file_hash, write_json
from window_frame.ai_guide.prediction.enhance_image import run_prediction
from window_frame.ai_guide.evaluation.evaluate_model import run_evaluation


class TrainingTests(unittest.TestCase):
    def test_training_resume_evaluate_predict(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            make_dataset(data, count=12)
            config = Settings(dataset=DatasetConfig(root=str(data), image_size=32, validation_fraction=.5,
                test_fraction=.2, scene_groups='', class_meanings_confirmed=True),
                model=ModelConfig(base_channels=8, max_correction=.25, confidence=.8),
                training=TrainingConfig(skip_audit=False, max_train_samples=None, max_validation_samples=None, epochs=1, segmentation_epochs=0, enhancement_epochs=0, batch_size=2,
                    learning_rate=.001, weight_decay=.0001, workers=0, cpu_threads=1, audit_workers=2, seed=42, device='cpu',
                    mixed_precision=False, accumulation_steps=2, tensorboard=False),
                output=OutputConfig(root=str(root / 'runs')))
            with patch('window_frame.ai_guide.training.save_and_load.torch.save', wraps=torch.save) as save, \
                 patch('window_frame.ai_guide.training.save_and_load.file_hash', wraps=file_hash) as hash_checkpoint, \
                 patch('window_frame.ai_guide.training.train_model.write_json', wraps=write_json) as write:
                run = run_training(config)
            self.assertEqual(save.call_count, 1)
            self.assertEqual(hash_checkpoint.call_count, 1)
            self.assertEqual(sum(call.args[0].name == 'history.json' for call in write.call_args_list), 1)
            comparisons = list((run / 'visualizations').glob('epoch_0001_*.png'))
            self.assertEqual(len(comparisons), 5)
            checkpoint = run / 'checkpoints/best.pt'
            model, state = load_checkpoint(checkpoint)
            self.assertEqual(state['epoch'], 0)
            self.assertTrue(checkpoint.with_suffix('.json').exists())
            last = run / 'checkpoints/last.pt'
            self.assertEqual(checkpoint.read_bytes(), last.read_bytes())
            self.assertEqual(checkpoint.with_suffix('.json').read_bytes(), last.with_suffix('.json').read_bytes())
            self.assertEqual(json.loads(checkpoint.with_suffix('.json').read_text())['sha256'], file_hash(checkpoint))
            evaluation = run_evaluation(checkpoint, output_root=root/'runs', device='cpu')
            self.assertTrue((evaluation/'summary.json').exists())
            prediction = run_prediction(checkpoint, data/'00001/input.png', root/'runs', 'cpu')
            summary = json.loads((prediction/'summary.json').read_text())
            folder = prediction / summary['successful'][0]
            for name in ('enhanced.png','rail_classes.png','rail_guide.png','soft_mask.npy','metadata.json'):
                self.assertTrue((folder/name).exists())
            config.training.epochs = 2
            with patch('window_frame.ai_guide.training.train_model.split_dataset') as split:
                resumed = run_training(config, run/'checkpoints/last.pt')
            split.assert_not_called()
            _, state = load_checkpoint(resumed/'checkpoints/last.pt')
            self.assertEqual(state['epoch'], 1)
            self.assertNotEqual(run, resumed)
            self.assertIn('early_stopping', state)
            config.training.epochs = 5
            with patch('window_frame.ai_guide.training.train_model.EarlyStopping.update', return_value=True):
                stopped = run_training(config)
            summary = json.loads((stopped/'summary.json').read_text())
            self.assertEqual(summary['stop_reason'], 'early_stopping')
            self.assertEqual(summary['completed_epoch'], 1)
            self.assertTrue((stopped/'checkpoints/last.pt').exists())
