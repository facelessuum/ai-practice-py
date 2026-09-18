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
from window_frame.ai_guide.prediction.enhance_image import run_prediction
from window_frame.ai_guide.evaluation.evaluate_model import run_evaluation


class TrainingTests(unittest.TestCase):
    def test_training_resume_evaluate_predict(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            make_dataset(data)
            config = dict(dataset=dict(root=str(data), image_size=32, validation_fraction=.2,
                test_fraction=.2, scene_groups='', class_meanings_confirmed=True),
                model=dict(base_channels=8, max_correction=.25, confidence=.8),
                training=dict(epochs=1, segmentation_epochs=0, enhancement_epochs=0, batch_size=2,
                    learning_rate=.001, weight_decay=.0001, workers=0, cpu_threads=1, audit_workers=2, seed=42, device='cpu',
                    mixed_precision=False, accumulation_steps=2, tensorboard=False),
                output=dict(root=str(root / 'runs')))
            run = run_training(config)
            checkpoint = run / 'checkpoints/best.pt'
            model, state = load_checkpoint(checkpoint)
            self.assertEqual(state['epoch'], 0)
            self.assertTrue(checkpoint.with_suffix('.json').exists())
            evaluation = run_evaluation(checkpoint, output_root=root/'runs', device='cpu')
            self.assertTrue((evaluation/'summary.json').exists())
            prediction = run_prediction(checkpoint, data/'00001/input.png', root/'runs', 'cpu')
            summary = json.loads((prediction/'summary.json').read_text())
            folder = prediction / summary['successful'][0]
            for name in ('enhanced.png','rail_classes.png','rail_guide.png','soft_mask.npy','metadata.json'):
                self.assertTrue((folder/name).exists())
            config['training']['epochs'] = 2
            resumed = run_training(config, run/'checkpoints/last.pt')
            _, state = load_checkpoint(resumed/'checkpoints/last.pt')
            self.assertEqual(state['epoch'], 1)
            self.assertNotEqual(run, resumed)
            self.assertIn('early_stopping', state)
            config['training']['epochs'] = 5
            with patch('window_frame.ai_guide.training.train_model.EarlyStopping.update', return_value=True):
                stopped = run_training(config)
            summary = json.loads((stopped/'summary.json').read_text())
            self.assertEqual(summary['stop_reason'], 'early_stopping')
            self.assertEqual(summary['completed_epoch'], 1)
            self.assertTrue((stopped/'checkpoints/last.pt').exists())
