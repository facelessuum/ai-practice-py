import hashlib
import json
from pathlib import Path
import random
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from window_frame.ai_guide.utils.save_files import file_hash, write_json
from window_frame.ai_guide.utils.setup_device import choose_device, seed_everything


class UtilityTests(unittest.TestCase):
    def test_json_output_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'nested' / 'settings.json'
            values = {'epochs': 3, 'enabled': True}
            write_json(path, values)
            self.assertEqual(json.loads(path.read_text()), values)
            self.assertEqual(file_hash(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_shared_device_selects_cuda_or_cpu(self):
        for available, expected in [(True, 'cuda'), (False, 'cpu')]:
            with self.subTest(cuda_available=available), \
                 patch('torch.cuda.is_available', return_value=available):
                shared = runpy.run_module('utils.utils')
                self.assertEqual(shared['DEVICE'], torch.device(expected))

    def test_explicit_cpu(self):
        self.assertEqual(choose_device('cpu'), torch.device('cpu'))

    def test_unavailable_cuda(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA requested but unavailable'):
                choose_device('cuda')

    def test_repeatable_random_seeds(self):
        seed_everything(42)
        expected = (random.random(), np.random.random(), torch.rand(3))
        seed_everything(42)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))
