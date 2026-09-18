import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from window_frame.ai_guide.dataset.load_samples import WindowFrameDataset
from window_frame.ai_guide.dataset.check_labels import audit_dataset
from window_frame.ai_guide.dataset.split_dataset import split_dataset


def make_dataset(root, count=6):
    for i in range(count):
        folder = Path(root) / f'{i:05d}'
        folder.mkdir(parents=True)
        image = np.full((24, 32, 3), 50 + i*12, dtype=np.uint8)
        mask = np.zeros((24, 32), dtype=np.uint8)
        classes = np.zeros_like(mask)
        target = image.copy()
        if i % 2:
            mask[4:20, 10:13] = 255
            classes[4:20, 10:13] = 2
            target[4:20, 10:13] += 20
        for name, array in [('input.png', image), ('output.png', target), ('mask.png', mask), ('rail_classes.png', classes)]:
            Image.fromarray(array).save(folder / name)


class DatasetTests(unittest.TestCase):
    def test_read_audit_and_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            make_dataset(root)
            report = audit_dataset(root)
            self.assertFalse(report['errors'])
            self.assertEqual(report, audit_dataset(root, workers=3))
            self.assertEqual(report['empty_masks'], 3)
            sample = WindowFrameDataset(root, ['00001'], 32)[0]
            self.assertEqual(sample['image'].shape, (3, 32, 32))
            self.assertTrue(set(sample['classes'].unique().tolist()) <= {0, 2, 255})
            Image.fromarray(np.zeros((24,32), dtype=np.uint8)).save(Path(root)/'00001/rail_classes.png')
            sample = WindowFrameDataset(root, ['00001'], 32)[0]
            self.assertIn(255, sample['classes'].unique().tolist())
            Image.fromarray(np.full((24,32), 6, dtype=np.uint8)).save(Path(root)/'00000/rail_classes.png')
            self.assertTrue(audit_dataset(root)['errors'])

    def test_duplicates_and_scenes_stay_together(self):
        samples = [dict(id=str(i), input_hash=str(i)) for i in range(12)]
        samples[1]['input_hash'] = samples[0]['input_hash']
        groups = {str(i): str(i) for i in range(12)}
        groups['2'] = groups['1']
        split = split_dataset(samples, scene_groups=groups)
        owners = {sample: key for key in ('train','validation','test') for sample in split[key]}
        self.assertEqual(owners['0'], owners['1'])
        self.assertEqual(owners['1'], owners['2'])
        self.assertEqual(len(owners), 12)
        self.assertEqual(split, split_dataset(samples, scene_groups=groups))
