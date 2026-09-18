import unittest
import numpy as np
import torch
from PIL import Image
from window_frame.ai_guide.model.window_frame_model import WindowFrameModel
from window_frame.ai_guide.prediction.enhance_image import predict_image
from window_frame.ai_guide.visualization.compare_images import to_image
from window_frame.ai_guide.evaluation.measure_accuracy import AccuracyMeter
from window_frame.ai_guide.prediction.process_large_image import tile_starts


class PredictionTests(unittest.TestCase):
    def test_original_size_and_exact_copy(self):
        torch.set_num_threads(1)
        array = np.random.default_rng(42).integers(0, 256, (37, 59, 3), dtype=np.uint8)
        model = WindowFrameModel(base_channels=8).eval()
        result = predict_image(model, Image.fromarray(array), 32, torch.device('cpu'))
        self.assertEqual(result['enhanced'].shape, (1, 3, 37, 59))
        np.testing.assert_array_equal(array, np.asarray(to_image(result['enhanced'][0])))

    def test_large_photo_tile_coverage(self):
        for height, width in [(3129, 3000), (2310, 3102), (17, 4097), (4097, 17),
                              (720, 1280), (1080, 1920), (2160, 3840), (9, 7)]:
            for length in (height, width):
                coverage = np.zeros(length, dtype=bool)
                for start in tile_starts(length, 512, 128):
                    coverage[start:start+512] = True
                self.assertTrue(coverage.all())
        for overlap in (-1, 512, 600):
            with self.assertRaises(ValueError):
                tile_starts(1920, 512, overlap)

    def test_overlapping_tiles_preserve_detail_and_protection(self):
        class PredictableModel(torch.nn.Module):
            settings = {'confidence': .8}

            def forward(self, image):
                batch, _, height, width = image.shape
                logits = image.new_full((batch, 6, height, width), -30)
                protected = image[:, 0] < .5
                logits[:, 2] = torch.where(protected, -30., 30.)
                logits[:, 5] = torch.where(protected, 30., -30.)
                return dict(class_logits=logits, mask_logits=image.new_full((batch,1,height,width), 30),
                            residual=image * .1)

        torch.set_num_threads(1)
        # A high-frequency pattern catches accidental whole-image downsampling.
        array = np.random.default_rng(3).integers(0, 256, (720,1280,3), dtype=np.uint8)
        result = predict_image(PredictableModel(), Image.fromarray(array), 512, torch.device('cpu'))
        original = torch.from_numpy(array.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
        protected = original[:, :1] < .5
        expected = torch.where(protected, original, (original * 1.1).clamp(0,1))
        torch.testing.assert_close(result['enhanced'], expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result['probabilities'].sum(1), torch.ones(1,720,1280))
        self.assertTrue(torch.equal(result['enhanced'][protected.expand_as(original)],
                                    original[protected.expand_as(original)]))

    def test_small_photo_and_resized_mode(self):
        model = WindowFrameModel(base_channels=8).eval()
        image = Image.new('RGB', (7, 9), (50, 100, 150))
        for mode in ('tiled', 'resized'):
            result = predict_image(model, image, 32, torch.device('cpu'), mode=mode)
            np.testing.assert_array_equal(np.asarray(image), np.asarray(to_image(result['enhanced'][0])))

    def test_exact_arbitrary_large_dimensions(self):
        """Full prediction/combination at the user's sizes, with a cheap fake model."""
        class BackgroundModel(torch.nn.Module):
            settings = {'confidence': .8}

            def forward(self, image):
                batch, _, height, width = image.shape
                logits = image.new_full((batch, 6, height, width), -30)
                logits[:, 0] = 30
                return dict(class_logits=logits,
                            mask_logits=image.new_full((batch, 1, height, width), -30),
                            residual=torch.zeros_like(image))

        torch.set_num_threads(1)
        for width, height in [(3000, 3129), (3102, 2310), (17, 1031), (1031, 17), (1, 1)]:
            with self.subTest(width=width, height=height):
                image = Image.new('RGB', (width, height), (50, 100, 150))
                result = predict_image(BackgroundModel(), image, 512, torch.device('cpu'))
                for name, channels in [('enhanced', 3), ('soft_mask', 1), ('probabilities', 6), ('edit_gate', 1)]:
                    self.assertEqual(result[name].shape, (1, channels, height, width))
                    self.assertTrue(torch.isfinite(result[name]).all())
                np.testing.assert_array_equal(np.asarray(to_image(result['enhanced'][0])), np.asarray(image))
                self.assertFalse(result['edit_gate'].any())
                del result

    def test_real_model_accepts_rectangular_sections(self):
        model = WindowFrameModel(base_channels=8).eval()
        shapes = []
        hook = model.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape[-2:])))
        try:
            result = predict_image(model, Image.new('RGB', (59, 37)), 512, torch.device('cpu'))
        finally:
            hook.remove()
        self.assertEqual(shapes, [(37, 59)])
        self.assertEqual(result['enhanced'].shape[-2:], (37, 59))

    def test_perfect_metrics(self):
        image = torch.rand(1, 3, 16, 16)
        classes = torch.full((1,16,16), 2, dtype=torch.long)
        probabilities = torch.zeros(1,6,16,16)
        probabilities[:,2] = 1
        result = dict(probabilities=probabilities,soft_mask=torch.ones(1,1,16,16),enhanced=image)
        batch = dict(image=image,target=image,classes=classes,mask=torch.ones(1,1,16,16))
        meter = AccuracyMeter()
        meter.update(result,batch)
        metrics = meter.compute()
        self.assertEqual(metrics['frame_iou'], 1)
        self.assertEqual(metrics['frame_mae'], 0)
        self.assertEqual(metrics['class_brier_score'], 0)
        self.assertEqual(metrics['confidence_calibration_error'], 0)
        self.assertIsNone(metrics['protected_mae'])
