import unittest
import torch
from window_frame.ai_guide.model.window_frame_model import WindowFrameModel, compose
from window_frame.ai_guide.training.calculate_loss import calculate_loss


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_shapes_and_backward(self):
        model = WindowFrameModel(base_channels=8)
        image = torch.rand(2, 3, 33, 41)
        result = model(image)
        self.assertEqual(result['class_logits'].shape, (2, 6, 33, 41))
        self.assertEqual(result['soft_mask'].shape, (2, 1, 33, 41))
        self.assertTrue(torch.equal(image, result['enhanced']))
        batch = dict(image=image, target=image * .9, mask=torch.rand(2, 1, 33, 41),
                     classes=torch.randint(0, 6, (2, 33, 41)))
        loss = calculate_loss(result, batch)['total']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.class_head.weight.grad)
        self.assertIsNotNone(model.correction_head[-1].weight.grad)

    def test_exact_protection(self):
        image = torch.rand(1, 3, 16, 16)
        residual = torch.ones_like(image) * .2
        for label in (0, 5):
            probabilities = torch.zeros(1, 6, 16, 16)
            probabilities[:, label] = 1
            output, gate = compose(image, residual, probabilities, torch.ones(1, 1, 16, 16))
            self.assertTrue(torch.equal(image, output))
            self.assertFalse(gate.any())
        probabilities[:] = 1/6
        output, _ = compose(image, residual, probabilities, torch.ones(1, 1, 16, 16))
        self.assertTrue(torch.equal(image, output))
        probabilities[:] = 0
        probabilities[:, 2] = 1
        output, _ = compose(image, residual, probabilities, torch.ones(1, 1, 16, 16))
        self.assertFalse(torch.equal(image, output))

    def test_all_unknown_labels(self):
        image = torch.rand(1, 3, 16, 16)
        model = WindowFrameModel(base_channels=8)
        batch = dict(image=image, target=image, mask=torch.ones(1, 1, 16, 16),
                     classes=torch.full((1, 16, 16), 255, dtype=torch.long))
        loss = calculate_loss(model(image), batch)['total']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
