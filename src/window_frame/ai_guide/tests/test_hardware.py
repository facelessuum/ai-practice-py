import unittest
from window_frame.ai_guide.training.configure_hardware import resource_plan


class HardwareTests(unittest.TestCase):
    def test_auto_cpu_balances_threads_and_workers(self):
        for count in (1, 2, 8, 16, 64):
            plan = resource_plan({}, 'cpu', available=count)
            self.assertEqual(plan['cpu_threads'] + plan['workers'], count)
            self.assertGreaterEqual(plan['audit_workers'], 1)
            self.assertLessEqual(plan['audit_workers'], count)

    def test_auto_gpu_reserves_cpu_for_loading(self):
        plan = resource_plan({}, 'cuda', available=16)
        self.assertEqual(plan['workers'], 8)
        self.assertEqual(plan['cpu_threads'], 4)

    def test_manual_settings_and_invalid_values(self):
        plan = resource_plan(dict(workers=0, cpu_threads=2, audit_workers=1), 'cpu', available=16)
        self.assertEqual(plan['workers'], 0)
        self.assertEqual(plan['cpu_threads'], 2)
        for setting, value in [('workers', -1), ('cpu_threads', 0), ('audit_workers', 'wrong')]:
            with self.assertRaises(ValueError):
                resource_plan({setting: value}, 'cpu', available=16)
