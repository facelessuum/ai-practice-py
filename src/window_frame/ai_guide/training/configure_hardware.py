"""Choose CPU resources automatically without oversubscribing data workers."""
import math
import os
from pathlib import Path
import torch


def available_cpu_count():
    """Respect process affinity and the common Linux container CPU quota."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def resource_plan(training, device_type, available=None):
    available = available_cpu_count() if available is None else max(1, available)
    workers = training.get('workers', 'auto')
    if workers == 'auto':
        workers = min(8, max(0, available - 1)) if device_type == 'cuda' else min(4, available // 4)
    threads = training.get('cpu_threads', 'auto')
    if threads == 'auto':
        threads = min(4, available) if device_type == 'cuda' else max(1, available - workers)
    audit_workers = training.get('audit_workers', 'auto')
    if audit_workers == 'auto':
        audit_workers = min(8, available)
    for name, value, minimum in [('workers', workers, 0), ('cpu_threads', threads, 1),
                                  ('audit_workers', audit_workers, 1)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f'{name} must be "auto" or an integer >= {minimum}')
    return dict(available_cpus=available, workers=workers, cpu_threads=threads,
                audit_workers=audit_workers)


def configure_hardware(training, device, plan):
    torch.set_num_threads(plan['cpu_threads'])
    cuda = device.type == 'cuda'
    benchmark = cuda and training.get('cudnn_benchmark', True)
    torch.backends.cudnn.benchmark = benchmark
    tf32 = cuda and training.get('allow_tf32', True)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return dict(**plan, device=str(device), torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                cudnn_benchmark=benchmark, allow_tf32=tf32,
                gpu_name=torch.cuda.get_device_name(device) if cuda else None,
                gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory if cuda else None,
                batch_size=training['batch_size'], prefetch_factor=2 if plan['workers'] else None,
                channels_last=cuda and training.get('channels_last', True))
