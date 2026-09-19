"""Read-only parallel dataset audit. Checks IDs, not their semantic correctness."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from .load_samples import input_path


def discover_samples(root):
    """Return the minimum sample metadata needed when validation is skipped."""
    root = Path(root)
    samples = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        path = input_path(folder)
        with Image.open(path) as image:
            source = np.asarray(image.convert("RGB"))
        samples.append(dict(
            id=folder.name,
            empty=None,
            input_hash=hashlib.sha256(source.tobytes() + str(source.shape).encode()).hexdigest(),
            files={},
        ))
    if not samples:
        raise ValueError("No sample folders found")
    return samples


def check_sample(folder):
    """Each worker owns its images and returns a small summary, not image arrays."""
    try:
        paths = [input_path(folder)] + [folder / x for x in ('output.png', 'mask.png', 'rail_classes.png')]
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.copy())
        if len({im.size for im in images}) != 1:
            raise ValueError('Image/label dimensions differ')
        if images[2].mode != 'L' or images[3].mode not in ('L', 'P'):
            raise ValueError('Masks must be grayscale; class maps must contain raw IDs')
        mask, classes = np.asarray(images[2]), np.asarray(images[3])
        ids, counts = np.unique(classes, return_counts=True)
        if not set(ids.tolist()) <= {0, 1, 2, 3, 4, 5, 255}:
            raise ValueError(f'Invalid rail IDs: {ids.tolist()}')
        source = np.asarray(images[0].convert('RGB'))
        target = np.asarray(images[1].convert('RGB'))
        protected = classes == 5
        return dict(
            sample=dict(id=folder.name, empty=not bool(mask.any()),
                input_hash=hashlib.sha256(source.tobytes() + str(source.shape).encode()).hexdigest(),
                files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),
            class_pixels={str(int(k)): int(v) for k, v in zip(ids, counts)},
            unknown_frame_pixels=int(((mask > 0) & (classes == 0)).sum()),
            protected_error=float(np.abs(source.astype(float) - target)[protected].sum()),
            protected_count=int(protected.sum()) * 3,
        )
    except (OSError, ValueError) as error:
        return dict(error=dict(id=folder.name, error=str(error)))


def audit_dataset(root, workers=1, on_progress=None):
    if workers < 1:
        raise ValueError('Audit workers must be positive')
    root = Path(root)
    report = dict(root=str(root.resolve()), samples=[], errors=[], class_pixels=Counter(),
                  empty_masks=0, unknown_frame_pixels=0, protected_target_mae=None)
    protected_error = protected_count = 0
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if on_progress:
        on_progress(0, len(folders))
    # map preserves sorted order, keeping audit fingerprints stable across worker counts.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(executor.map(check_sample, folders), 1):
            if on_progress:
                on_progress(completed, len(folders))
            if 'error' in result:
                report['errors'].append(result['error'])
                continue
            report['samples'].append(result['sample'])
            report['class_pixels'].update(result['class_pixels'])
            report['empty_masks'] += int(result['sample']['empty'])
            report['unknown_frame_pixels'] += result['unknown_frame_pixels']
            protected_error += result['protected_error']
            protected_count += result['protected_count']
    report['class_pixels'] = dict(report['class_pixels'])
    if protected_count:
        report['protected_target_mae'] = protected_error / protected_count / 255
    if not report['samples']:
        report['errors'].append(dict(id='', error='No usable samples found'))
    return report
