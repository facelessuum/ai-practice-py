from pathlib import Path
import torch
import numpy as np


def main() -> Path:

    output_path = Path("model")
    output_path.mkdir(parents=True, exist_ok=True)
    run_number = 1

    while True:
        output_dir = output_path / f"run_{run_number:04d}"
        try:
            output_dir.mkdir()
        except FileExistsError:
            run_number += 1
            continue
        return output_dir

    return
    a = np.array(
        [
            [[1, 2, 3, 4], [4, 5, 6, 2], [7, 8, 9, 1]],
            [[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4]],
            [[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4]],
            [[1, 2, 3, 4], [4, 5, 6, 7], [1, 2, 3, 4]],
        ],
        dtype="int8",
    )

    b = np.ones([2, 2, 3])

    print(a)
    print(f"A {a.shape}", "SHAPE")
    print(f"A {a.ndim}D", "DIMENSION")
    print(f"A {a.dtype}", "TYPE")
    print(f"A {a.itemsize}", "ITEMSIZE")

    print(b)
    print(f"B {b.shape}", "SHAPE")
    print(f"B {b.ndim}D", "DIMENSION")
    print(f"B {b.dtype}", "TYPE")
    print(f"B {b.itemsize}", "ITEMSIZE")
