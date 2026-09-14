import numpy as np

def main() -> None:
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

    print("---")
    print("---")
    print("---")

    print(b)
    print(f"B {b.shape}", "SHAPE")
    print(f"B {b.ndim}D", "DIMENSION")
    print(f"B {b.dtype}", "TYPE")
    print(f"B {b.itemsize}", "ITEMSIZE")
