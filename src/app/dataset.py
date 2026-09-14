from pathlib import Path
from .utils.type.dataset.dataset import (
    ProjectResultType,
    ImageFileType,
    TestResultType,
    ResultDataType,
)

data_dir = Path("data")
results_dir = data_dir / "result"

image_extensions = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "avif"}


def get_images_in_fold(fold: Path) -> list[ImageFileType]:
    return [
        {"path": str(path), "file_name": str(path.name)}
        for path in fold.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in image_extensions
    ]


def get_project_results(proj_dir: Path) -> list[ProjectResultType]:

    masks_dir = proj_dir / "masks"
    images_dir = proj_dir / "images"

    results: list[ProjectResultType] = [
        {
            "masks": get_images_in_fold(masks_dir),
            "images": get_images_in_fold(images_dir),
        }
    ]

    return results


def get_test_results(test_dir: Path) -> list[TestResultType]:
    test_results: list[TestResultType] = []
    for project_dir in test_dir.iterdir():
        if not project_dir.is_dir():
            continue

        proj_result = get_project_results(project_dir)
        test_results.append(
            {
                "projectName": project_dir.name,
                "path": str(project_dir),
                "results": proj_result,
            }
        )
    return test_results


def get_results_data() -> list[ResultDataType]:
    results: list[ResultDataType] = []
    for test_dir in results_dir.iterdir():
        if not test_dir.is_dir():
            continue
        results.append(
            {
                "project_data": get_test_results(test_dir),
                "test_run": test_dir.name,
                "path": str(test_dir),
            }
        )

    return results
