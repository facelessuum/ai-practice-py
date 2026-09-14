from typing import TypedDict


class ImageFileType(TypedDict):
    file_name: str
    path: str


class ProjectResultType(TypedDict):
    masks: list[ImageFileType]
    images: list[ImageFileType]


class TestResultType(TypedDict):
    path: str
    projectName: str
    results: list[ProjectResultType]


class ResultDataType(TypedDict):
    project_data: list[TestResultType]
    path: str
    test_run: str
