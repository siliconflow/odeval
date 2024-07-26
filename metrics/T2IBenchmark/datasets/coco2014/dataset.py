from typing import Any, Callable, Optional

from datasets import load_dataset
from PIL import Image

from T2IBenchmark.loaders import ImageDataset


class COCOImageDataset(ImageDataset):
    def __init__(self, preprocess_fn: Optional[Callable[[Image.Image], Any]] = None):
        super().__init__(paths=[], preprocess_fn=preprocess_fn)
        self.ds = load_dataset("stasstaf/MS-COCO-validation")["test"]

    def __getitem__(self, idx: int) -> Any:
        image = self.ds[idx]["image"]
        preproc = self.preprocess_fn(image)
        return preproc

    def __len__(self) -> int:
        return len(self.ds)
