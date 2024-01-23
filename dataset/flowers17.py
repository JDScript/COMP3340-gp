from pathlib import Path
from typing import Callable, Optional, Tuple, Any
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
    verify_str_arg,
)
from scipy.io import loadmat
import PIL.Image
import numpy as np


class Flowers17(VisionDataset):
    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/"
    _file_dict = {
        "image": ("17flowers.tgz", "b59a65d8d1a99cd66944d474e1289eab"),
        "datasplits": ("datasplits.mat", "4828cddfd0d803c5abbdebcb1e148a1e"),
    }
    _splits_map = {"train": "trn", "val": "val", "test": "tst"}
    _classes = [
        "Daffodil",
        "Snowdrop",
        "Lily Valley",
        "Bluebell",
        "Crocus",
        "Iris",
        "Tigerlily",
        "Tulip",
        "Fritillary",
        "Sunflower",
        "Daisy",
        "Colts Foot",
        "Dandelalion",
        "Cowslip",
        "Buttercup",
        "Windflower",
        "Pansy",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_id: str = "1",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._split_id = verify_str_arg(split_id, "split_id", ("1", "2", "3"))
        self._base_folder = Path(self.root) / "flowers-17"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        datasplits = loadmat(
            self._base_folder / self._file_dict["datasplits"][0], squeeze_me=True
        )

        image_ids = datasplits[self._splits_map[self._split] + self._split_id]
        image_id_to_label = dict(enumerate(np.repeat(range(len(self._classes)), 80), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:04d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["datasplits"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["datasplits"]:
            filename, md5 = self._file_dict[id]
            download_url(
                self._download_url_prefix + filename, str(self._base_folder), md5=md5
            )
