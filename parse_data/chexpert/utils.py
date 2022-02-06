import re
from enum import Enum
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

from utils.img import center_crop, remove_border


class Pathology(Enum):
    """
    Enum class of CheXpert pathlogies
    """

    enlarged_cardiomediastinum = "enlarged-cardiomediastinum"
    cardiomegaly = "cardiomegaly"
    lung_opacity = "lung-opacity"
    lung_lesion = "lung-lesion"
    edema = "edema"
    consolidation = "consolidation"
    pneumonia = "pneumonia"
    atelectasis = "atelectasis"
    pneumothorax = "pneumothorax"
    pleural_effusion = "pleural-effusion"
    pleural_other = "pleural-other"
    fracture = "fracture"
    support_device = "support-device"


class CheXpertImg:
    def __init__(
        self,
        idx: int,
        path: str,
        sex: str,
        age: int,
        pathologies: List[int],
    ) -> None:

        match = re.search(r"(.+?)/patient(\d+?)/study(\d+?)/view(\d+?)", path)
        if match is None:
            print("Invalid path")
            exit(1)

        self.idx = idx
        self.loc: str = match.group(1)
        self.patient: str = match.group(2)
        self.study: str = match.group(3)
        self.view: str = match.group(4)

        self.sex = sex
        self.age = age
        self.pathologies: List[Pathology] = []

        # presence is the value in CheXpert labels CSV (-1, 0 or 1)
        for pathology, presence in zip(Pathology, pathologies):
            if presence > 0:
                self.pathologies.append(pathology)

    def __repr__(self) -> str:
        string = f"{self.idx}_{self.loc}_{self.patient}_{self.study}_{self.view}_{self.sex}_{self.age}"
        for pathology in self.pathologies:
            string += f"_{pathology.value}"
        return f"{string}.png"

    @property
    def path(self) -> str:
        return f"{self.loc}/patient{self.patient}/study{self.study}/view{self.view}_frontal.jpg"

    def img(self, base_dir: Path) -> np.ndarray:
        return cv.imread(str(base_dir / self.path), cv.IMREAD_GRAYSCALE)

    def proc_img(self, size: int, base_dir: Path) -> np.ndarray:
        img = remove_border(self.img(base_dir))
        img = center_crop(img)
        img = cv.resize(img, (size, size))

        if np.mean(img) < 50:
            print(img, "has low brightness")
        return img

    @classmethod
    def from_str(cls, string: str) -> "CheXpertImg":
        idx, loc, patient, study, view, sex, age, *others = string.split("_")
        pathologies = [1 if pathology.value in string else 0 for pathology in Pathology]
        return CheXpertImg(
            int(idx),
            f"{loc}/patient{patient}/study{study}/view{view}_frontal.jpg",
            sex,
            int(age),
            pathologies,
        )
