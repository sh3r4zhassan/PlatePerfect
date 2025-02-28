import dataclasses as dcls
import functools
import glob
from types import MappingProxyType
from typing import List, NamedTuple

import cv2
import numpy as np
import open_clip
import torch
from numpy.typing import NDArray
from PIL import Image

import Menu

YOLO_CLASSES = Menu.INGREDIETNS
__YOLO_CLS_SET = frozenset(YOLO_CLASSES)
__YOLO_CLS_TO_IDX = MappingProxyType({x: i for i, x in enumerate(YOLO_CLASSES)})


def one_hot_by_ingredients(ingredients: List[str]) -> NDArray:
    assert all((ing in __YOLO_CLS_SET) for ing in ingredients)
    array = np.zeros([len(YOLO_CLASSES)])
    ing_idx = [__YOLO_CLS_TO_IDX[ing] for ing in ingredients]
    array[ing_idx] = 1
    return array


class Mask666(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    bounding_boxes: NDArray
    mask: NDArray
    polygon: list
    probability: float
    label: str


@dcls.dataclass(frozen=True)
class Image666:
    # Name of the image.
    name: str

    # Shape = [H, W, 3]
    # Format: RGB
    image: NDArray

    # Shape = [n]
    # Format: index of the classes
    classes: NDArray

    # Shape = [n, H, W]
    seg_masks: NDArray

    # Shape = [n, 4]
    # Format: [x_min, y_min, x_max, y_max]
    bounding_boxes: NDArray

    # Shape = [n]
    # Take the exponential to get the actual probability
    log_prob: NDArray

    @property
    def ingredients(self) -> List[str]:
        return [YOLO_CLASSES[i] for i in self.classes]

    @property
    def one_hot(self) -> NDArray:
        return one_hot_by_ingredients(self.ingredients)

    @functools.cached_property
    def argmax_seg_mask(self):
        mapping = {idx: klass for idx, klass in enumerate(self.classes)}
        argmax = np.argmax(self.seg_masks, axis=0)
        argmax_map = np.vectorize(mapping.get)
        return argmax_map(argmax)

    @classmethod
    def from_messy(cls, name: str, image: NDArray, results: List[Mask666]):
        classes = np.array([__YOLO_CLS_TO_IDX[r.label] for r in results])
        bounding_boxes = np.array([[r.x0, r.y0, r.x1, r.y1] for r in results])
        seg_masks = np.array([r.mask for r in results])
        log_prob = np.array([np.log(r.probability) for r in results])
        return cls(
            name=name,
            image=image,
            classes=classes,
            seg_masks=seg_masks,
            bounding_boxes=bounding_boxes,
            log_prob=log_prob,
        )


@dcls.dataclass(frozen=True)
class Menu666:
    items: List[Image666]

    @property
    def _items_hamming_matrix(self):
        return np.array([img.one_hot for img in self.items])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def match_by_ingredient(self, image: Image666, top_k: int = -1) -> List[Image666]:
        hamming = self._items_hamming_matrix
        img_hamming = image.one_hot

        # Dot product on the second dimension
        matching_matrix = np.einsum("ij,i->j", hamming, img_hamming)

        if top_k < 0:
            top_k = len(matching_matrix)

        argsort = np.argsort(matching_matrix)[::-1]
        return [self.items[idx] for idx in argsort[:top_k]]

    def match_by_orb(
        self, image: Image666, top_k: int = -1, top_matches: int = 100
    ) -> List[Image666]:
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        descriptor = self.detect(image)

        matches = [bf.match(descriptor, desc) for desc in self.items_orb]

        if top_matches > 0:
            matches = [
                (idx, sorted(match_item, key=lambda x: x.distance)[:top_matches])
                for idx, match_item in matches
            ]

        best_by_distance = sorted(
            matches, key=lambda idx_match: sum([m.distance for m in idx_match[1]])
        )

        if top_k < 0:
            top_k = len(best_by_distance)

        best_indices = [idx for idx, _ in best_by_distance[:top_k]]
        return [self.items[idx] for idx in best_indices]

    @classmethod
    def from_items(cls, items: List[Image666]):
        return cls(items)

    @functools.cached_property
    def orb(self):
        return cv2.ORB_create()

    def detect(self, image: Image666):
        _, desc = self.orb.detectAndCompute(image.argmax_seg_mask, None)
        return desc

    @functools.cache
    def items_orb(self):
        return [self.detect(img) for img in self.items]


def cosine(a: torch.Tensor, b: torch.Tensor):
    a = a / a.norm(2, dim=-1, keepdim=True)
    b = b / b.norm(2, dim=-1, keepdim=True)
    return (a * b).sum()


if __name__ == "__main__":

    def imread(img: str):
        img = preprocess(Image.open(img))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[None, ...]
        return img

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    orb = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img = imread("90.jpg")
    rotated = imread("90r.jpg")
    flipped = imread("90f.jpg")
    flipped_rotated = imread("90fr.jpg")
    others = [
        imread(jpg)
        for jpg in glob.glob("*.jpg")
        if jpg not in ["90.jpg", "90r.jpg", "90f.jpg", "90fr.jpg"]
    ]

    with torch.no_grad():
        print(img.shape)
        encoded = model.encode_image(img)
        encoded_rotated = model.encode_image(rotated)
        encoded_flipped = model.encode_image(flipped)
        encoded_flipped_rotated = model.encode_image(flipped_rotated)
        encoded_others = [model.encode_image(im) for im in others]

        print(cosine(encoded, encoded_rotated))
        print(cosine(encoded, encoded_flipped))
        print(cosine(encoded, encoded_flipped_rotated))
        print([cosine(encoded, o) for o in encoded_others])
