import dataclasses as dcls
import functools
from types import MappingProxyType
from typing import List, NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

YOLO_CLASSES = [
    "background",
    "rice",
    "eels on rice",
    "pilaf",
    "chicken-n-egg on rice",
    "pork cutlet on rice",
    "beef curry",
    "sushi",
    "chicken rice",
    "fried rice",
    "tempura bowl",
    "bibimbap",
    "toast",
    "croissant",
    "roll bread",
    "raisin bread",
    "chip butty",
    "hamburger",
    "pizza",
    "sandwiches",
    "udon noodle",
    "tempura udon",
    "soba noodle",
    "ramen noodle",
    "beef noodle",
    "tensin noodle",
    "fried noodle",
    "spaghetti",
    "Japanese-style pancake",
    "takoyaki",
    "gratin",
    "sauteed vegetables",
    "croquette",
    "grilled eggplant",
    "sauteed spinach",
    "vegetable tempura",
    "miso soup",
    "potage",
    "sausage",
    "oden",
    "omelet",
    "ganmodoki",
    "jiaozi",
    "stew",
    "teriyaki grilled fish",
    "fried fish",
    "grilled salmon",
    "salmon meuniere",
    "sashimi",
    "grilled pacific saury",
    "sukiyaki",
    "sweet and sour pork",
    "lightly roasted fish",
    "steamed egg hotchpotch",
    "tempura",
    "fried chicken",
    "sirloin cutlet",
    "nanbanzuke",
    "boiled fish",
    "seasoned beef with potatoes",
    "hambarg steak",
    "beef steak",
    "dried fish",
    "ginger pork saute",
    "spicy chili-flavored tofu",
    "yakitori",
    "cabbage roll",
    "rolled omelet",
    "egg sunny-side up",
    "fermented soybeans",
    "cold tofu",
    "egg roll",
    "chilled noodle",
    "stir-fried beef and peppers",
    "simmered pork",
    "boiled chicken and vegetables",
    "sashimi bowl",
    "sushi bowl",
    "fish-shaped pancake with bean jam",
    "shrimp with chill source",
    "roast chicken",
    "steamed meat dumpling",
    "omelet with fried rice",
    "cutlet curry",
    "spaghetti meat sauce",
    "fried shrimp",
    "potato salad",
    "green salad",
    "macaroni salad",
    "Japanese tofu and vegetable chowder",
    "pork miso soup",
    "chinese soup",
    "beef bowl",
    "kinpira-style sauteed burdock",
    "rice ball",
    "pizza toast",
    "dipping noodles",
    "hot dog",
    "french fries",
    "mixed rice",
    "goya chanpuru",
    "others",
    "beverage",
]

__YOLO_CLS_SET = frozenset(YOLO_CLASSES)
__YOLO_CLS_TO_IDX = MappingProxyType({x: i for i, x in enumerate(YOLO_CLASSES)})


def one_hot_by_ingredients(ingredients: List[str]) -> NDArray:
    assert all((ing in __YOLO_CLS_SET) for ing in ingredients)
    array = np.zeros([len(YOLO_CLASSES)])
    ing_idx = [__YOLO_CLS_TO_IDX[ing] for ing in ingredients]
    array[ing_idx] = 1
    return array


class Mask(NamedTuple):
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
class MyImage:
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
    def from_messy(cls, name: str, image: NDArray, results: List[Mask]):
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
class Menu:
    items: List[MyImage]

    @property
    def _items_hamming_matrix(self):
        return np.array([img.one_hot for img in self.items])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def match_by_ingredient(self, image: MyImage, top_k: int = -1) -> List[MyImage]:
        hamming = self._items_hamming_matrix
        img_hamming = image.one_hot

        # Dot product on the second dimension
        matching_matrix = np.einsum("ij,i->j", hamming, img_hamming)

        if top_k < 0:
            top_k = len(matching_matrix)

        argsort = np.argsort(matching_matrix)[::-1]
        return [self.items[idx] for idx in argsort[:top_k]]

    def match_by_orb(
        self, image: MyImage, top_k: int = -1, top_matches: int = 100
    ) -> List[MyImage]:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
    def from_items(cls, items: List[MyImage]):
        return cls(items)

    @functools.cached_property
    def orb(self):
        return cv2.ORB_create()

    def detect(self, image: MyImage):
        _, desc = self.orb.detectAndCompute(image.argmax_seg_mask, None)
        return desc

    @functools.cache
    def items_orb(self):
        return [self.detect(img) for img in self.items]
