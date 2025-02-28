import os
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image, ImageDraw

import rc

imagesize = 640
model_path = "runs/segment/train3/weights/best.onnx"


def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_polygon(mask):
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[contour[0][0], contour[0][1]] for contour in contours[0][0]]
    return polygon


def get_mask(row, box, img_width, img_height):
    mask = row.reshape(160, 160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype("uint8") * 255
    x1, y1, x2, y2 = box
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    img_mask = Image.fromarray(mask, "L")
    img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
    mask = np.array(img_mask)
    return mask


def ingredient_similarity(ingredients1, ingredients2):
    # Sort the ingredients to ensure arrangement does not affect the score
    ingredients1 = sorted(ingredients1)
    ingredients2 = sorted(ingredients2)
    return SequenceMatcher(None, ingredients1, ingredients2).ratio()


def find_best_matching_dishes(predicted_ingredients, menu_dict):
    best_matches = [None, None]
    best_confidences = [0.0, 0.0]

    for dish, ingredients in menu_dict.items():
        confidence = ingredient_similarity(predicted_ingredients, ingredients)

        if confidence > best_confidences[0]:
            best_confidences = [confidence, best_confidences[0]]
            best_matches[0] = dish
        elif confidence > best_confidences[1]:
            best_confidences[1] = confidence
            best_matches[1] = dish

    return best_matches, best_confidences


def labels(results):
    predicted_ingredients = []
    for result in results:
        predicted_ingredients.append(result[4])
    return predicted_ingredients
