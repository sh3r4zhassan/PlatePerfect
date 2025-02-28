from datetime import datetime
import onnxruntime as ort
from Menu import menu_dictionary, ingredients
from Functions import *


def main():
    num_classes = len(ingredients)
    yolo_classes = ingredients

    model = ort.InferenceSession(model_path)

    while True:
        image_path = input("Enter the image path (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            break

        try:
            img = Image.open(image_path)
            img2 = img.copy()

            img_width, img_height = img.size
            img = img.convert("RGB")
            img = img.resize((imagesize, imagesize))
            input_data = np.array(img)
            input_data = input_data.transpose(2, 0, 1)
            input_data = input_data.reshape(1, 3, imagesize, imagesize).astype('float32')
            input_data /= 255.0

            outputs = model.run(None, {"images": input_data})
            output0 = outputs[0][0].transpose()
            boxes = output0[:, 0:4 + num_classes]
            masks = output0[:, 4 + num_classes:]
            output1 = outputs[1][0].reshape(32, 160 * 160)
            masks = np.dot(masks, output1)
            boxes = np.hstack((boxes, masks))

            objects = []
            for row in boxes:
                prob = row[4:num_classes + 4].max()
                if prob < 0.35:
                    continue
                xc, yc, w, h = row[:4]
                class_id = row[4:num_classes].argmax()
                x1 = (xc - w / 2) / 640 * img_width
                y1 = (yc - h / 2) / 640 * img_height
                x2 = (xc + w / 2) / 640 * img_width
                y2 = (yc + h / 2) / 640 * img_height
                label = yolo_classes[class_id]
                mask = get_mask(row[4 + num_classes:boxes.shape[1]], (x1, y1, x2, y2), img_width, img_height)
                polygon = get_polygon(mask)
                objects.append([x1, y1, x2, y2, label, prob, mask, polygon])

            objects.sort(key=lambda x: x[5], reverse=True)
            result = []
            while len(objects) > 0:
                result.append(objects[0])
                objects = [object for object in objects if iou(object, objects[0]) < 0.65]

            print("Total ingredients:", len(result))

            draw = ImageDraw.Draw(img2, "RGBA")

            for obj in result:
                [x1, y1, x2, y2, label, prob, mask, polygon] = obj
                polygon = [(int(x1 + point[0]), int(y1 + point[1])) for point in polygon]
                draw.polygon(polygon, fill=(0, 255, 0, 125))
                draw.rectangle((x1, y1, x2, y2), None, "#00ff00")
                draw.text((x1, y1), "%.2f%% %s" % (prob, label))

            predicted_ingredients = labels(result)
            best_matches, best_confidences = find_best_matching_dishes(predicted_ingredients, menu_dictionary)
            threshold = 0.4

            timestamp = datetime.now().strftime("%y%m%d%H%M%S")

            output_list = []

            if best_confidences[0] >= threshold:
                output_list.append(f"{best_matches[0]} with a Confidence of {best_confidences[0]:.2%}.")
                output_list.append(f"{best_matches[1]} with a Confidence of {best_confidences[1]:.2%}.")
                img2.save(f"output/{timestamp}_{best_matches[0]}.png")
            else:
                output_list.append("No dish identified.")
                output_list.append(f"The closest dishes are {best_matches[0]} with a Confidence of {best_confidences[0]:.2%} and {best_matches[1]} with a Confidence of {best_confidences[1]:.2%}.")
                img2.save(f"output/{timestamp}_Unidentified.png")

            print(output_list)
            

            img2.show()
            # return output_list

        except Exception as e:
            print(f"Error processing the image: {str(e)}")

if __name__ == "__main__":
    main()
