import os

def calculate_iou(box1, box2):
    # Convert YOLO format to (x_min, y_min, x_max, y_max)
    box1 = convert_yolo_to_coordinates(box1)
    box2 = convert_yolo_to_coordinates(box2)

    # Calculate the intersection coordinates
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate the intersection area
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou

def convert_yolo_to_coordinates(box):
    x, y, w, h = map(float, box)
    x_min = (x - w / 2)
    y_min = (y - h / 2)
    x_max = (x + w / 2)
    y_max = (y + h / 2)
    return x_min, y_min, x_max, y_max

def calculate_f1_score(ground_truth_folder, predicted_file, iou_threshold):
    # Read the predicted bounding box coordinates from the file
    with open(predicted_file, 'r') as f:
        predicted_boxes = [line.strip().split() for line in f.readlines()]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_box in predicted_boxes:
        max_iou = 0
        for file in os.listdir(ground_truth_folder):
            gt_file = os.path.join(ground_truth_folder, file)
            with open(gt_file, 'r') as f:
                ground_truth_boxes = [line.strip().split() for line in f.readlines()]
            for gt_box in ground_truth_boxes:
                if len(gt_box) == 5:
                    iou = calculate_iou(pred_box[1:], gt_box[1:])
                    if iou > max_iou:
                        max_iou = iou
        if max_iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(os.listdir(ground_truth_folder)) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


ground_truth_folder = 'labels/labelled_frames_Video_Francisco_Heshiki'
predicted_file = 'bounding_boxes.txt'
iou_threshold = 0.5
output_file = 'f1_score_result.txt'

f1_score = calculate_f1_score(ground_truth_folder, predicted_file, iou_threshold)

with open(output_file, 'w') as f:
    f.write(f"F1 Score: {f1_score}\n")

print(f"F1 Score saved in {output_file}")

