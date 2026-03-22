import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from torchvision import models
from torchvision import transforms
from PIL import Image
import time

yolov5_weight_file = 'rider_helmet_number_medium.pt'  # Full path may be required
helmet_classifier_weight = 'helment_no_helmet98.6.pth'  # Full path may be required
conf_set = 0.35
frame_size = (800, 480)
head_classification_threshold = 3.0  # Adjust if needed to detect non-helmet more aggressively

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolov5_weight_file, map_location=device)
cudnn.benchmark = True
names = model.module.names if hasattr(model, 'module') else model.names

# Load image classification model
model2 = torch.load(helmet_classifier_weight, map_location=device)
model2.eval()

transform = transforms.Compose([
    transforms.Resize(144),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function for image classification
def img_classify(frame):
    if frame.shape[0] < 46:  # Skip small head sizes (adjustable)
        return [None, 0]

    frame = transform(Image.fromarray(frame))
    frame = frame.unsqueeze(0)
    prediction = model2(frame)
    result_idx = torch.argmax(prediction).item()
    prediction_conf = sorted(prediction[0])

    cs = (prediction_conf[-1] - prediction_conf[-2]).item()  # Confidence score
    if cs > head_classification_threshold:
        return [True, cs] if result_idx == 0 else [False, cs]
    else:
        return [None, cs]


# Function for object detection
def object_detection(frame):
    img = torch.from_numpy(frame)
    img = img.permute(2, 0, 1).float().to(device)
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_set, 0.30)  # Prediction, confidence, IoU

    detection_result = []
    for i, det in enumerate(pred):
        if len(det):
            for d in det:
                x1 = int(d[0].item())
                y1 = int(d[1].item())
                x2 = int(d[2].item())
                y2 = int(d[3].item())
                conf = round(d[4].item(), 2)
                c = int(d[5].item())

                detected_name = names[c]

                print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}')
                detection_result.append([x1, y1, x2, y2, conf, c])

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Box
                if c != 1:  # If it is not a head bounding box
                    frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 1, cv2.LINE_AA)

    return frame, detection_result


# Function to check if a smaller box is inside a larger box
def inside_box(big_box, small_box):
    x1 = small_box[0] - big_box[0]
    y1 = small_box[1] - big_box[1]
    x2 = big_box[2] - small_box[2]
    y2 = big_box[3] - small_box[3]
    return not bool(min([x1, y1, x2, y2, 0]))
