import cv2
import torch
import numpy as np
from collections import Counter, deque
from ultralytics import YOLO
from ocr_model import CRNN


YOLO_PATH = "models/yolo_best.pt"
OCR_PATH = "models/ocr_best.pt"

IMG_WIDTH = 256
IMG_HEIGHT = 64

CHARACTERS = list(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

IDX2CHAR = {
    i + 1: char
    for i, char in enumerate(CHARACTERS)
}

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

plate_detector = YOLO(YOLO_PATH)

ocr_model = CRNN(
    num_classes=37
).to(device)

checkpoint = torch.load(
    OCR_PATH,
    map_location=device,
    weights_only=False
)

ocr_model.load_state_dict(
    checkpoint["model_state"]
)

ocr_model.eval()

print("YOLO loaded")
print("OCR loaded")
print("OCR epoch:", checkpoint["epoch"])
print(
    "OCR accuracy:",
    checkpoint["best_char_acc"]
)


def preprocess_plate(image):
    if image is None or image.size == 0:
        return None, None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (IMG_WIDTH, IMG_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    debug = gray.copy()

    image = gray.astype(
        np.float32
    )

    image /= 255.0

    image = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    image = (image - 0.5) / 0.5

    return image.to(device), debug


def decode_prediction(output):
    output = output.permute(
        1,
        0,
        2
    )

    predictions = output.argmax(
        2
    )

    result = []

    for prediction in predictions:
        text = []
        previous = 0

        for p in prediction.cpu().numpy():
            p = int(p)

            if (
                p != previous
                and p != 0
                and p in IDX2CHAR
            ):
                text.append(
                    IDX2CHAR[p]
                )

            previous = p

        result.append(
            "".join(text)
        )

    if not result:
        return ""

    return result[0]


def recognize_plate(image):
    tensor, debug = preprocess_plate(
        image
    )

    if tensor is None:
        return "", None

    with torch.no_grad():
        output = ocr_model(
            tensor
        )

    text = decode_prediction(
        output
    )

    return text, debug


def format_plate(text):
    text = "".join(
        char
        for char in text
        if char.isalnum()
    )

    if len(text) == 9:
        return (
            text[:4]
            + "-"
            + text[4:7]
            + "."
            + text[7:]
        )

    return text


history = deque(
    maxlen=7
)


def stabilize_text(text):
    if not text:
        return ""

    history.append(text)

    values = Counter(
        history
    )

    return values.most_common(
        1
    )[0][0]


cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    raise RuntimeError(
        "Không mở được camera"
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

print("Camera opened")
print("Nhấn Q để thoát")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = plate_detector.predict(
        source=frame,
        conf=0.4,
        imgsz=640,
        verbose=False
    )

    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        confidence = float(
            box.conf[0]
        )

        frame_h, frame_w = frame.shape[:2]

        x1 = max(
            0,
            min(x1, frame_w - 1)
        )

        y1 = max(
            0,
            min(y1, frame_h - 1)
        )

        x2 = max(
            0,
            min(x2, frame_w)
        )

        y2 = max(
            0,
            min(y2, frame_h)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        plate_crop = frame[
            y1:y2,
            x1:x2
        ]

        if plate_crop.size == 0:
            continue

        text, ocr_input = recognize_plate(
            plate_crop
        )

        stable_text = stabilize_text(
            text
        )

        display_text = format_plate(
            stable_text
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if display_text:
            label = (
                f"{display_text} | "
                f"{confidence:.2f}"
            )
        else:
            label = (
                f"Plate | "
                f"{confidence:.2f}"
            )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    30,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Plate Crop",
            plate_crop
        )

        if ocr_input is not None:
            enlarged = cv2.resize(
                ocr_input,
                (
                    IMG_WIDTH * 2,
                    IMG_HEIGHT * 2
                ),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow(
                "OCR Input",
                enlarged
            )

    cv2.imshow(
        "License Plate Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()