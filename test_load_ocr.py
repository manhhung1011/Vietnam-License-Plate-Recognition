import torch
from ocr_model import CRNN


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = CRNN(
    num_classes=37
).to(device)

checkpoint = torch.load(
    "models/ocr_best.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

print("Load OCR thành công")
print("Epoch:", checkpoint["epoch"])
print(
    "Best char accuracy:",
    checkpoint["best_char_acc"]
)