import torch

checkpoint = torch.load(
    "models/ocr_best.pt",
    map_location="cpu",
    weights_only=False
)

state_dict = checkpoint["model_state"]

print("===== 10 LAYER CUỐI =====")

for i, (name, tensor) in enumerate(state_dict.items()):
    if i >= 50:
        print(
            f"{i:02d}. {name:40s} {tuple(tensor.shape)}"
        )