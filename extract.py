import torch

model_path = "path_to_pytorch_model.bin"
state_dict = torch.load(model_path, map_location="cpu")
align_state_dict = {}
for k, v in state_dict.items():
    if "align_stages" in k:                   
        print(f"Found align parameter: {k}")
        align_state_dict[k] = v

save_path = "path_to_save"
torch.save(align_state_dict, save_path)

