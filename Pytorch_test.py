import torch

# Check if CUDA (NVIDIA GPU) is available
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can run on the GPU.")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch will run on the CPU.")

# Check if a simple operation runs on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(f"Tensor is on: {x.device}")