import torch


def test_gpu():
    """ Gives user information about GPU """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if DEVICE == "cuda":
        print(f"Index of selected device: {torch.cuda.current_device()}")
        print(f"Name of selected device: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1000, 1000, device=DEVICE)  # Creates a random 1000x1000 matrix
    y = x @ x                                   # matrix multiplication
    print("Result lives on:", y.device)

if __name__ == "__main__":
    test_gpu()