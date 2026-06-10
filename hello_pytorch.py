import torch

def hello_pytorch():
    print("="*30)
    print("Hello PyTorch")
    print("="*30)
    print("Pytorch Version : ", torch.__version__)
    print("Simple basic operations with pytorch: ")
    x = torch.tensor([1, 2, 3])
    print("Your first tensor : ", x)
    y = torch.tensor([4, 5, 6])
    print("Your second tensor : ", y)
    z = x + y
    print("Your third tensor after summing x and y : ", z)

    print("=" * 30)
    print("Let's see some features of x :")
    print("=" * 30)

    print("Type of x : ", type(x))
    print("Data type of x : ", x.dtype)
    print("Shape of x : ", x.shape)
    print("Device of x : ", x.device)

if __name__ == "__main__":
     hello_pytorch()