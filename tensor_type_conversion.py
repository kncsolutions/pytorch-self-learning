# In this section we explore tensor type conversion in PyTorch.
#
# This is extremely important because:
#
# neural networks usually expect floating-point tensors
# indexing often requires integer tensors
# GPU execution needs device conversion
# datatype mismatches are common sources of bugs
#
# Tensor conversion is therefore one of the most frequently used operations in PyTorch.
from logging import exception

import torch

def type_conversion():
    x = torch.tensor([1, 2, 3])
    print("Original :", x.dtype)
    print("Float :", x.float().dtype)
    print("Double :", x.double().dtype)
    print("Int :", x.int().dtype)
    print("Long :", x.long().dtype)
    print("Bool :", x.bool().dtype)

    x_to = x.to(torch.float64)
    print("After using tensor.to() method the datatype is :", x_to.dtype)
    try:
        x_to = x.to("cuda")
        print("After using tensor.to() method to change device type the device is :", x_to.device)
    except Exception:
        print("x"*50)
        print("Torch not compiled with CUDA enabled or GPU not available")
        print("x" * 50)
        pass

    if torch.cuda.is_available():

        gpu_tensor = x.cuda()

        print("GPU tensor device : ", gpu_tensor.device)

    else:

        print("CUDA not available")

    ############################################################
    # Study the code and output side by side to draw conclusions
    ############################################################




if __name__ == "__main__":
    type_conversion()