import torch
# Tensor Information Playground
#
# In this section we explore how to inspect tensors in PyTorch.
#
# This is extremely important because in deep learning:
#
# shape mismatches
# datatype mismatches
# device mismatches
#
# are among the most common errors.
#
# Learning tensor inspection properly is a major milestone.
BREAK_SIZE = 50
def study_information():
    print("=" * BREAK_SIZE)
    print("TENSOR INFORMATION")
    print("=" * BREAK_SIZE)
    print("Pytorch Version : ", torch.__version__)
    print("=" * BREAK_SIZE)
    x = torch.tensor([ [1, 2, 3], [4, 5, 6] ])
    print(f"The tensor x : {x}")
    print(f"The shape of tensor x : {x.shape}")
    print(f"The size of tensor x : {x.size()}")
    print(f"The datatype for tensor x : {x.dtype}")
    print(f"The device being used to process x : {x.device}")
    print(f"The number of dimensions of  x : {x.ndim}")
    print(f"The number of elements in  x : {x.numel()}")
    print(f"The gradient tracking status of  x : {x.requires_grad}")
    ########################################################################
    # Look at the output for each function carefully
    ########################################################################
    print("." * BREAK_SIZE)
    y = torch.rand(size = [4,4,3])
    print(f"The tensor y : {y}")
    print(f"The shape of tensor y : {y.shape}")
    print(f"The size of tensor x : {y.size()}")
    print(f"The number of dimensions of  y : {y.ndim}")
    print(f"The number of elements in  y : {y.numel()}")
    print("." * BREAK_SIZE)
    print(f"The gradient tracking status of  y : {y.requires_grad}")
    y.requires_grad_(True)
    print(f"After modification the gradient tracking status of  y : {y.requires_grad}")
    ########################################################################
    # As an experiment try using different values in size list in the
    # argument. Try changing the length of the list as well.
    # What changes you observe.
    ########################################################################
    print("." * BREAK_SIZE)
    z = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad = True, dtype = torch.float64)
    print(f"The gradient tracking status of  z : {z.requires_grad}")
    ########################################################################
    # What happens when you remove  dtype = torch.float64 from the argument?
    # Can you explain why?
    ########################################################################
if __name__ == "__main__":
    study_information()