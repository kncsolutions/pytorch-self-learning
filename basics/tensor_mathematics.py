# Mathematical Operations Playground
#
# In this notebook we explore mathematical operations in PyTorch.
#
# These operations form the mathematical foundation of:
#
# deep learning
# transformers
# optimization
# embeddings
# latent spaces
# scientific computation
import torch

BREAK_SIZE = 60


def mathematical_operations():

    print("=" * BREAK_SIZE)
    print("PYTORCH MATHEMATICAL OPERATIONS PLAYGROUND")
    print("=" * BREAK_SIZE)

    print(f"PyTorch Version : {torch.__version__}")

    print("=" * BREAK_SIZE)

    ############################################################
    # BASE TENSORS
    ############################################################

    x = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    y = torch.tensor([
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    print("Tensor x :")
    print(x)

    print("\nTensor y :")
    print(y)

    ############################################################
    # torch.add()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.add()")

    z1 = torch.add(x, y)

    print(z1)

    ############################################################
    # What happened here?
    # Was the addition element-wise?
    ############################################################

    ############################################################
    # torch.sub()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.sub()")

    z2 = torch.sub(x, y)

    print(z2)

    ############################################################
    # Observe:
    # Which tensor is subtracted from which?
    ############################################################

    ############################################################
    # torch.mul()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.mul()")

    z3 = torch.mul(x, y)

    print(z3)

    ############################################################
    # Important:
    # Is this matrix multiplication?
    # Or element-wise multiplication?
    ############################################################

    ############################################################
    # torch.div()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.div()")

    z4 = torch.div(y, x)

    print(z4)

    ############################################################
    # Observe:
    # Division occurred element by element.
    ############################################################

    ############################################################
    # torch.matmul()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.matmul()")

    z5 = torch.matmul(x, y)

    print(z5)

    ############################################################
    # Important:
    # This is REAL matrix multiplication.
    ############################################################

    ############################################################
    # torch.mm()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.mm()")

    z6 = torch.mm(x, y)

    print(z6)

    ############################################################
    # Question:
    # Compare torch.matmul() vs torch.mm()
    ############################################################

    ############################################################
    # torch.bmm()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.bmm()")

    a = torch.rand(2, 2, 3)

    b = torch.rand(2, 3, 2)

    print("Tensor a shape :", a.shape)
    print("Tensor b shape :", b.shape)

    z7 = torch.bmm(a, b)

    print("Result shape :", z7.shape)

    print(z7)

    ############################################################
    # Important:
    # bmm() = batch matrix multiplication
    ############################################################

    ############################################################
    # torch.exp()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.exp()")

    z8 = torch.exp(x)

    print(z8)

    ############################################################
    # Observe:
    # e^x computed element-wise.
    ############################################################

    ############################################################
    # torch.log()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.log()")

    z9 = torch.log(x)

    print(z9)

    ############################################################
    # Observe:
    # Natural logarithm applied element-wise.
    ############################################################

    ############################################################
    # torch.sqrt()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.sqrt()")

    z10 = torch.sqrt(x)

    print(z10)

    ############################################################
    # torch.pow()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.pow()")

    z11 = torch.pow(x, 2)

    print(z11)

    ############################################################
    # Observe:
    # Each element squared.
    ############################################################

    ############################################################
    # torch.abs()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.abs()")

    neg_tensor = torch.tensor([
        [-1.0, -2.0],
        [3.0, -4.0]
    ])

    print("Original tensor :")
    print(neg_tensor)

    z12 = torch.abs(neg_tensor)

    print("Absolute tensor :")
    print(z12)

    ############################################################
    # torch.sin()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.sin()")

    z13 = torch.sin(x)

    print(z13)

    ############################################################
    # torch.cos()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.cos()")

    z14 = torch.cos(x)

    print(z14)

    ############################################################
    # torch.tanh()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.tanh()")

    z15 = torch.tanh(x)

    print(z15)

    ############################################################
    # Important:
    # tanh() is a very important activation function.
    ############################################################

    ############################################################
    # torch.relu()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.relu()")

    relu_input = torch.tensor([
        [-2.0, -1.0],
        [3.0, 4.0]
    ])

    print("Input tensor :")
    print(relu_input)

    z16 = torch.relu(relu_input)

    print("ReLU output :")
    print(z16)

    ############################################################
    # Observe:
    # Negative values become zero.
    ############################################################

    ############################################################
    # torch.sigmoid()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.sigmoid()")

    z17 = torch.sigmoid(x)

    print(z17)

    ############################################################
    # Important:
    # Sigmoid compresses values between 0 and 1.
    ############################################################

    print("\n" + "=" * BREAK_SIZE)
    print("END OF MATHEMATICAL OPERATIONS PLAYGROUND")
    print("=" * BREAK_SIZE)


if __name__ == "__main__":

    mathematical_operations()