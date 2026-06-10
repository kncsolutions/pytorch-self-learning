import torch
BREAK_SIZE = 50
def create_tensors():
    print("=" * BREAK_SIZE )
    print("TENSOR CREATION")
    print("=" * BREAK_SIZE )
    print("Pytorch Version : ", torch.__version__)
    print("=" * BREAK_SIZE )
    print(f"method torch.tensor() :")
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"The tensor x : {x}")
    print(f"The shape of tensor x : {x.shape}")
    print(f"The datatype for tensor x : {x.dtype}")
    print(f"method torch.Tensor() :")
    y = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"The tensor y : {y}")
    print(f"The shape of tensor y : {y.shape}")
    print(f"The datatype for tensor y : {y.dtype}")

    ##############################################################
    # What difference you observe between x and y ?
    # Jot down carefully.
    ##############################################################
    print("#" * BREAK_SIZE )
    z1 = torch.zeros(3, 4)
    print(f"method torch.zeros() :")
    print(f"The tensor z1 : {z1}")
    print(f"The shape of tensor z1 : {z1.shape}")
    print(f"The datatype for tensor z1 : {z1.dtype}")

    ##############################################################
    # Can you explain what torch.zeros(a,b) did?
    # Write your answer here?
    ##############################################################
    print("#" * BREAK_SIZE )
    print(f"method torch.ones() :")
    z2 = torch.ones(3, 4)
    print(f"The tensor z2 : {z2}")
    print(f"The shape of tensor z2 : {z2.shape}")
    print(f"The datatype for tensor z2 : {z2.dtype}")

    ##############################################################
    # How z2 is different from z1?
    # Write your answer here?
    ##############################################################
    print("#" * BREAK_SIZE )
    print(f"method torch.empty() :")
    z3 =  torch.empty(7, 7)
    print(f"The tensor z3 : {z3}")
    print(f"The shape of tensor z3 : {z3.shape}")
    print(f"The datatype for tensor z3 : {z3.dtype}")
    ##############################################################
    # What peculiarity you observe when you call torch.empty(a,b)?
    # Can you explain the behavior?(Hint: Look at the values)
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.rand() :")
    z4 = torch.rand(3, 4)
    print(f"The tensor z4 : {z4}")
    print(f"The shape of tensor z4 : {z4.shape}")
    print(f"The datatype for tensor z4 : {z4.dtype}")
    ##############################################################
    # Jot down the properties of z4.
    # You will need it for the next question.
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.randint() :")
    z5 = torch.randint(0, 10, (3, 4))
    print(f"The tensor z5 : {z5}")
    print(f"The shape of tensor z5 : {z5.shape}")
    print(f"The datatype for tensor z5 : {z5.dtype}")
    ##############################################################
    # How z5 is different from z4.
    # Can you guess when you will use torch.rand()? When you can
    # use torch.randint()?
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.arange() :")
    z6 = torch.arange(2, 11)
    print(f"The tensor z6 : {z6}")
    print(f"The shape of tensor z6 : {z6.shape}")
    print(f"The datatype for tensor z6 : {z6.dtype}")
    ##############################################################
    # Can you guess what is the function of torch.arange()
    # function?(be cautious with the spelling. It is not arrange.)
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.linspace() :")
    z7 = torch.linspace(0.1, 1.0, steps = 5 )
    print(f"The tensor z7 : {z7}")
    print(f"The shape of tensor z7 : {z7.shape}")
    print(f"The datatype for tensor z7 : {z7.dtype}")
    ##############################################################
    # So torch.linspace() basically generated a set of numbers
    # equal to steps argument with both the numbers specified
    # in the first two arguments included. Can you guess how
    # the intermediary numbers have been generated?
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.logspace() :")
    z8 = torch.logspace(0.1, 1.0, steps=5)
    print(f"The tensor z8 : {z8}")
    print(f"The shape of tensor z8 : {z8.shape}")
    print(f"The datatype for tensor z8 : {z8.dtype}")
    ##############################################################
    # If you understood z7 then you can easily guess z8.
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.eye() :")
    z9 = torch.eye(4)
    print(f"The tensor z9 : {z9}")
    print(f"The shape of tensor z9 : {z9.shape}")
    print(f"The datatype for tensor z9 : {z9.dtype}")
    ##############################################################
    # Can you name the matrix z9 is holding?
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.full() :")
    z10 = torch.full((3, 3),4)
    print(f"The tensor z10 : {z10}")
    print(f"The shape of tensor z10 : {z10.shape}")
    print(f"The datatype for tensor z10 : {z10.dtype}")
    ##############################################################
    # So guess what torch.full() is doing?
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.zeros_like() :")
    tmp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z11 = torch.zeros_like(tmp)
    print(f"The tensor z11 : {z11}")
    print(f"The shape of tensor z11 : {z11.shape}")
    print(f"The datatype for tensor z11 : {z11.dtype}")
    ##############################################################
    # Again do a simple guess what's going on in
    # torch.zeros_like() ?
    ##############################################################
    print("#" * BREAK_SIZE)
    print(f"method torch.ones_like() :")
    tmp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z12 = torch.ones_like(tmp)
    print(f"The tensor z12 : {z12}")
    print(f"The shape of tensor z12 : {z12.shape}")
    print(f"The datatype for tensor z12 : {z12.dtype}")
    print("." * BREAK_SIZE)
    print(f"method torch.rand_like() :")
    tmp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z13 = torch.rand_like(tmp.float())
    print(f"The tensor z13 : {z13}")
    print(f"The shape of tensor z13 : {z13.shape}")
    print(f"The datatype for tensor z13 : {z13.dtype}")
    print("." * BREAK_SIZE)
    print(f"method torch.randn_like() :")
    tmp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z14 = torch.randn_like(tmp.float())
    print(f"The tensor z14 : {z14}")
    print(f"The shape of tensor z14 : {z14.shape}")
    print(f"The datatype for tensor z14 : {z14.dtype}")
    ##############################################################
    # Compare Z12, Z13, Z14
    ##############################################################
    print("." * BREAK_SIZE)
    print(f"method torch.randn() :")
    z15 = torch.randn(2, 3)
    print(f"The tensor z15 : {z15}")
    print(f"The shape of tensor z15 : {z15.shape}")
    print(f"The datatype for tensor z15 : {z15.dtype}")
    ##############################################################
    # Guess what is randn
    ##############################################################


if __name__ == "__main__":
    create_tensors()