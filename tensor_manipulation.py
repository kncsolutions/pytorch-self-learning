# # Tensor Manipulation Playground
#
# In this section we explore tensor manipulation in PyTorch.
#
# This is one of the MOST important topics in deep learning.
#
# Why?
#
# Because neural networks constantly reshape tensors:
#
# * batching
# * embeddings
# * attention heads
# * image channels
# * latent vectors
# * transformer sequences
#
# Tensor manipulation is therefore foundational to:
#
# * transformers
# * CNNs
# * diffusion models
# * reinforcement learning
# * semantic latent systems

import torch
BREAK_SIZE = 50
def study_manipulation():
    print("=" * BREAK_SIZE)
    print("TENSOR MANIPULATION")
    print("=" * BREAK_SIZE)
    print("Pytorch Version : ", torch.__version__)
    print("=" * BREAK_SIZE)
    a = torch.tensor([
        [1, 2],
        [3, 4]
    ])

    b = torch.tensor([
        [5, 6],
        [7, 8]
    ])


    print(f"Tensor A : {a}")
    print(f"")
    print(f"Tensor B : {b}")
    print("=" * BREAK_SIZE)
    x = torch.cat((a, b), dim=0)
    y = torch.cat((a, b), dim=1)
    print(f"Concatenation by rows :\n {x}")
    print(f"Shape change after concatenation by rows : {x.shape}")
    print(f"Concatenation by columns :\n {y}")
    print(f"Shape change after concatenation by columns : {y.shape}")
    ################################################################
    # Observe the outputs; how the shapes change. Experiment by
    # changing the size of one of
    # the tensors.
    ################################################################
    print("=" * BREAK_SIZE)
    x = torch.stack((a, b), dim=0)
    y = torch.stack((a, b), dim=1)
    print(f"Stack tensors by rows :\n {x}")
    print(f"Shape change after stacking by rows : {x.shape}")
    print(f"Stack tensors by columns :\n {y}")
    print(f"Shape change after stacking by columns : {y.shape}")
    ################################################################
    # Observe the outputs; how the stacking took place but  shapes
    # remained unchanged. Experiment by changing the size of one of
    # the tensors.
    ################################################################
    print("=" * BREAK_SIZE)
    x = torch.arange(10)
    chunk_size = 3
    print(f"Original tensor  :\n{ x}")
    print(f"Split by a chunk size of {chunk_size} :\n {torch.split(x, chunk_size)}")
    x = torch.rand(3,4)
    chunk_size = 2
    print(f"Original tensor  :\n{x}")
    print(f"Split by a chunk size of {chunk_size} :\n {torch.split(x, chunk_size)}")
    ################################################################
    # Experiment by changing the chunk_size
    ################################################################
    print("=" * BREAK_SIZE)
    x = torch.arange(10)
    chunk_numbers = 3
    print(f"Original tensor  :\n{x}")
    print(f"Split by number of  chunks {chunk_numbers} :\n {torch.chunk(x, chunk_numbers)}")
    x = torch.rand(3, 4)
    chunk_numbers = 2
    print(f"Original tensor  :\n{x}")
    print(f"Split by number of  chunks {chunk_numbers} :\n {torch.chunk(x, chunk_numbers)}")
    ################################################################
    # Observe how split() and chunk() behave differently.
    ################################################################
    print("=" * BREAK_SIZE)
    x = torch.arange(12)
    x_view34 = x.view(3,4)
    print(f"Original tensor  :\n{x}")
    print(f"Split by a view {3,4}  :\n {x_view34}")
    print(f"Split updated by a view {4, 3}  :\n {x_view34.view(4,3)}")
    print(f"Split updated by a view {12}  :\n {x_view34.view(12)}")

    print("." * BREAK_SIZE)
    x = torch.arange(12)
    x_reshape34 = x.reshape(3, 4)
    print(f"Original tensor  :\n{x}")
    print(f"Split by a reshape {3, 4}  :\n {x_reshape34}")
    print(f"Split updated by a reshape {4, 3}  :\n {x_reshape34.reshape(4, 3)}")
    print(f"Split updated by a reshape {12}  :\n {x_reshape34.reshape(12)}")

    print("." * BREAK_SIZE)
    x = torch.rand(2, 3, 4)
    print(f"Original tensor  :\n{x}")
    print(f"Shape of original tensor  : {x.shape}")
    x_permuted = x.permute(2, 0, 1)
    print(f"Permutation view of tensor  :\n{x_permuted}")
    print(f"Shape after permutation view : {x_permuted.shape}")

    print("." * BREAK_SIZE)

    x = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(f"Original tensor  :\n{x}")
    print(f"Shape of original tensor  : {x.shape}")
    print(f"Transposed tensor  :\n{ x.transpose(0, 1)}")
    print("." * BREAK_SIZE)

    y = torch.rand(3, 4, 3)
    print(f"Original tensor  :\n{y}")
    print(f"Shape of original tensor  : {y.shape}")
    print(f"Transposed tensor  :\n{y.transpose(0, 1)}")

    print("=" * BREAK_SIZE)
    print("Demonstration of tensor.squeeze()")
    x = torch.rand(1, 3, 1, 4)
    print(f"Original tensor  :\n{x}")
    print(f"Shape of original tensor  : {x.shape}")
    x_squeezed = x.squeeze()
    print(f"After squeeze  :\n{x_squeezed}")
    print(f"Shape after squeeze  : {x_squeezed.shape}")

    print("=" * BREAK_SIZE)
    print("Demonstration of tensor.unsqueeze()")
    x = torch.tensor([1, 2, 3])
    print(f"Original tensor  :\n{x}")
    print(f"Shape of original tensor  : {x.shape}")
    x_unsqueezed = x.unsqueeze(0)
    print(f"After unsqueeze  :\n{x_unsqueezed}")
    print(f"Shape after unsqueeze  : {x_unsqueezed.shape}")

    print("=" * BREAK_SIZE)
    print("Demonstration of tensor.flatten()")
    x = torch.rand((3, 4))
    print(f"Original tensor  :\n{x}")

    x_flattened = x.flatten()
    print(f"Original tensor  :\n{x}")
    print(f"Shape of original tensor  : {x.shape}")
    print(f"After flattening :\n{x_flattened}")
    print(f"Shape after flattening  : {x_flattened.shape}")

    print("=" * BREAK_SIZE)
    print("Demonstration of tensor.repeat()")
    x = torch.tensor([1, 2, 3])
    repeat_by = 2
    x_repeated = x.repeat(repeat_by)
    print(f"Original tensor  :\n{x}")
    print(f"Shape of original tensor  : {x.shape}")
    print(f"After repeat by {repeat_by} :\n{x_repeated}")
    print(f"Shape after repeat  : {x_repeated.shape}")

    print("=" * BREAK_SIZE)
    print("Demonstration of tensor.expand()")

    x = torch.tensor([[1], [2], [3]])
    x_expanded = x.expand( 3, 4)
    print(f"Original tensor  :\n{x}")
    print(f"Shape of original tensor  : {x.shape}")
    print(f"After expansion :\n{x_expanded}")
    print(f"Shape after expansion  : {x_expanded.shape}")


if __name__ == "__main__":
    study_manipulation()
