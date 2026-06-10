# Why Reduction Operations Matter In Deep Learning
#
# These operations are foundational for:
#
# loss functions
# pooling layers
# normalization
# attention scoring
# probability systems
# latent state summarization
# statistical feature extraction
#
# Without reduction operations:
#
# neural networks
# transformers
# optimization systems
#
# would not function properly.
import torch

BREAK_SIZE = 60


def reduction_operations():

    print("=" * BREAK_SIZE)
    print("PYTORCH REDUCTION OPERATIONS PLAYGROUND")
    print("=" * BREAK_SIZE)

    print(f"PyTorch Version : {torch.__version__}")

    print("=" * BREAK_SIZE)

    ############################################################
    # BASE TENSOR
    ############################################################

    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    print("Base Tensor x :")
    print(x)

    print(f"\nShape of x : {x.shape}")

    ############################################################
    # torch.sum()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.sum()")

    z1 = torch.sum(x)

    print("Sum of all elements :")
    print(z1)

    ############################################################
    # Mathematical Meaning:
    #
    # sum = 1 + 2 + 3 + 4 + 5 + 6
    #
    ############################################################

    print("\nRow-wise Sum :")
    print(torch.sum(x, dim=1))

    print("\nColumn-wise Sum :")
    print(torch.sum(x, dim=0))

    ############################################################
    # Observe carefully:
    # dim=0 → columns collapse
    # dim=1 → rows collapse
    ############################################################

    ############################################################
    # torch.mean()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.mean()")

    z2 = torch.mean(x)

    print("Mean of all elements :")
    print(z2)

    ############################################################
    # Mathematical Formula:
    #
    # mean = (sum of elements) / total elements
    #
    ############################################################

    print("\nRow-wise Mean :")
    print(torch.mean(x, dim=1))

    print("\nColumn-wise Mean :")
    print(torch.mean(x, dim=0))

    ############################################################
    # torch.std()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.std()")

    z3 = torch.std(x)

    print("Standard Deviation :")
    print(z3)

    ############################################################
    # Standard deviation measures:
    #
    # how spread out values are.
    #
    ############################################################

    ############################################################
    # torch.var()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.var()")

    z4 = torch.var(x)

    print("Variance :")
    print(z4)

    ############################################################
    # Relationship:
    #
    # variance = (std)^2
    #
    ############################################################

    ############################################################
    # torch.max()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.max()")

    z5 = torch.max(x)

    print("Maximum value :")
    print(z5)

    ############################################################
    # Row-wise Maximum
    ############################################################

    print("\nRow-wise Maximum :")

    row_max = torch.max(x, dim=1)

    print(row_max)

    ############################################################
    # Important Observation:
    #
    # torch.max(..., dim=1)
    # returns:
    #
    # values
    # indices
    #
    ############################################################

    ############################################################
    # torch.min()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.min()")

    z6 = torch.min(x)

    print("Minimum value :")
    print(z6)

    ############################################################
    # Column-wise Minimum
    ############################################################

    print("\nColumn-wise Minimum :")

    col_min = torch.min(x, dim=0)

    print(col_min)

    ############################################################
    # torch.argmax()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.argmax()")

    z7 = torch.argmax(x)

    print("Index of maximum value :")
    print(z7)

    ############################################################
    # Important:
    #
    # Flattened index is returned.
    #
    ############################################################

    ############################################################
    # Row-wise Argmax
    ############################################################

    print("\nRow-wise Argmax :")

    print(torch.argmax(x, dim=1))

    ############################################################
    # torch.argmin()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.argmin()")

    z8 = torch.argmin(x)

    print("Index of minimum value :")
    print(z8)

    ############################################################
    # Column-wise Argmin
    ############################################################

    print("\nColumn-wise Argmin :")

    print(torch.argmin(x, dim=0))

    ############################################################
    # torch.prod()
    ############################################################

    print("\n" + "#" * BREAK_SIZE)
    print("method torch.prod()")

    z9 = torch.prod(x)

    print("Product of all elements :")
    print(z9)

    ############################################################
    # Mathematical Meaning:
    #
    # 1 × 2 × 3 × 4 × 5 × 6
    #
    ############################################################

    ############################################################
    # Row-wise Product
    ############################################################

    print("\nRow-wise Product :")

    print(torch.prod(x, dim=1))

    ############################################################
    # Column-wise Product
    ############################################################

    print("\nColumn-wise Product :")

    print(torch.prod(x, dim=0))

    ############################################################
    # FINAL SUMMARY
    ############################################################

    print("\n" + "=" * BREAK_SIZE)
    print("REDUCTION OPERATIONS SUMMARY")
    print("=" * BREAK_SIZE)

    print("""
Reduction operations reduce tensor dimensions by
aggregating values.

Examples:
- sum      → total accumulation
- mean     → average
- std      → spread measurement
- var      → variance
- max/min  → extreme values
- argmax   → position of largest value
- argmin   → position of smallest value
- prod     → multiplicative aggregation
""")

    ############################################################
    # Important Deep Learning Insight
    ############################################################

    print("=" * BREAK_SIZE)

    print("""
Reduction operations are heavily used in:

- loss computation
- attention mechanisms
- pooling layers
- statistical normalization
- feature aggregation
- probability systems
- latent state summarization
""")

    print("=" * BREAK_SIZE)


if __name__ == "__main__":

    reduction_operations()
