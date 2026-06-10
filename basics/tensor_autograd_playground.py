# %%
import torch

BREAK_SIZE = 80


def autograd_playground():
    # %%
    print("=" * BREAK_SIZE)
    print("PYTORCH AUTOGRAD PLAYGROUND")
    print("=" * BREAK_SIZE)

    print(f"PyTorch Version : {torch.__version__}")

    print("=" * BREAK_SIZE)
    # %% [markdown]
    ####################################################################
    # BASIC AUTOGRAD CONCEPT
    ####################################################################

    ####################################################################
    # Autograd = Automatic Differentiation Engine
    #
    # PyTorch dynamically builds a computational graph.
    #
    # Every operation becomes a graph node.
    #
    # During backward propagation:
    #
    # gradients are computed using chain rule.
    #
    ####################################################################

    ####################################################################
    # MATHEMATICAL FOUNDATION
    #
    # Suppose:
    #
    # y = f(x)
    #
    # We want:
    #
    # dy/dx
    #
    # PyTorch computes this automatically.
    #
    ####################################################################
    # %% [markdown]
    ####################################################################
    # CREATE TENSOR WITH GRADIENT TRACKING
    ####################################################################
    # %%
    print("\n" + "#" * BREAK_SIZE)
    print("CREATE TENSOR WITH requires_grad=True")

    x = torch.tensor(2.0, requires_grad=True)

    print("Tensor x :")
    print(x)

    print("\nGradient tracking enabled :")
    print(x.requires_grad)
    # %% [markdown]
    ####################################################################
    # COMPUTATIONAL GRAPH
    ####################################################################


    ####################################################################
    # MATHEMATICAL FUNCTION
    #
    # y = x² + 3x + 2
    #
    ####################################################################

    ####################################################################
    # DERIVATIVE
    #
    # dy/dx = 2x + 3
    #
    ####################################################################
    # %%
    print("\n" + "#" * BREAK_SIZE)
    print("BUILD COMPUTATIONAL GRAPH")
    y = x**2 + 3*x + 2

    print("Computed y :")
    print(y)
    # %% [markdown]
    ####################################################################
    # tensor.backward()
    ####################################################################



    ####################################################################
    # DESCRIPTION:
    #
    # Computes gradients using backpropagation.
    #
    ####################################################################

    ####################################################################
    # BASIC ALGORITHM:
    #
    # 1. Start from output node
    #
    # 2. Traverse graph backward
    #
    # 3. Apply chain rule
    #
    # 4. Accumulate gradients
    #
    ####################################################################

    ####################################################################
    # CHAIN RULE
    #
    # If:
    #
    # z = f(y)
    # y = g(x)
    #
    # then:
    #
    # dz/dx = dz/dy × dy/dx
    #
    ####################################################################
    # %%
    y.backward()
    # %% [markdown]
    ####################################################################
    # Expected derivative:
    #
    # dy/dx = 2x + 3
    #
    # x = 2
    #
    # dy/dx = 7
    #
    ####################################################################
    print("\n" + "#" * BREAK_SIZE)
    print("method tensor.backward()")
    print("Gradient dy/dx :")
    print(x.grad)

    # %% [markdown]
    ####################################################################
    # torch.autograd.grad()
    ####################################################################


    ####################################################################
    # DESCRIPTION:
    #
    # Computes gradients manually without storing
    # them permanently in .grad
    #
    ####################################################################

    ####################################################################
    # MATHEMATICAL FUNCTION
    #
    # z = x³
    #
    ####################################################################

    ####################################################################
    # DERIVATIVE
    #
    # dz/dx = 3x²
    #
    ####################################################################
    # %%
    print("\n" + "#" * BREAK_SIZE)
    print("method torch.autograd.grad()")
    a = torch.tensor(3.0, requires_grad=True)

    z = a**3

    grad = torch.autograd.grad(
        outputs=z,
        inputs=a
    )

    print("Computed z :")
    print(z)

    print("\nGradient dz/dx :")
    print(grad)
    # %% [markdown]
    ####################################################################
    # Expected:
    #
    # dz/dx = 3 × 3² = 27
    #
    ####################################################################

    ####################################################################
    # Important Difference:
    #
    # backward()
    # -> accumulates into .grad
    #
    # autograd.grad()
    # -> returns gradients directly
    #
    ####################################################################
    # %% [markdown]
    ####################################################################
    # torch.no_grad()
    ####################################################################



    ####################################################################
    # DESCRIPTION:
    #
    # Temporarily disables gradient tracking.
    #
    ####################################################################

    ####################################################################
    # Why Important?
    #
    # During inference:
    #
    # gradients are unnecessary.
    #
    # This saves:
    #
    # - memory
    # - computation
    #
    ####################################################################
    # %%
    print("\n" + "#" * BREAK_SIZE)
    print("method torch.no_grad()")
    with torch.no_grad():

        b = torch.tensor(5.0, requires_grad=True)

        c = b * 2

        print("Tensor c :")
        print(c)

        print("\nGradient tracking enabled?")
        print(c.requires_grad)
    # %% [markdown]
    ####################################################################
    # Important Observation:
    #
    # requires_grad becomes False inside no_grad()
    #
    ####################################################################
    # %% [markdown]
    ####################################################################
    # torch.enable_grad()
    ####################################################################



    ####################################################################
    # DESCRIPTION:
    #
    # Re-enables gradient tracking.
    #
    ####################################################################
    # %%
    print("\n" + "#" * BREAK_SIZE)
    print("method torch.enable_grad()")
    with torch.no_grad():

        p = torch.tensor(4.0, requires_grad=True)

        with torch.enable_grad():

            q = p**2

            print("Tensor q :")
            print(q)

            print("\nGradient tracking enabled?")
            print(q.requires_grad)
    # %% [markdown]
    ####################################################################
    # Important:
    #
    # enable_grad() overrides no_grad()
    #
    ####################################################################
    # %% [markdown]
    ####################################################################
    # torch.set_grad_enabled()
    ####################################################################



    ####################################################################
    # DESCRIPTION:
    #
    # Dynamically enables/disables gradients.
    #
    ####################################################################
    # %%
    print("\n" + "#" * BREAK_SIZE)
    print("method torch.set_grad_enabled()")
    training_mode = True

    with torch.set_grad_enabled(training_mode):

        r = torch.tensor(6.0, requires_grad=True)

        s = r**2

        print("Training mode :", training_mode)

        print("Gradient tracking enabled?")
        print(s.requires_grad)
    # %% [markdown]
    ####################################################################
    # This is extremely useful in:
    #
    # training loops
    #
    # Example:
    #
    # if training:
    #     enable gradients
    #
    # else:
    #     disable gradients
    #
    ####################################################################
    # %% [markdown]
    ####################################################################
    # PRACTICAL MINI NEURAL NETWORK EXAMPLE
    ####################################################################


    # %% [markdown]
    ####################################################################
    # Suppose:
    #
    # y = wx + b
    #
    # Loss:
    #
    # L = (y - target)^2
    #
    ####################################################################

    ####################################################################
    # GOAL:
    #
    # compute:
    #
    # dL/dw
    # dL/db
    #
    ####################################################################
    # %%
    print("\n" + "#" * BREAK_SIZE)
    print("MINI NEURAL NETWORK STYLE EXAMPLE")
    w = torch.tensor(2.0, requires_grad=True)

    b = torch.tensor(1.0, requires_grad=True)

    x_input = torch.tensor(3.0)

    target = torch.tensor(10.0)

    ####################################################################
    # Forward Pass
    ####################################################################

    y_pred = w * x_input + b

    ####################################################################
    # Mean Squared Error
    ####################################################################

    loss = (y_pred - target)**2

    print("Predicted y :")
    print(y_pred)

    print("\nLoss :")
    print(loss)

    ####################################################################
    # Backward Pass
    ####################################################################

    loss.backward()
    # %% [markdown]
    ####################################################################
    # MATHEMATICAL DERIVATIVES
    #
    # L = (wx+b-target)^2
    #
    # dL/dw = 2(wx+b-target)x
    #
    # dL/db = 2(wx+b-target)
    #
    ####################################################################

    print("\nGradient dL/dw :")
    print(w.grad)

    print("\nGradient dL/db :")
    print(b.grad)
    # %%
    ####################################################################
    # FINAL SUMMARY
    ####################################################################

    print("\n" + "=" * BREAK_SIZE)
    print("AUTOGRAD SUMMARY")
    print("=" * BREAK_SIZE)

    print("""
Autograd is PyTorch's automatic differentiation engine.

Core ideas:

1. Build computational graph
2. Track operations
3. Apply chain rule
4. Compute gradients automatically

Core APIs:

- backward()
- autograd.grad()
- no_grad()
- enable_grad()
- set_grad_enabled()

These APIs power:

- neural networks
- transformers
- optimization
- reinforcement learning
- diffusion models
- latent semantic systems
""")

    print("=" * BREAK_SIZE)


if __name__ == "__main__":

    autograd_playground()