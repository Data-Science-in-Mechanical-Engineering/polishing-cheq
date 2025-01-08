from torch import nn
from typing import List, Type


def create_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        layer_norm: bool = False
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), by serially chaining input layer of input dim, hidden layers of dimensions
    specified by hidden_dims and a final output layer, with activation functions inbetween.

    :param input_dim: Dimension of the input tensor
    :param output_dim: Dimension of the output tensor
    :param hidden_dims: List of number of units per hidden layer. The length of the list corresponds to
                        the number of hidden layers
    :param activation_fn: The activation function to use after each layer
    :return: list of MLP layers and activations
    """

    if len(hidden_dims) > 0:
        modules = [nn.Linear(in_features=input_dim, out_features=hidden_dims[0]), activation_fn()]
        modules.append(nn.LayerNorm(normalized_shape=hidden_dims[0])) if layer_norm else None
    else:
        modules = []

    for i in range(len(hidden_dims) - 1):
        modules.append(nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1]))
        modules.append(nn.LayerNorm(normalized_shape=hidden_dims[i])) if layer_norm else None
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_input_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        modules.append(nn.Linear(in_features=last_layer_input_dim, out_features=output_dim))

    return modules
