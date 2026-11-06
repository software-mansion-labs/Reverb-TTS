from echo.model.model import EchoModel


def freeze(model: EchoModel) -> EchoModel:
    """
    Turns a model into a fixed reference: no gradients, and out of training mode.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model.eval()
