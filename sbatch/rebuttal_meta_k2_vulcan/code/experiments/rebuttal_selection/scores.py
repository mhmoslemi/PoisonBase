"""PyTorch ports of the cited authors' sample-weight and poisoning scores.

Sources:
  uber-research/learning-to-reweight-examples/models/reweight_model.py
  kohpangwei/influence-release/influence/genericNeuralNet.py
  kohpangwei/influence-release/influence/dataset_poisoning.py
See README.md for the targeted-poisoning adaptation and numerical settings.
"""
import torch
import torch.nn.functional as F


def meta_weight_gradients(net, images, labels, target, target_label, step_size=1.0):
    """Negative derivative of target CE through a zero-weighted SGD step.

    This explicitly differentiates sample weights, as the authors' legacy
    implementation does. It does not substitute a stored alignment score.
    The checkpoint is held fixed; BN uses its saved evaluation statistics.
    """
    parameters = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    weights = images.new_zeros(len(images), requires_grad=True)
    losses = F.cross_entropy(net(images), labels, reduction='none')
    gradients = torch.autograd.grad(
        (weights * losses).sum(), tuple(parameters.values()), create_graph=True)
    updated = {name: parameter - step_size * gradient
               for (name, parameter), gradient in zip(parameters.items(), gradients)}
    logits = torch.func.functional_call(net, (updated, buffers), (target,))
    outer_loss = F.cross_entropy(logits, target_label)
    return -torch.autograd.grad(outer_loss, weights)[0].detach()


def normalize_meta_weights(values):
    """Official nonnegative clipping and sum normalization over the whole pool."""
    positive = values.clamp_min(0)
    denominator = positive.sum()
    return positive / (denominator + (denominator == 0).to(values.dtype))


def loss_gradient(net, images, labels):
    loss = F.cross_entropy(net(images), labels)
    return tuple(value.detach() for value in torch.autograd.grad(loss, tuple(net.parameters())))


def hessian_vector_product(net, images, labels, vector):
    """Exact mean-CE HVP; the vector is constant in this differentiation."""
    parameters = tuple(net.parameters())
    loss = F.cross_entropy(net(images), labels)
    gradient = torch.autograd.grad(loss, parameters, create_graph=True)
    product = sum((a * b.detach()).sum() for a, b in zip(gradient, vector))
    values = torch.autograd.grad(product, parameters)
    return tuple(value.detach() for value in values)


def lissa_update(rhs, estimate, hvp, scale, damping):
    """Authors' recurrence; its effective ridge is scale*damping."""
    return tuple((a + (1.0 - damping) * b - c / scale).detach()
                 for a, b, c in zip(rhs, estimate, hvp))


def influence_input_scores(net, images, labels, inverse_hvp, pixel_scale=None):
    """Official poisoning selector: L1 norm of the influence input derivative.

    Unlike upweighting influence, this measures sensitivity to perturbing an
    existing training image. Convert normalized-image derivatives to raw-pixel
    derivatives when the dataset uses channel-specific standard deviations.
    The minus sign and common 1/n factor do not affect this norm/ranking.
    """
    images = images.detach().requires_grad_(True)
    losses = F.cross_entropy(net(images), labels, reduction='sum')
    gradient = torch.autograd.grad(losses, tuple(net.parameters()), create_graph=True)
    interaction = sum((a * b.detach()).sum() for a, b in zip(gradient, inverse_hvp))
    pixels = torch.autograd.grad(interaction, images)[0]
    if pixel_scale is not None:
        pixels = pixels * pixel_scale
    return pixels.detach().flatten(1).abs().sum(1)


def vector_norm(vector):
    return sum(value.double().square().sum() for value in vector).sqrt().item()
