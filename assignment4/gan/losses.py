import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss_real = bce_loss(logits_real, torch.ones_like(logits_real), reduction='mean')
    loss_fake = bce_loss(1 - logits_fake, torch.ones_like(logits_fake), reduction='mean')
    # loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake), reduction='mean')

    return (loss_real + loss_fake)


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = bce_loss(logits_fake, torch.ones_like(logits_fake), reduction='mean')

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    
    loss_real = 0.5 * torch.mean((scores_real - 1) ** 2, dim=0)
    loss_fake = 0.5 * torch.mean(scores_fake ** 2, dim=0)

    loss = loss_real + loss_fake

    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = 0.5 * torch.mean((scores_fake - 1) ** 2, dim=0)

    return loss

def wassertein_descriminator_loss(scores_real, scores_fake):

    loss = torch.mean(scores_fake, dim=0) - torch.mean(scores_real, dim=0)

    return loss

def wassertein_generator_loss(scores_fake):
    return -torch.mean(scores_fake, dim=0)

def wassertein_gp_descriminator_loss(D, scores_real, scores_fake, real_data, fake_data, lambda_gp=10):
    batch_size, c, h, w = real_data.size
    epsilon = torch.randn(batch_size).to(real_data.device)
    epsilon = epsilon.view(batch_size, c, h, w)
    x_sample = epsilon * real_data + (1 - epsilon) * fake_data
    x_sample.requires_grad = True
    scores_sample = D(x_sample)
    gradients = torch.autograd.grad(
        outputs=scores_sample, 
        inputs=x_sample,
        grad_outputs=torch.ones_like(scores_sample),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = lambda_gp * ((gradients.norm(dim=1) - 1) ** 2).mean()
    loss = torch.mean(scores_fake, dim=0) - torch.mean(scores_real, dim=0) + gradient_penalty
    return loss

def wassertein_descriminator_loss_split(scores):
    return torch.mean(scores, dim=0)

def gradient_gp_loss(D, real_data, fake_data, lambda_gp=10):
    batch_size, c, h, w = real_data.size()
    epsilon = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    x_hat = epsilon * real_data + (1 - epsilon) * fake_data
    x_hat.requires_grad = True
    scores = D(x_hat)
    gradients = torch.autograd.grad(
        outputs=scores, 
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = lambda_gp * ((gradients.norm(dim=1) - 1) ** 2).mean()
    return gradient_penalty