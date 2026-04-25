import torch

from xdl_densecaps.models.densenet_capsnet import CapsuleMarginLoss, DenseNetCapsHead


def test_caps_head_outputs_binary_probabilities_and_backprops():
    head = DenseNetCapsHead(
        in_channels=4,
        feature_h=2,
        feature_w=2,
        num_classes=2,
        primary_caps_dim=4,
        capsule_dim=4,
        digit_caps_dim=6,
        num_capsules=3,
        capsule_routing_iters=2,
        digit_routing_iters=2,
    )
    criterion = CapsuleMarginLoss()

    features = torch.randn(2, 4, 2, 2)
    labels = torch.tensor([0, 1])

    probabilities, digit_capsules = head(features)
    loss = criterion(probabilities, labels)
    loss.backward()

    assert probabilities.shape == (2, 2)
    assert digit_capsules.shape == (2, 2, 6)
    assert torch.all(probabilities >= 0)
    assert head.capsules.weight.grad is not None
