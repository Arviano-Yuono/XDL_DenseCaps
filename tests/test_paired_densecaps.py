import torch

from xdl_densecaps.models import CapsuleMarginLoss, PairedDenseCapsFusionHead


def test_paired_fusion_head_outputs_multiclass_probabilities_and_backprops():
    head = PairedDenseCapsFusionHead(
        num_branch_capsules=3,
        capsule_dim=4,
        num_classes=3,
        digit_caps_dim=5,
        routing_iters=2,
    )
    criterion = CapsuleMarginLoss()

    whole_capsules = torch.randn(2, 3, 4)
    detail_capsules = torch.randn(2, 3, 4)
    labels = torch.tensor([0, 2])

    probabilities, digit_capsules = head(whole_capsules, detail_capsules)
    loss = criterion(probabilities, labels)
    loss.backward()

    assert probabilities.shape == (2, 3)
    assert digit_capsules.shape == (2, 3, 5)
    assert torch.all(probabilities >= 0)
    assert head.digit_capsules.weight.grad is not None
