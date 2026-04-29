"""Model modules for XDL DenseCaps."""

from xdl_densecaps.models.backbone import (
    DENSENET121_TRUNK_LAYERS,
    DenseNet121BackboneConfig,
    DenseNet121FeatureExtractor,
)
from xdl_densecaps.models.densenet_capsnet import (
    CapsuleMarginLoss,
    DenseNetCapsHead,
    DenseNetCapsNetClassifier,
    DenseNetCapsNetConfig,
    RoutingCapsuleLayer,
)
from xdl_densecaps.models.densenet_classifier import DenseNet121Classifier, DenseNetClassifierConfig
from xdl_densecaps.models.lesion_proposal import DenseNetLesionProposalModel, LesionProposalOutput
from xdl_densecaps.models.paired_densecaps import (
    DenseCapsEncoderBranch,
    PairedDenseCapsFusionHead,
    PairedDenseCapsNetClassifier,
    PairedDenseCapsNetConfig,
)

__all__ = [
    "CapsuleMarginLoss",
    "DENSENET121_TRUNK_LAYERS",
    "DenseCapsEncoderBranch",
    "DenseNet121BackboneConfig",
    "DenseNetCapsHead",
    "DenseNetCapsNetClassifier",
    "DenseNetCapsNetConfig",
    "DenseNet121Classifier",
    "DenseNetClassifierConfig",
    "DenseNet121FeatureExtractor",
    "DenseNetLesionProposalModel",
    "LesionProposalOutput",
    "PairedDenseCapsFusionHead",
    "PairedDenseCapsNetClassifier",
    "PairedDenseCapsNetConfig",
    "RoutingCapsuleLayer",
]
