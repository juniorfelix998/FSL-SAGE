# ------------------------------------------------------------------------------
# ImageNet-pretrained initialization for the split ResNet-18 client/server pair.
#
# WHY THIS IS SHARED: both zeroth-order methods in this benchmark
# (HKU-WILL-Lab/HO-SFL's `mu_splitfed` and `ho_sfl`) require it, and the
# reference implementation enables it for both via one `conf/base.yaml`
# (`model.use_pretrained: True`, `model.freeze_bn: True`). It is not a nicety:
# a P-direction zeroth-order estimator's convergence scales with d/P, so a
# ~683k-parameter client updated by ~5-10 probe directions cannot be trained
# from scratch in the round budgets these methods use. The reference works
# because the client starts near-optimal and the first-order server does the
# learning; without pretrained weights the method degenerates to training a head
# on a frozen random feature extractor, which is what produced this harness's
# near-chance HO-SFL numbers.
# ------------------------------------------------------------------------------
import torch
import torchvision.models as tv_models


# ------------------------------------------------------------------------------
def pretrained_resnet18_state_dict():
    pretrained = tv_models.resnet18(
        weights=tv_models.ResNet18_Weights.IMAGENET1K_V1, progress=False
    )
    return pretrained.state_dict()


# ------------------------------------------------------------------------------
# cut-aware: ResNetClient/ResNetServer each carry their own `client_layers`
# (1/2/3, see src/models/resnet.py) reflecting which of the 4 ResNet stages
# they were built with -- read directly off the model rather than assumed, so
# this works correctly under any of the harness's shallow/middle/deep cuts.
_ALL_STAGE_PREFIXES = ('layer1.', 'layer2.', 'layer3.', 'layer4.')


def load_pretrained_client(model, full_state):
    prefixes = ('conv1.', 'bn1.') + _ALL_STAGE_PREFIXES[:model.client_layers]
    client_state = {
        k: v for k, v in full_state.items() if k.startswith(prefixes)
    }
    model.load_state_dict(client_state, strict=True)


# ------------------------------------------------------------------------------
def load_pretrained_server(model, full_state):
    # fc excluded: pretrained fc is 1000-way (ImageNet), this benchmark's is
    # num_classes-way -- left at the harness's own random init.
    prefixes = _ALL_STAGE_PREFIXES[model.client_layers:]
    server_state = {
        k: v for k, v in full_state.items() if k.startswith(prefixes)
    }
    model.load_state_dict(server_state, strict=False)


# ------------------------------------------------------------------------------
def freeze_batchnorm_affine(model):
    '''Freeze BatchNorm's affine weight/bias, matching the reference's
    src/models/registry.py.

    This does NOT put BatchNorm in eval mode or freeze its running mean/var --
    the reference calls a blanket `.train()` on the whole model every round,
    re-enabling ordinary train-mode running-stat updates regardless of this
    flag. The lasting effect is that BatchNorm's affine params become invisible
    to the zeroth-order perturbation loops (`_perturb` skips any param with
    `requires_grad=False`), so they stay pinned at their pretrained values while
    running stats keep adapting -- and the probe budget is not wasted on them.
    '''
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
