import timm
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
from torchvision.models import resnet18

from .surrogate_utils import Read_AtomMap, num_input_features

def replace_bn_with_in(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            setattr(model, name, nn.InstanceNorm2d(num_features, affine=True))
        else:
            replace_bn_with_in(module)


class CNNBaseModel(nn.Module):
    def __init__(self, model_type: str, num_feat: int):
        super(CNNBaseModel, self).__init__()
        # Load a pre-trained ResNet18
        if model_type.lower() == "resnet18":
            self.base_model = resnet18(pretrained=False)
        else:
            raise Exception(f"Invalid model_type: {model_type}")

        self.base_model.conv1 = torch.nn.Conv2d(
            num_feat,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        if model_type.lower() == "convnext-tiny":
            self.base_model.classifier[2] = torch.nn.Linear(
                self.base_model.classifier[2].in_features, 1
            )
        else:
            self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 1)

        self.fin_act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fin_act(self.base_model(x))


class PeriodicTableSurrogateModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Define the model architecture.
        """
        super(PeriodicTableSurrogateModel, self).__init__()
        self.num_feat = num_input_features(
            cfg.sg_model.atom_map_type, cfg.sg_model.model_type
        )
        self.base_model = self.define_base_model(cfg.sg_model.model_type)
        self.atom_map = self.define_AtomMap(cfg)

        self.model_type = cfg.sg_model.model_type

    def x_atom_map_process(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type.startswith("T4PT-"):
            x = x.squeeze(-1).squeeze(-1)
        else:
            x = torch.sum(x * self.atom_map, dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_atom_map_process(x)
        return self.base_model(x)

    def define_AtomMap(self, cfg: DictConfig):
        atom_map = torch.from_numpy(Read_AtomMap(cfg)).float()
        if cfg.sg_model.atom_map_type == "LearnableAtomMap":
            required_grad = True
        else:
            required_grad = False

        atom_map = torch.nn.parameter.Parameter(atom_map, requires_grad=required_grad)
        return atom_map

    def define_base_model(self, model_type: str):
        if model_type.lower().startswith("resnet"):
            base_model = CNNBaseModel(model_type, self.num_feat)
        else:
            raise ValueError(f"Invalid model_type, your input : {model_type}")

        return base_model
