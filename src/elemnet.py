import re

import numpy as np
import torch
from torch import nn



def parse_architecture(architecture_str):
    """
    (English): Parse architecture string like "1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1"
    and extract (units, layers, dropout flag, residual flag, etc.) of each block
    so that it can be constructed with nn.Sequential

    """
    blocks = architecture_str.strip().split("-")
    arch_info = []
    for block in blocks:
        # "1024x4D" -> units=1024, layers=4, dropout
        # "64x2" -> units=64, layers=2, no dropout
        # "1" -> units=1 (output layer)
        if "x" in block:
            main_part = block.split("x")  # e.g. ["1024", "4D"]
            units_part = main_part[0]

            units_match = re.findall(r"\d+", units_part)
            units = int(units_match[0]) if units_match else 32

            tail_part = main_part[1]
            layers_match = re.findall(r"\d+", tail_part)
            layers = int(layers_match[0]) if layers_match else 1
            # dropout, residual
            dropout_flag = "D" in tail_part
            residual_flag = ("R" in tail_part) or ("B" in tail_part)

            arch_info.append((units, layers, dropout_flag, residual_flag))
        else:
            units_match = re.findall(r"\d+", block)
            units = int(units_match[0]) if units_match else 1
            dropout_flag = "D" in block
            residual_flag = "R" in block
            arch_info.append((units, 1, dropout_flag, residual_flag))
    return arch_info


class ElemNet(torch.nn.Module):
    """
    PyTorch Network
    """

    def __init__(self, input_dim, architecture_str, activation="relu", dropouts=[]):
        super(ElemNet, self).__init__()
        self.arch_info = parse_architecture(architecture_str)
        self.activation = nn.ReLU

        layers = []
        in_features = input_dim
        dropout_idx = 0

        for i, (units, num_layers, do_flag, res_flag) in enumerate(self.arch_info):
            for layer_i in range(num_layers):
                fc = nn.Linear(in_features, units)
                nn.init.zeros_(fc.bias)
                nn.init.xavier_uniform_(fc.weight, gain=nn.init.calculate_gain("relu"))
                layers.append(fc)

                # 活性化
                if i == len(self.arch_info) - 1 and layer_i == num_layers - 1:
                    print(
                        f"block {i} layer_i:{layer_i} nn.Linear({in_features}, {units})"
                    )

                else:
                    layers.append(self.activation())
                    print(
                        f"block {i} layer_i:{layer_i} nn.Linear({in_features}, {units}), Activation: {self.activation}"
                    )

                in_features = units

            if do_flag and dropout_idx < len(dropouts):
                p = dropouts[dropout_idx]
                p_dropout = float(f"{np.round(1.0 - p, 2):.2f}")
                layers.append(nn.Dropout(p=p_dropout))  # あとで消す
                print(
                    f"block {i}: nn.Dropout({p_dropout}) (p={p})",
                )

            if do_flag:
                dropout_idx += 1

        if self.arch_info[-1][0] != 1:
            layers.append(nn.Linear(in_features, 1))
            print(f"final block {len(self.arch_info)}: nn.Linear({in_features}, 1)")

        print("len(layers):", len(layers))
        for l_i, layer in enumerate(layers):
            print(f"layer {l_i}: {layer}")

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out.view(-1)
