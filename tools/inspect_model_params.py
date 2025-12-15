#!/usr/bin/env python3
import torch
import _init_paths
from config import cfg, update_config
from nets import build_spnv2

def build_model_from_cfg(cfg_file):
    class Args:
        pass
    args = Args()
    args.cfg = cfg_file
    args.opts = []
    update_config(cfg, args)
    model = build_spnv2(cfg)
    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    # φ0 GN
    cfg_file_phi0 = "experiments/offline_train_full_config_phi0_gn.yaml"
    model_phi0 = build_model_from_cfg(cfg_file_phi0)
    params_phi0 = count_params(model_phi0)
    print(f"φ0 total params: {params_phi0:,}")

    # φ3 BN
    cfg_file_phi3 = "experiments/offline_train_full_config_phi3_BN.yaml"
    model_phi3 = build_model_from_cfg(cfg_file_phi3)
    params_phi3 = count_params(model_phi3)
    print(f"φ3 total params: {params_phi3:,}")
