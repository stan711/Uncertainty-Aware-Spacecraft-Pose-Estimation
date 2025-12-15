# Uncertainty-Aware Spacecraft Pose Estimation (SPNv2)

This repository contains a **course project** that reproduces and extends
**Spacecraft Pose Network v2 (SPNv2)** with a focus on **uncertainty-aware
inference**, including Monte Carlo Dropout and post-processing filters.

> ⚠️ **Academic Notice**  
> This codebase is intended for **educational and research purposes only**.
> It is **not an official implementation** of SPNv2 and is **not affiliated**
> with the original authors.

---

## Project Context

This work was developed as part of a **university class project** focused on
replicating state-of-the-art spacecraft pose estimation methods and exploring
uncertainty modeling techniques for spaceborne vision systems.

The primary goals are:
- Faithful replication of SPNv2 baselines
- Quantitative analysis of epistemic uncertainty
- Controlled experimental extensions (MC Dropout, filtering)

---

## Original Work and Attribution

This project is based on the following paper and codebase:

> **SPNv2: Robust Multi-task Learning and Online Refinement for Spacecraft Pose Estimation across Domain Gap**  
> Authors: *[Tae Ha Parka, Simone D’Amicoa]*  
> Paper: *https://arxiv.org/pdf/2203.04275*  
> Code: https://github.com/tpark94/spnv2

All credit for the original model architecture, training strategy, and dataset
design belongs to the SPNv2 authors.  
This repository **does not claim originality** over the core SPNv2 method.

---

## Repository Structure

core/ Model, dataset, loss, solver
configs/ Experiment configurations
scripts/ Training and evaluation entry points
tools/ Analysis and utilities
docs/ Project documentation

Training logs, checkpoints, and outputs are intentionally excluded to keep the
repository lightweight and reproducible.

---

## Reproducibility Notes

- Experiments are configured via YAML/JSON configs
- Random seeds are set where applicable
- Minor numerical differences from published results are expected due to:
  - Hardware differences
  - Framework / CUDA version differences
  - Non-deterministic GPU kernels

---

## License and Usage

This repository is released under the **MIT License**.  
Please note that the **original SPNv2 code and models may be subject to a
different license**, which must be respected independently.

---

## Citation

If you use this repository, please cite the **original SPNv2 paper**, not this
course project.


