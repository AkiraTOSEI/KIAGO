# Train and inference scripts for SuperDiff
These codes are used for comparison experiments of SuperDiff([Yuan & Dordevic, Sci Rep 14, 10275 (2024)](https://www.nature.com/articles/s41598-024-61040-3)).

These codes are almost same as [original GitHub repository](https://github.com/sdkyuanpanda/SuperDiff) of SuperDiff, however, we implement classifier guidance on SuperDiff.

## Train SuperDiff
`python python diffusion1d_v4_ilvr_unet_train.py`

After training, the trained models are located in `./models/`.

## Inference SuperDiff with/without Guidance
`python superdiff_guidance.py`

Then, you can find inference results in `./results/`

## Evaluation
After inference, please use the following commands. Result files are `results/*/summary.csv`.

`python evaluate.py` # For usual ILVR and element substitution

`python evaluate_hsc.py` # For hydride superconductors

