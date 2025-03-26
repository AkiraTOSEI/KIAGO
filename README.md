# Knowledge-Integrated Adaptive Gradient-based Optimisation (KIAGO) for Superconducting Materials Design


## Summary
KIAGO is a framework for superconducting materials design that leverages domain knowledge to guide the optimisation process. The framework is based on a gradient-based optimisation algorithm using deep learning property prediction models. KIAGO can flexibly incorporate physical insights, such as elemental composition, oxidation states, and charge neutrality, in an adaptive manner. Our experiments demonstrate that KIAGO outperforms conventional elemental substitution and generative models in proposing high-$T_c$ candidates efficiently. KIAGO can propose high-$T_c$ candidates without being limited by the original distribution, and it can maintain charge neutrality and oxidation states perfectly. Moreover, KIAGO can propose candidate compositions that share the same elements as hydride superconductors reported in other literature despite their absence from the SuperCon dataset. These results highlight the potential of KIAGO for discovering novel materials.

![hohoge](./images/figure1.png)

## Paper
For detailed information, please refer to our paper:

[A Straightforward Gradient-Based Approach for High-Tc Superconductor Design: Leveraging Domain Knowledge via Adaptive Constraints](https://arxiv.org/abs/2403.13627)

## Prerequisites
We use PyTorch 2.2.1 and Pytorch-Lightning 1.9.5 for the implementation. Download files from the following link. And then place pre-trained models (`elemnet.pth` and `baseline_surrogate.pt`) in the `models/surrogate_model` directory and other numpy files (`random-0.05-0.15-balancing.npz` and `AtomMap-base.npy`) in the home directory. 
https://drive.google.com/drive/folders/1WBcDbxnqJiP2eZz0AuQ1m5lMbWnSMdSw?usp=sharing


## Usage
To run the code, please execute the following command.
When you run this code, you will conduct three experiments in our research paper: 1. Generating superconductors with higher $T_c$ based on existing ones. 2. Element substitution 3. Proposing novel hydride superconductors.

```bash
python main.py
```
Overview of the main script
In main.py, the main_experiment function is a function that executes the experiment, and in InverseOpt4PeriodicTable function, hyperparameters are reflected, and then optimization of the inverse problem is performed by InvModule4PeriodicTable instance.
Here, we give noise to the base material (YBa2Cu3O7) to create 4096 initial values, and optimize them simultaneously. The optimization is performed in two stages: first, the composition is optimized without converting composition ratios into integers, and then the composition is optimized with a special loss to convert compositional ratios into integers. The results are output to the results4inverse directory.

## Output files
In the `results4inverse` directory, the optimization results are output. The main output is a csv file that outputs the results of the 4096 samples optimized. The npz file is an intermediate file for constructing the csv. The overview of each column is as follows.
- Initial Optimized Composition: Result of composition optimization (first stage)
- Rounded Optimized Composition: Composition formula after further optimization using the loss converting composition ratios into integers (second stage optimization)
- Rounded Optimized Tc: Predicted Tc after second stage optimization
- Initial Optimized Tc: Predicted Tc after first stage optimization
- Rounded Optimized Ef: Predicted formation energy after second stage optimization
- Initial Optimized Ef: Predicted formation energy after first stage optimization
- structure_atom_count: Number of atoms in the optimized composition

