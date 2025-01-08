# polishing-cheq

***

Code accompanying the paper **"CHEQ-ing the Box: Safe Variable Impedance Learning for Robotic Polishing".**

## Installation
Creation via virtual environment:
```bash
scripts/build.sh
```

## Usage
Training can be performed with the parallel setup. For this open up two shell terminals and activate the virtual environment.
Afterwards perform
```bash
python learn.py
```
in the first terminal, followed by
```bash
python act.py
```
in the second terminal.

## Logging
The logging of the runs is done with [Weights and Biases](https://wandb.ai). It is necessary to create an account and perform an intial login as described on their website. The details of your account can be specified in the [config](config/config.yaml)-file.
