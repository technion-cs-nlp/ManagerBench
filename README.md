# ManagerBench - Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs



## Overview

ManagerBench is a comprehensive evaluation framework designed to assess the safety-pragmatism trade-off in autonomous Large Language Models (LLMs). This repository contains the implementation, benchmark, and evaluation tools presented in our paper.

![img.png](img.png)
## Table of Contents
- [Abstract](#abstract)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation Parameters](#evaluation-parameters)
- [Results](#results)
- [Citation](#citation)



### Abstract
As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions.

![img_2.png](img_2.png)
### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/technion-cs-nlp/ManagerBench.git
cd ManagerBench
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate manager_bench
```

3. Ensure the benchmark data is properly placed in the `bench/` directory.

## Usage

### Basic Evaluation

To run ManagerBench on a specific model:

```bash
python ManagerBench.py --model_name <model_name> --full_evaluation
```

Replace `<model_name>` with your desired model identifier.

### Example

```bash
python ManagerBench.py --model_name gpt-4o-2024-08-06 --full_evaluation
```

Results will be automatically saved in the `results/` folder and displayed in the terminal, will show the result of the model similar to the results in Table 1.

## Configuration

### API Keys

If you're evaluating a model that requires an API key, configure it in the `api_key.py` file.



### Evaluation Parameters

The following optional parameters allow you to customize the evaluation:

| Parameter | Description | Default | 
|-----------|-------------|---------|
| `--full_evaluation` | Enable comprehensive evaluation as shown in Table 1 | `False` |
| `--benefit_percentage` | Specify the percentage of benefit scenarios | `50%`   |
| `--harm_percentage` | Specify the percentage of harm scenarios | `5%`    | 
| `--push_evaluation` | Enable goal-oriented prompt evaluation | `False` | 


### Advanced Usage Example

```bash
python ManagerBench.py \
  --model_name gpt-4o-2024-08-06 \
  --benefit_percentage 60 \
  --harm_percentage 10 \
  --push_evaluation \
```
Will evaluate GPT-4o on with nudge, human harm percentage of 10% and a benefit of 60%.

## Results

- **Output Location**: All evaluation results are saved in the `results/` directory
- **Format**: Final result are provided at the terminal output


## Citation

If you use ManagerBench in your research, please cite our paper:

```bibtex
@article{simhi2025managerbench,
  title={ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs},
  author={Simhi, Adi and Herzig, Jonathan and Tutek, Martin and Itzhak, Itay and Szpektor, Idan and Belinkov, Yonatan},
  journal={arXiv preprint arXiv:2510.00857},
  year={2025}
}
```


## Repository Structure

```
ManagerBench/
├── bench/                  # Benchmark datasets
├── results/                # Evaluation results
├── ManagerBench.py         # Main evaluation script
├── api_key.py              # API configuration
├── environment.yml         # Conda environment specification
└── README.md               # This file
```





