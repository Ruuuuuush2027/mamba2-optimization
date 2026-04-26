# mamba2-optimization (from mamba2-minimal)

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

![Mamba-2](https://github.com/state-spaces/mamba/blob/f9dbb4fdb2705d71282e0db184d177c6375623f0/assets/ssd_algorithm.png)
> **Transformers are SSMs: Generalized Models and Efficient Algorithms**\
>     **Through Structured State Space Duality**\
> Tri Dao*, Albert Gu*\
> Paper: https://arxiv.org/abs/2405.21060

Mamba is a new class of foundation models, most notable for _not_ being based on the Transformer architecture. Instead it is in the family of State Space Models (SSMs) that maps a sequence through a hidden state in the fashion of RNNs. This approach enables linear scaling in computation and memory with respect to sequence length during training (unlike transformer's quadratic complexity), as well as constant time per step during inference. Mamba-2 builds upon Mamba-1 by imposing additional constraints on certain SSM parameters, allowing it to have much larger state dimensions and significantly improved training speed.

This implementation contains both original Mamba2 and Mamba2MC where we implemented a hidden state cache to try to improve performance.

## Some codes:
- run inference_test using checkpoints: `python inference_test.py --model-type "Mamba2MC" --checkpoint-dir "./checkpoints/mamba2-finetune/Mamba2MC-final"`
- run finetune: `python finetune.py --model-type "Mamba2MC"` or just Mamba2 `python finetune.py --model-type "Mamba2" --freeze-epochs 0` + `--resume-from-checkpoint "RESUME_PATH"` if need to continue finetuning from a checkpoint
- run benchmark: `python run_benchmark.py --model-type Mamba2 --checkpoint-path "YOUR_PATH"`