# Mamba-2 Memory Caching (MC) Experiments

A project aimed at improving the [Mamba-2](https://arxiv.org/abs/2405.21060) foundation model by augmenting it with an explicit Memory Caching mechanism to enhance long-context recall and performance. Built on top of a minimal PyTorch implementation of Mamba-2.

## Models & Algorithm Differences

We implemented and compared three models to test our hypothesis:
1. **Mamba2 (Baseline)**: The original State Space Model (SSM) architecture. It compresses history into a fixed-size state, which provides linear scaling but can sometimes struggle with the exact recall of distant tokens compared to Transformers.
2. **Mamba2MC (Memory Caching)**: We augment Mamba-2 with a hidden state cache. It periodically caches previous segment hidden states. During generation, it uses a learned projection matrix (`W`) and a scalar gate (`online_bias`) to compute a weighted mix of these historical segments, blending them with the current token's hidden state. This acts as a pseudo-attention mechanism over compressed historical chunks.
3. **Mamba2MC_Select (Selective Memory Caching)**: A more advanced variant of the MC model. Instead of keeping a naive sliding window of recent segments, it introduces a scoring network (`select_score`). It scores each segment and retains only the top-K highest-scoring segments in the cache, allowing the model to hold onto important context for much longer sequences without blowing up memory.

## Dataset & Training

**Dataset**: We used a mixed dataset consisting of **WikiText** and **FineWeb**. The total training budget was **~27M tokens**.

**Training Setup**:
To properly train the newly initialized memory caching parameters without destroying the pre-trained Mamba-2 weights, we utilized a staged training approach:
*   **Freeze Stage (1 Epoch)**: Base Mamba-2 parameters are frozen. Only the new MC parameters (the mixing weights, gates, and selector networks) are trained.
*   **Finetune Stage (1 Epoch)**: All parameters are unfrozen and fine-tuned end-to-end.

**Hyperparameters**: 
*   Learning Rate: 2e-5 (Cosine scheduler with 0.01 warmup ratio)
*   Batch Size: 2
*   Gradient Accumulation Steps: 8
*   Max Grad Norm: 1.0
*   Weight Decay: 0.01
*   Hardware: 1x **NVIDIA L40S** GPU (took approximately **1 day per model**)

## Evaluation Metrics & Results

We evaluated the models on several benchmarks: **WikiText Perplexity**, **PIQA**, **HellaSwag**, **ARC**, and a long-context **NIAH (Needle-In-A-Haystack)** retrieval task.

| Model | WikiText PPL ↓ | PIQA Accuracy ↑ | HellaSwag Accuracy ↑ | ARC Accuracy ↑ | NIAH Accuracy ↑ |
| --- | --- | --- | --- | --- | --- |
| **Mamba2** | 10.310 | 0.736 | 0.454 | 0.321 | 1.000 |
| **Mamba2 MC** | 10.095 | 0.728 | 0.446 | 0.311 | 0.850 |
| **Mamba2 MC Select** | 10.088 | 0.728 | 0.446 | 0.311 | 0.900 |

## Analysis & Conclusion

The core idea behind this project was to combine the efficiency of Mamba-2's SSM with the explicit historical routing of Transformer KV-caches. 

**What works?** 
* **Language Modeling**: We observed a slight improvement in **WikiText Perplexity** (dropping from 10.31 to 10.08) for the MC models, indicating that the cache mechanism was successfully learning to predict the next token better on the fine-tuning distribution. 
* **Long-Context Focus**: While the MC models slightly regressed on zero-shot reasoning tasks and the NIAH task compared to the fully pre-trained original Mamba-2, **Mamba2 MC Select** improved upon the base MC model in the **NIAH** task (0.900 vs 0.850). This demonstrates that the selective caching mechanism successfully improves long-context focus and recall over a naive sliding window cache.

**Why did it fail to significantly beat the baseline?**
The primary bottleneck was the **lack of fine-tuning data**. We attempted to teach a 1.3B parameter model an entirely new routing and attention-like mechanism using only **27M tokens**. Learning to properly score, retain, and cross-attend to memory segments requires observing a vast amount of diverse, long-context data. Because our dataset was too small, the model likely overfit its new parameters to the local distribution of our training mix, slightly degrading its generalized pre-trained zero-shot capabilities. 

We also observed that the added scoring network chunk in MC Select did not receive enough training data to significantly differentiate its generalized zero-shot performance from pure memory caching MC (scoring identically on PIQA, HellaSwag, and ARC).

Future work with a significantly larger token budget (e.g., 5B+ tokens) and longer sequence lengths during training would be necessary to fully realize the potential of the Mamba2MC and MC_Select architectures.

## Running the Code
- **Inference test:** `python inference_test.py --model-type "Mamba2MC" --checkpoint-dir "./checkpoints/mamba2-finetune/Mamba2MC-final" --prompt "Your prompt"`
- **Finetune:** `python finetune.py --model-type "Mamba2MC"` (Use `--freeze-epochs 0` for base Mamba2, and `--resume-from-checkpoint` to continue)
- **Benchmark:** `python run_benchmark.py --model-type Mamba2 --checkpoint-path "YOUR_PATH"`