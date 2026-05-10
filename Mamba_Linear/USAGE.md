# Mamba2 vs Mamba2MC — Benchmark Workflow

## 0. 环境
```bash
pip install torch transformers datasets tqdm einops matplotlib numpy
```
GPU: 4070 Super 12G 够跑 130M~400M 级别的 Mamba2，2.7B 的 Mamba2 可能要降 batch 或开 fp16/bf16。

## 1. Checkpoint 路径
默认读取：
```
mamba2-ckpts/checkpoints/mamba2-finetune/Mamba2MC-final/pytorch_model.bin
```
不同位置用 `--ckpt /path/to/dir` 覆盖。Mamba2MC 训过的权重也可以直接载入到 baseline `Mamba2`（`W` 和 `online_bias` 在 baseline 里不存在，会自然忽略）。

## 2. 跑 benchmark

**主对比（baseline vs MC，全部任务）**
```bash
python run_benchmark.py \
    --models baseline mc_default \
    --tasks wikitext piqa longbench niah speed
```
结果会逐个写到 `results/main_*.json`。

**冒烟测试（小样本，几分钟跑完，确认管线 OK）**
```bash
python run_benchmark.py \
    --models baseline mc_default \
    --tasks wikitext piqa \
    --max-chunks 5 --max-examples 30
```

**消融：扫 segment_size**
```bash
python run_benchmark.py \
    --tasks wikitext niah \
    --sweep-segment-size 32 64 128 256
```
→ `results/sweep_seg_*.json`

**消融：扫 max_cached_segments**
```bash
python run_benchmark.py \
    --tasks wikitext niah \
    --sweep-cache-slots 4 8 16 32
```
→ `results/sweep_cache_*.json`

**只跑速度/显存**
```bash
python run_benchmark.py \
    --tasks speed \
    --speed-seq-lens 512 1024 2048 4096 8192 \
    --speed-n-new 64
```

## 3. 出图 + 报告
所有 JSON 都跑完之后：
```bash
python plot_and_report.py
```
会生成：
```
report/
├── figures/
│   ├── ppl_bar.png
│   ├── piqa_bar.png
│   ├── longbench_bar.png
│   ├── niah_<modelname>.png
│   ├── speed.png
│   ├── sweep_seg.png
│   └── sweep_cache.png
└── report.md
```
`report.md` 里已经按 *主对比表 → NIAH → segment_size sweep → cache sweep → 分析* 排好版，直接交。

## 4. 常用增量命令
- 想多加一组实验：跑一次对应的 `run_benchmark.py`，然后再跑一次 `plot_and_report.py` 即可（旧 JSON 会被复用）。
- 删除某个实验：直接删 `results/<name>.json`。
- 想换 LongBench 子集：`--longbench-tasks qasper hotpotqa gov_report passage_retrieval_en`。
- 想换 NIAH 网格：`--niah-ctx-lens 1024 2048 4096 8192 --niah-depths 0 0.2 0.4 0.6 0.8 1.0`。

## 5. 文件总览
```
mamba2.py                 # 基础 Mamba2 (未改)
mamba2_mc.py              # 你的 Memory-Cache Mamba2 (未改)
benchmarks/
  model_utils.py          # 统一加载 spec
  wikitext.py             # PPL
  piqa.py                 # PIQA
  longbench.py            # LongBench 子集
  niah.py                 # Needle in a Haystack
  speed.py                # latency / throughput / VRAM
run_benchmark.py          # 入口：跑实验 → results/*.json
plot_and_report.py        # 读 JSON → report/ + figures/
```
