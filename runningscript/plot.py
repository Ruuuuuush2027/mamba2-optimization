#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_category(df, col, out_path, order=None, title=None):
    grp = df.groupby(col)["sm_pass"].agg(["mean", "count"]).reset_index()
    if order is not None:
        grp[col] = pd.Categorical(grp[col], categories=order, ordered=True)
        grp = grp.sort_values(col)

    plt.figure(figsize=(8, 5))
    plt.bar(grp[col].astype(str), grp["mean"])
    plt.ylim(0, 1)
    plt.ylabel("Semantic Match Accuracy")
    plt.xlabel(col)
    plt.title(title or f"Accuracy by {col}")
    for i, (_, row) in enumerate(grp.iterrows()):
        plt.text(i, min(row["mean"] + 0.02, 0.98), f'n={int(row["count"])}', ha='center', fontsize=9)
    save_fig(out_path)


def plot_accuracy_vs_tokens(df, out_path, bins=10):
    d = df.copy()
    d = d.dropna(subset=["approx_tokens"])
    d["token_bin"] = pd.cut(d["approx_tokens"], bins=bins)

    grp = d.groupby("token_bin", observed=False)["sm_pass"].agg(["mean", "count"]).reset_index()
    labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in grp["token_bin"]]

    plt.figure(figsize=(10, 5))
    plt.plot(labels, grp["mean"], marker="o")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Semantic Match Accuracy")
    plt.xlabel("Approx. Token Length Bin")
    plt.title("Accuracy vs Approx. Token Length")
    for i, row in grp.iterrows():
        plt.text(i, min(row["mean"] + 0.02, 0.98), f'n={int(row["count"])}', ha='center', fontsize=9)
    save_fig(out_path)


def plot_accuracy_by_interference(df, out_path):
    order = ["weak", "medium", "strong"]
    if "interference_strength" not in df.columns:
        return
    plot_accuracy_by_category(
        df,
        "interference_strength",
        out_path,
        order=order,
        title="Accuracy by Interference Strength",
    )


def plot_accuracy_by_task(df, out_path):
    order = [
        "preference", "fact", "habit", "reason", "relation",
        "attribute", "event", "knowledge", "numeric", "shift",
    ]
    plot_accuracy_by_category(
        df,
        "task_type",
        out_path,
        order=order,
        title="Accuracy by Task Type",
    )


def plot_accuracy_by_difficulty(df, out_path):
    order = ["easy", "medium", "hard"]
    plot_accuracy_by_category(
        df,
        "difficulty",
        out_path,
        order=order,
        title="Accuracy by Difficulty",
    )


def plot_accuracy_by_eval_mode(df, out_path):
    order = ["incremental", "full_context"]
    plot_accuracy_by_category(
        df,
        "eval_mode",
        out_path,
        order=order,
        title="Accuracy by Evaluation Mode",
    )


def plot_heatmap(df, out_path):
    if "task_type" not in df.columns or "difficulty" not in df.columns:
        return

    pivot = (
        df.pivot_table(
            index="task_type",
            columns="difficulty",
            values="sm_pass",
            aggfunc="mean",
        )
        .reindex(index=[
            "preference", "fact", "habit", "reason", "relation",
            "attribute", "event", "knowledge", "numeric", "shift"
        ])
        .reindex(columns=["easy", "medium", "hard"])
    )

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Semantic Match Accuracy")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Accuracy Heatmap: Task Type x Difficulty")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)

    save_fig(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per_sample.jsonl from eval.py")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = load_jsonl(args.input)
    if df.empty:
        raise RuntimeError("Empty input file.")

    # normalize columns if missing
    for col in ["task_type", "difficulty", "eval_mode", "interference_strength", "approx_tokens"]:
        if col not in df.columns:
            df[col] = "unknown"

    if "sm_pass" not in df.columns:
        raise RuntimeError("Input must contain sm_pass from eval.py.")

    plot_accuracy_vs_tokens(df, out_dir / "accuracy_vs_tokens.png", bins=10)
    plot_accuracy_by_interference(df, out_dir / "accuracy_by_interference_strength.png")
    plot_accuracy_by_task(df, out_dir / "accuracy_by_task_type.png")
    plot_accuracy_by_difficulty(df, out_dir / "accuracy_by_difficulty.png")
    plot_accuracy_by_eval_mode(df, out_dir / "accuracy_by_eval_mode.png")
    plot_heatmap(df, out_dir / "accuracy_heatmap_task_x_difficulty.png")

    # save a compact report
    report = {
        "n_samples": int(len(df)),
        "overall_sm": float(df["sm_pass"].mean()),
        "by_task": df.groupby("task_type")["sm_pass"].mean().to_dict(),
        "by_difficulty": df.groupby("difficulty")["sm_pass"].mean().to_dict(),
        "by_eval_mode": df.groupby("eval_mode")["sm_pass"].mean().to_dict(),
        "by_interference_strength": df.groupby("interference_strength")["sm_pass"].mean().to_dict(),
    }
    with open(out_dir / "plot_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved plots to: {out_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()