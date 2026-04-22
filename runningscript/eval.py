#!/usr/bin/env python3
# eval.py
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"[^\w\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def try_load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer, util
        return SentenceTransformer(model_name), util
    except Exception:
        return None, None
    
def bucket_by_length(n):
    if n < 200:
        return "short"
    elif n < 600:
        return "medium"
    elif n < 1200:
        return "long"
    else:
        return "very_long"

def semantic_score(pred: str, golds: list[str], embedder=None, util=None) -> float:
    if not golds:
        return 0.0

    if embedder is None or util is None:
        # fallback: normalized substring overlap heuristic
        p = normalize(pred)
        scores = []
        for g in golds:
            g = normalize(g)
            if not g:
                continue
            if p == g:
                scores.append(1.0)
            elif g in p or p in g:
                scores.append(0.9)
            else:
                scores.append(0.0)
        return max(scores) if scores else 0.0

    texts = [pred] + golds
    vecs = embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    pred_vec = vecs[0:1]
    gold_vecs = vecs[1:]
    sims = util.cos_sim(pred_vec, gold_vecs).tolist()[0]
    return max(sims) if sims else 0.0

def exact_match(pred: str, golds: list[str]) -> int:
    p = normalize(pred)
    for g in golds:
        g2 = normalize(g)
        if not g2:
            continue
        if p == g2:
            return 1
        # for short factual answers, allow strict containment
        if len(g2.split()) <= 4 and g2 in p:
            return 1
    return 0

def safe_get(d, key, default="unknown"):
    v = d.get(key, default)
    return default if v is None else v

def aggregate(rows, key_fields):
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(safe_get(r["sample"], k) for k in key_fields)
        buckets[key].append(r)

    out = []
    for key, items in sorted(buckets.items(), key=lambda x: x[0]):
        n = len(items)
        em = sum(x["em"] for x in items) / n
        sm = sum(x["sm_pass"] for x in items) / n
        avg_sim = sum(x["sim"] for x in items) / n
        out.append({
            **{k: v for k, v in zip(key_fields, key)},
            "n": n,
            "em": round(em, 4),
            "sm": round(sm, 4),
            "avg_sim": round(avg_sim, 4),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL dataset file")
    ap.add_argument("--preds", required=True, help="JSONL predictions file with fields: id, prediction")
    ap.add_argument("--out_dir", default="eval_out")
    ap.add_argument("--sem_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--sem_threshold", type=float, default=0.78)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(args.data)
    preds = load_jsonl(args.preds)

    pred_map = {}
    for p in preds:
        sid = p.get("id")
        if sid is None:
            continue
        pred_map[sid] = p.get("prediction", "")

    embedder, util = try_load_embedder(args.sem_model)

    per_sample = []
    missing = 0

    for s in data:
        sid = s["id"]
        if sid not in pred_map:
            missing += 1
            continue

        pred = str(pred_map[sid])
        golds = s.get("gold_answers", [])
        sim = semantic_score(pred, golds, embedder=embedder, util=util)
        em = exact_match(pred, golds)
        sm_pass = 1 if sim >= args.sem_threshold else 0

        row = {
            "id": sid,
            "prediction": pred,
            "gold_answers": golds,
            "em": em,
            "sim": round(float(sim), 6),
            "sm_pass": sm_pass,
            "length_bucket": bucket_by_length(safe_get(s, "context_tokens", safe_get(s, "approx_tokens", 0))),
            "sample": s,
        }
        per_sample.append(row)

    if not per_sample:
        raise RuntimeError("No matched predictions found. Check ids in preds.jsonl.")

    # overall
    n = len(per_sample)
    overall = {
        "n_scored": n,
        "n_missing": missing,
        "em": round(sum(r["em"] for r in per_sample) / n, 4),
        "sm": round(sum(r["sm_pass"] for r in per_sample) / n, 4),
        "avg_sim": round(sum(r["sim"] for r in per_sample) / n, 4),
        "sem_threshold": args.sem_threshold,
        "sem_model": args.sem_model if embedder is not None else "fallback_substring",
    }

    # grouped reports
    group_specs = [
        ["task_type"],
        ["difficulty"],
        ["eval_mode"],
        ["interference_type"],
        ["interference_strength"],
        ["length_bucket"],
        ["task_type", "difficulty"],
        ["task_type", "eval_mode"],
        ["task_type", "interference_strength"],
    ]

    grouped = {}
    for fields in group_specs:
        grouped["+".join(fields)] = aggregate(per_sample, fields)

    # write outputs
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall,
                "grouped": grouped,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_dir / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for r in per_sample:
            slim = {
                "id": r["id"],
                "prediction": r["prediction"],
                "gold_answers": r["gold_answers"],
                "em": r["em"],
                "sim": r["sim"],
                "sm_pass": r["sm_pass"],
                "task_type": safe_get(r["sample"], "task_type"),
                "difficulty": safe_get(r["sample"], "difficulty"),
                "eval_mode": safe_get(r["sample"], "eval_mode"),
                "interference_type": safe_get(r["sample"], "interference_type"),
                "interference_strength": safe_get(r["sample"], "interference_strength"),
                "approx_tokens": safe_get(r["sample"], "approx_tokens"),
                "context_tokens": safe_get(r["sample"], "context_tokens"),
                "length_bucket": r["length_bucket"],
            }
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'per_sample.jsonl'}")
    if missing:
        print(f"Warning: {missing} samples had no prediction.")

if __name__ == "__main__":
    main()