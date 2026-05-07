from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


TRAINING_GEOMETRY_FIELDS: tuple[tuple[str, Any], ...] = (
    ("model_type", str),
    ("block_size", int),
    ("mc_segment_size", int),
    ("mc_max_cached_segments", int),
    ("mc_select_keep_top_k", int),
    ("mc_select_score_threshold", float),
    ("mc_backprop_history", "bool"),
)

RUNTIME_ALIGNMENT_FIELDS: tuple[str, ...] = (
    "mc_segment_size",
    "mc_max_cached_segments",
    "mc_select_keep_top_k",
    "mc_select_score_threshold",
    "mc_backprop_history",
)

TRAINING_ALIGNMENT_FIELDS: tuple[str, ...] = (
    "block_size",
    *RUNTIME_ALIGNMENT_FIELDS,
)


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean metadata value: {value!r}")


def _serialize_meta_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _field_parser(field_name: str):
    for candidate_name, parser in TRAINING_GEOMETRY_FIELDS:
        if candidate_name == field_name:
            return _parse_bool if parser == "bool" else parser
    raise KeyError(f"Unknown training geometry field: {field_name}")


def find_explicit_arg_dests(parser, argv: list[str]) -> set[str]:
    explicit_dests: set[str] = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        for option_string in action.option_strings:
            if any(
                token == option_string or token.startswith(f"{option_string}=")
                for token in argv
            ):
                explicit_dests.add(action.dest)
                break
    return explicit_dests


def read_checkpoint_meta(ckpt_dir: Path) -> dict[str, str]:
    meta_path = ckpt_dir / "meta.txt"
    if not meta_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in meta_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_checkpoint_meta(meta_path: Path, values: Mapping[str, Any]) -> None:
    lines = [
        f"{key}={_serialize_meta_value(value)}"
        for key, value in values.items()
        if value is not None
    ]
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_training_geometry(args) -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    for field_name, _ in TRAINING_GEOMETRY_FIELDS:
        if hasattr(args, field_name):
            geometry[field_name] = getattr(args, field_name)
    return geometry


def parse_saved_training_geometry(meta: Mapping[str, str]) -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    for field_name, _ in TRAINING_GEOMETRY_FIELDS:
        raw_value = meta.get(field_name)
        if raw_value is None:
            continue
        geometry[field_name] = _field_parser(field_name)(raw_value)
    return geometry


def print_training_geometry(geometry: Mapping[str, Any], print_fn=print) -> None:
    if not geometry:
        return
    print_fn("\n=== Saved Training Geometry ===")
    for key in (
        "model_type",
        "block_size",
        "mc_segment_size",
        "mc_max_cached_segments",
        "mc_select_keep_top_k",
        "mc_select_score_threshold",
        "mc_backprop_history",
    ):
        if key in geometry:
            print_fn(f"{key}: {geometry[key]}")


def align_args_with_checkpoint_geometry(
    args,
    meta: Mapping[str, str],
    explicit_dests: set[str],
    fields: Iterable[str],
    print_fn=print,
    context: str = "runtime",
) -> dict[str, Any]:
    saved_geometry = parse_saved_training_geometry(meta)
    saved_model_type = saved_geometry.get("model_type")
    if saved_model_type is not None and getattr(args, "model_type", saved_model_type) != saved_model_type:
        print_fn(
            f"Warning: checkpoint was trained as model_type={saved_model_type}, "
            f"but current run requests model_type={args.model_type}."
        )

    for field_name in fields:
        if field_name not in saved_geometry or not hasattr(args, field_name):
            continue

        saved_value = saved_geometry[field_name]
        current_value = getattr(args, field_name)
        if field_name in explicit_dests:
            if current_value != saved_value:
                print_fn(
                    f"Warning: keeping explicit {field_name}={current_value} even though "
                    f"checkpoint metadata recorded {field_name}={saved_value}."
                )
            continue

        if current_value != saved_value:
            print_fn(
                f"Adopting checkpoint {field_name}={saved_value} for {context} alignment "
                f"(current {field_name}={current_value})."
            )
            setattr(args, field_name, saved_value)
    return saved_geometry
