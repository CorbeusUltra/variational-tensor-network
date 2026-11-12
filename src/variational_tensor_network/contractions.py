"""Reuse contraction plans for repeated tensor networks."""

import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import opt_einsum as oe
from opt_einsum.contract import ContractExpression
from torch import Tensor

from ._utils.path import find_project_root

CACHE_DIR = find_project_root(__file__) / "data" / "contraction_paths"
type Shapes = tuple[tuple[int, ...], ...]


@lru_cache(maxsize=128)
def _get_expression(equation: str, shapes: Shapes, strategy: str, memory_limit: int) -> ContractExpression:
    key = {
        "schema": 1,
        "opt_einsum_version": oe.__version__,
        "equation": equation,
        "shapes": [list(shape) for shape in shapes],
        "strategy": strategy,
        "memory_limit": memory_limit,
        "use_blas": True,
    }
    key_bytes = json.dumps(key, sort_keys=True).encode("utf-8")
    plan_file = CACHE_DIR / f"{sha256(key_bytes).hexdigest()}.json"
    if plan_file.is_file():
        saved = json.loads(plan_file.read_text(encoding="utf-8"))
        if saved["key"] != key:
            raise ValueError(f"Contraction cache key mismatch in {plan_file}.")
        optimize = [tuple(step) for step in saved["path"]]
    else:
        optimize = strategy
    path, info = oe.contract_path(equation, *shapes, shapes=True, optimize=optimize, memory_limit=memory_limit)
    if len(path) != len(shapes) - 1 or any(len(step) != 2 or step[0] == step[1] for step in path):
        raise ValueError("Expected a complete pairwise contraction path.")
    if info.largest_intermediate > memory_limit:
        raise MemoryError(f"Largest intermediate: {info.largest_intermediate} elements, limit: {memory_limit}.")
    expression = oe.contract_expression(equation, *shapes, optimize=path)
    if isinstance(optimize, str):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", dir=CACHE_DIR, encoding="utf-8", delete=False) as stream:
            json.dump({"key": key, "path": path}, stream, indent=2)
        Path(stream.name).replace(plan_file)
    return expression


def cached_einsum(equation: str, *operands: Tensor, strategy: str = "auto-hq", memory_limit: int = 2**22) -> Tensor:
    """Contract current tensors with an expression cached by operand shapes."""
    shapes = tuple(tuple(tensor.shape) for tensor in operands)
    expression = _get_expression(equation, shapes, strategy, memory_limit)
    return expression(*operands, backend="torch")
