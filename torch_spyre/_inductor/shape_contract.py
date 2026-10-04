# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The declared shape contract for a dynamic dimension, and where it lives.

A dynamic dimension is declared as three numbers: the range it varies over
(``min``, ``max``) and the step between admissible sizes (``granularity``).
The range is something PyTorch can hold and guard on. The granularity is not,
and that is the whole reason this module exists.

Why the granularity needs its own channel. With ``min=128`` and ``G=64``
declared, the ShapeEnv's divisibility set holds three true facts at once::

    Mod(s, 64)      the one we meant
    Mod(s, 2)       also true
    Mod(s, s//2)    also true

A reader wanting "the" granularity has to pick among them, and picking the
largest is a heuristic rather than a declaration. We shipped a bundle declaring
granularity 128 while its own loop stepped 64 exactly that way: the value was
read from a derived lower bound that an unrelated op's guard had raised during
tracing. Nothing was individually wrong and three correct steps produced a
wrong contract. So the granularity is told, not inferred.

The data moves in two hops, because the two ends do not exist at the same time:

    .to(...)          the real tensor exists, no symbol yet   -> _declared
    lowering          the symbol exists, real inputs reachable -> _by_symbol

``bind_graph_inputs`` is the hop between them. It runs as part of lowering,
where ``V.get_real_inputs()`` still pairs with ``graph.graph_input_names``, so
it can look up the tensor a graph input came from. ``propagate_named_dims``
already moves named dims across the same seam the same way.
"""

import dataclasses

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.utils.weak import WeakTensorKeyDictionary

from . import config
from .errors import Unsupported
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("shape_contract")


@dataclasses.dataclass(frozen=True)
class DimContract:
    """What the caller promised about one varying dimension.

    Attributes:
        min: Smallest admissible size.
        max: Largest admissible size. The device buffer is reserved for this,
            and every tiling decision is made against it.
        granularity: Step between admissible sizes. Admissible sizes are the
            multiples of this inside ``[min, max]``.
    """

    min: int
    max: int
    granularity: int

    def admits(self, size: int) -> bool:
        """Is `size` one of the sizes this contract promised to serve?"""
        return self.min <= size <= self.max and size % self.granularity == 0

    def validate(self) -> None:
        """Reject a contract that cannot be served, naming what to change.

        Checked here rather than at the first compile so the caller finds out
        while they still have the declaration in front of them.

        Raises:
            Unsupported: The declaration is not usable.
        """
        for name, value in (
            ("min", self.min),
            ("max", self.max),
            ("granularity", self.granularity),
        ):
            if not isinstance(value, int) or value <= 0:
                raise Unsupported(
                    f"dynamic dim: {name}={value!r} must be a positive int"
                )

        if self.min > self.max:
            raise Unsupported(
                f"dynamic dim: min={self.min} is greater than max={self.max}"
            )

        # The backend asserts this too, on its side, when it builds the symbol.
        # Catching it here gives a message about the declaration instead of one
        # about an SDSC field.
        if self.max % self.granularity:
            raise Unsupported(
                f"dynamic dim: max={self.max} must be a multiple of "
                f"granularity={self.granularity}. Nearest usable maxima: "
                f"{self.max - self.max % self.granularity} and "
                f"{self.max + self.granularity - self.max % self.granularity}"
            )

        # A size below 8 reaches a row-count branch in the serving plugin's
        # linear wrapper, which under a symbolic size would be a branch on the
        # symbol. It holds today only because real minima are far above it, and
        # a rule that holds by luck is worth writing down.
        if self.min < 8:
            raise Unsupported(
                f"dynamic dim: min={self.min} must be at least 8"
            )

        buckets = self.max // self.granularity
        if buckets > config.max_buckets:
            raise Unsupported(
                f"dynamic dim: max/granularity = {buckets} exceeds "
                f"max_buckets={config.max_buckets}. Raise the granularity, "
                f"lower the max, or raise MAX_BUCKETS"
            )


# Hop 1. Keyed by the real tensor, written eagerly by the transfer API, so it
# has to be weak: a contract must not be the reason a tensor stays alive.
_declared = WeakTensorKeyDictionary()

# Hop 2. Keyed by symbol name, rebuilt per compile by bind_graph_inputs.
_by_symbol: "dict[str, DimContract]" = {}


def record(tensor: torch.Tensor, dim: int, contract: DimContract) -> None:
    """Declare `contract` for `dim` of `tensor`. Called before any tracing.

    Idempotent for the same contract, because a caller may reasonably declare
    the same thing twice. A *different* contract for the same dimension is a
    mistake worth reporting rather than silently resolving one way.

    Args:
        tensor: The real (non-fake) tensor the contract is declared on.
        dim: Which dimension varies.
        contract: The declaration. Validated here.

    Raises:
        Unsupported: The contract is unusable, or it conflicts with one already
            declared for this dimension.
    """
    contract.validate()

    if not 0 <= dim < tensor.dim():
        raise Unsupported(
            f"dynamic dim {dim} is out of range for a rank-{tensor.dim()} tensor"
        )

    per_dim = _declared.get(tensor)
    if per_dim is None:
        per_dim = {}
        _declared[tensor] = per_dim

    existing = per_dim.get(dim)
    if existing is not None and existing != contract:
        raise Unsupported(
            f"dynamic dim {dim} already declared as {existing}, cannot "
            f"redeclare as {contract}"
        )

    per_dim[dim] = contract
    logger.info("[shape-contract] declared dim %d: %s", dim, contract)


def declared_for(tensor: torch.Tensor, dim: int) -> "DimContract | None":
    """The contract declared for `dim` of `tensor`, if any."""
    per_dim = _declared.get(tensor)
    return per_dim.get(dim) if per_dim else None


def for_symbol(name: str) -> "DimContract | None":
    """The contract behind symbol `name`, if one was declared.

    The read side. Callers in lowering and codegen use this instead of
    inferring a granularity from the ShapeEnv.
    """
    return _by_symbol.get(name)


def require_for_symbol(name: str, where: str) -> DimContract:
    """Like `for_symbol`, but refuses rather than returning None.

    Use this wherever proceeding without a declaration would produce a result
    instead of an error. A dimension marked dynamic with no declared contract
    is the one configuration that compiles, reuses a single binary and returns
    a correctly shaped tensor whose rows past the warm-up size are garbage. It
    has to fail here instead.

    Args:
        name: Symbol name.
        where: What is asking, quoted back in the message.

    Raises:
        Unsupported: Nothing was declared for this symbol.
    """
    contract = _by_symbol.get(name)
    if contract is None:
        raise Unsupported(
            f"{where}: dimension symbol {name} is dynamic but no contract was "
            f"declared for it. Declare it with "
            f"to('spyre', dynamic={{dim: dict(min=..., max=..., "
            f"granularity=...)}}). Without min, max and granularity the "
            f"compiled kernel cannot tell which sizes it may serve"
        )
    return contract


def reset() -> None:
    """Drop the per-compile symbol bindings.

    Only hop 2. The declarations in hop 1 belong to the caller's tensors and
    outlive any one compile.
    """
    _by_symbol.clear()


def bind_graph_inputs(graph: GraphLowering) -> None:
    """Re-key declared contracts from tensors onto symbol names.

    This is the hop between the two halves. For each graph input it pairs the
    lowered input with the real tensor it came from, reads any contract
    declared on that tensor, and files it under the symbol sitting in that
    dimension of the input's layout.

    Nothing is inferred. A graph input with no declaration contributes nothing
    here, and whether that is an error is decided by the reader that needs it,
    via `require_for_symbol`.

    Args:
        graph: The graph being lowered.
    """
    if not graph.graph_input_names:
        return

    real_inputs = V.get_real_inputs()

    for name, real_input in zip(graph.graph_input_names, real_inputs):
        if not isinstance(real_input, torch.Tensor):
            continue

        per_dim = _declared.get(real_input)
        if not per_dim:
            continue

        layout = _input_layout(graph, name)
        if layout is None:
            continue

        for dim, contract in per_dim.items():
            if dim >= len(layout.size):
                # The traced graph disagrees with the declaration about rank.
                # Worth saying so rather than binding the wrong dimension.
                logger.warning(
                    "[shape-contract] %s: declared dim %d but the lowered "
                    "input has rank %d, skipping",
                    name,
                    dim,
                    len(layout.size),
                )
                continue

            extent = layout.size[dim]
            symbols = getattr(extent, "free_symbols", None)
            if not symbols:
                # Declared dynamic, lowered concrete. Dynamo specialised it,
                # usually because something branched on the size. Not an error
                # on its own: the kernel is simply static, which is correct.
                logger.info(
                    "[shape-contract] %s dim %d declared dynamic but lowered "
                    "as the constant %s, so this kernel is static",
                    name,
                    dim,
                    extent,
                )
                continue

            if len(symbols) != 1:
                raise Unsupported(
                    f"{name} dim {dim} lowered to the expression {extent} with "
                    f"{len(symbols)} symbols. A declared dynamic dimension must "
                    f"be a single symbol so one contract describes it"
                )

            sym_name = str(next(iter(symbols)))
            previous = _by_symbol.get(sym_name)
            if previous is not None and previous != contract:
                # Two inputs share a symbol, which is what we want for a tied
                # dimension, but they were declared differently. Downstream
                # there is exactly one loop for that symbol, so one of the two
                # declarations would be silently ignored.
                raise Unsupported(
                    f"{name} dim {dim} and an earlier input share the symbol "
                    f"{sym_name} but declare different contracts, {previous} "
                    f"versus {contract}. Declare tied dimensions identically"
                )

            _by_symbol[sym_name] = contract
            logger.info(
                "[shape-contract] bound %s -> %s (from %s dim %d)",
                sym_name,
                contract,
                name,
                dim,
            )


def _input_layout(graph: GraphLowering, name: str):
    """The layout of graph input `name`, or None if it has no simple one.

    Graph inputs are normally ``TensorBox(StorageBox(InputBuffer))``. Anything
    else is not something we can read a dimension off, so the caller skips it
    rather than guessing.
    """
    from torch._inductor.ir import InputBuffer, StorageBox, TensorBox

    box = graph.graph_inputs.get(name)
    if (
        isinstance(box, TensorBox)
        and isinstance(box.data, StorageBox)
        and isinstance(box.data.data, InputBuffer)
    ):
        return box.data.data.layout
    return None
