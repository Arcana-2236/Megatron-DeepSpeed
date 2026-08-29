# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Callable, Dict, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import (
    get_emerging_optimizers_version,
    get_pg_rank,
    get_pg_size,
    is_emerging_optimizers_min_version,
    log_single_rank,
)

from .optimizer_config import ParamKey, ParamPredicate

try:
    from emerging_optimizers import registry
    from emerging_optimizers import utils as eo_utils
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object


logger = logging.getLogger(__name__)

# newton_schulz_tp() gained the use_syrk kwarg in emerging_optimizers 0.4.0. Earlier releases
# expose use_syrk on the non-TP newton_schulz() only, so 0.3.x still rejects it here. Spelled
# ".dev0" so pre-release builds of that line are accepted too, matching how the TE minimums
# elsewhere in the tree are written.
_SYRK_MIN_EO_VERSION = "0.4.0.dev0"


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is not None:
        eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    else:
        eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
            eopt_name, config, model_chunks, pg_collection
        )
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _is_muon_excluded(param):
    """True for parameters that should use the scalar optimizer instead of Muon."""
    return not getattr(param, 'use_muon', True) or _is_nonlinear_or_embedding(param)


def _get_qkv_split_shapes(model_cfg) -> list[int]:
    """Compute QKV split shapes from model config."""
    query_projection_size = (
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels
    )
    if getattr(model_cfg, 'attention_output_gate', False):
        return [
            query_projection_size,
            query_projection_size,
            model_cfg.kv_channels,
            model_cfg.kv_channels,
        ]
    return [query_projection_size, model_cfg.kv_channels, model_cfg.kv_channels]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


# tp_mode="auto" selects tp_mode per weight (Dense/GTP weights only)
_AUTO_TP_MODES = ("duplicated", "distributed")


@dataclass(frozen=True)
class HardwareProfile:
    """HW spec for different GPU. Use for cost model when selecting TP mode.
    Bandwidths are UNIDIRECTIONAL."""

    bf16_peak_tflops: float  # dense, fp32_matmul_prec = "medium" for now
    bw_intra_gbps: float  # collectives staying inside one NVLink domain
    bw_inter_gbps: float  # collectives crossing domains, over the fabric
    alpha_coll_us: float  # Latency term, Fixed cost of ONE collective, independent of payload.


_PROFILES = {
    # Keys are matched as a substring of the reported device name ("NVIDIA GB200").
    # Both bandwidths are PER GPU. Only run on GB200 & GB300 for now.
    # TODO: May need to add other HW Spec
    # alpha_coll_us is the 64-rank value; it is what dense GTP uses, which is all tp_mode
    # "auto" currently decides. A 2-rank group measures 68us, so this over-prices latency
    # if auto is ever extended to expert weights at EGTP=2.
    "GB200": HardwareProfile(
        bf16_peak_tflops=2500.0, bw_intra_gbps=900.0, bw_inter_gbps=100.0, alpha_coll_us=134.0
    ),
    "GB300": HardwareProfile(
        bf16_peak_tflops=2500.0, bw_intra_gbps=900.0, bw_inter_gbps=100.0, alpha_coll_us=134.0
    ),
}


def _hardware_profile() -> Optional[HardwareProfile]:
    """Profile for the local GPU, or None when the hardware is not in the registry."""
    try:
        name = torch.cuda.get_device_properties(0).name
    except Exception:  # noqa: BLE001 - no CUDA device: fall back, do not fail
        return None
    return next((prof for key, prof in _PROFILES.items() if key in name), None)


def _select_tp_mode(
    m: int,
    n: int,
    group_size: int,
    steps: int,
    use_syrk: bool,
    elem_size: int,
    communication_crosses_domain: bool,
    profile: Optional[HardwareProfile] = None,
    candidates: tuple[str, ...] = _AUTO_TP_MODES,
) -> str:
    """Cost model for per weight tp_mode selection. Mirrors the op sequence in
    scaled_orthogonalize_fn_with_gtp_remat -- keep in sync.
    """
    cost = _tp_mode_costs(
        m, n, group_size, steps, use_syrk, elem_size,
        communication_crosses_domain, profile, candidates,
    )
    return "duplicated" if cost is None else min(candidates, key=cost.get)


def _tp_mode_costs(
    m: int,
    n: int,
    group_size: int,
    steps: int,
    use_syrk: bool,
    elem_size: int,
    communication_crosses_domain: bool,
    profile: Optional[HardwareProfile] = None,
    candidates: tuple[str, ...] = _AUTO_TP_MODES,
) -> Optional[Dict[str, float]]:
    """Per-mode estimated seconds for one orthogonalization, or None without a profile.

    The layout uses these values directly: LPT balances rank TIME, and under tp_mode="auto"
    one rank holds a mix of modes, so FLOPs alone are not comparable across them --
    communication and launch latency are 60-80% of `distributed`'s total.
    """
    min_dim, max_dim = min(m, n), max(m, n)
    # dist no longer forces the transpose: it orients so the Gram lands on min(m, n),
    # transposing when m > n and resharding via all-to-all when m < n. So both modes
    # iterate on the same [min, max] matrix and share the replicated min^3 term; dist
    # differs only in sharding the min^2*max terms over the group.
    max_partitioned = max_dim // group_size
    gram = 1 if use_syrk else 2  # SYRK halves the two gram ops
    # Per NS step: gram X@X.T + gram A@A + GEMM B@X.
    flops = {
        "duplicated": steps
        * (gram * (min_dim * min_dim * max_dim + min_dim**3) + 2 * min_dim * min_dim * max_dim),
        "distributed": steps
        * (
            gram * (min_dim * min_dim * max_partitioned + min_dim**3)
            + 2 * min_dim * min_dim * max_partitioned
        ),
    }
    if profile is None:
        # Unregistered hardware: no bandwidth or latency numbers, so no cost can be formed.
        # Callers fall back to their own default rather than guess; selecting on FLOPs alone
        # would always answer distributed, committing to `steps` Gram all-reduces per weight
        # with nothing to price them against.
        return None

    ring_fraction = (
        group_size - 1
    ) / group_size  # ring: each rank moves (group_size-1)/group_size of the buffer
    # The two all-to-alls of the m < n path are omitted: they are
    # (max_dim/min_dim)/(group_size*steps) of the Gram all-reduce volume, under 1.5% for
    # every shape in this model, and only matter if max/min exceeds group_size*steps.
    num_bytes = {
        "duplicated": m * n * elem_size * ring_fraction,  # one all-gather
        "distributed": steps
        * 2
        * min_dim
        * min_dim
        * elem_size
        * ring_fraction,  # gram all-reduce per step, Gram is [min, min]
    }
    # Launch/link latency, which the bandwidth term alone cannot capture: dup issues ONE
    # all-gather while dist issues one Gram all-reduce per step, plus the zero-payload
    # distributed_normalize_p2 scalar all-reduce, plus the two all-to-alls when it reshards.
    # Their volume is negligible but each is a full collective launch. At steps=16 that is
    # 17-19 launches against dup's 1, which decides every small shape.
    num_collectives = {
        "duplicated": 1,
        "distributed": steps + 1 + (2 if m < n else 0),
    }
    bw = (profile.bw_inter_gbps if communication_crosses_domain else profile.bw_intra_gbps) * 1e9
    peak = profile.bf16_peak_tflops * 1e12
    alpha = profile.alpha_coll_us * 1e-6
    cost = {
        mode: flops[mode] / peak + num_bytes[mode] / bw + num_collectives[mode] * alpha
        for mode in candidates
    }
    return cost


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed", "auto"] = "duplicated",
        use_syrk: bool = False,
        expert_batch_size: int = 1,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if expert_batch_size < 1:
            raise ValueError(f"expert_batch_size must be at least 1, got {expert_batch_size}")
        if use_syrk and not is_emerging_optimizers_min_version(_SYRK_MIN_EO_VERSION):
            raise ValueError(
                f"use_syrk requires emerging_optimizers >= {_SYRK_MIN_EO_VERSION}, but "
                f"{get_emerging_optimizers_version()} is installed. Upgrade "
                "emerging_optimizers or drop --muon-use-syrk."
            )

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
            tp_mode_this_group: str = tp_mode,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            # Only forward the kwarg when enabled; older emerging_optimizers do not
            # accept it at all, and __init__ has already rejected use_syrk on those.
            ns_kwargs = {"use_syrk": True} if use_syrk else {}
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode_this_group == "blockwise" else tp_mode_this_group,
                **ns_kwargs,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        # For the tp_mode="auto" cost model (_resolve_tp_mode / _select_tp_mode).
        self.num_ns_steps = num_ns_steps
        self.use_syrk = use_syrk
        self.expert_batch_size = expert_batch_size
        self._chunk_cache: Dict[int, tuple] = {}  # param-group index -> (len, chunks)
        if expert_batch_size > 1:
            log_single_rank(
                logger, logging.INFO, f"muon expert_batch_size={expert_batch_size}"
            )
        self.elem_size = 2 if fp32_matmul_prec == "medium" else 4  # bf16 vs tf32/fp32
        self._tp_mode_cache: Dict[tuple, str] = {}
        self._hw_profile = _hardware_profile() if tp_mode == "auto" else None

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    @staticmethod
    def _all_gather_tensor(t, group, dim):
        """All-gather equal-size shards of ``t`` over ``group`` and concat along ``dim``."""
        shards = [torch.empty_like(t) for _ in range(get_pg_size(group))]
        torch.distributed.all_gather(shards, t.contiguous(), group)
        return torch.cat(shards, dim=dim)

    @staticmethod
    def _all_to_all_tensor(t, group, scatter_dim, gather_dim):
        """Re-shard ``t`` over ``group``: split along ``scatter_dim``, concat along ``gather_dim``.

        Moves a tensor sharded on ``gather_dim`` to one sharded on ``scatter_dim``. With
        ``scatter_dim=1, gather_dim=0`` a row-sharded ``[M/G, N]`` becomes a column-sharded
        ``[M, N/G]``; swapping the two arguments inverts it. Both dims must divide ``G``.
        """
        group_size = get_pg_size(group)
        send = [c.contiguous() for c in t.chunk(group_size, dim=scatter_dim)]
        recv = [torch.empty_like(c) for c in send]
        torch.distributed.all_to_all(recv, send, group)
        return torch.cat(recv, dim=gather_dim)

    @staticmethod
    def _strip_pad(t, pad_length):
        """Drop the trailing ``pad_length`` rows of dim 0 (no-op if ``pad_length == 0``)."""
        return t[:-pad_length] if pad_length else t

    @staticmethod
    def _restore_pad(t, pad_length):
        """Re-append ``pad_length`` zero rows to dim 0 (no-op if ``pad_length == 0``)."""
        return torch.nn.functional.pad(t, (0, 0, 0, pad_length)) if pad_length else t

    def _resolve_tp_mode(self, m: int, n: int, group_size: int) -> str:
        """Cached per-shape mode for tp_mode="auto", dense (GTP) weights only.

        communication_crosses_domain=False always: this is only called for dense weights
        (see scaled_orthogonalize_fn_with_gtp_remat), and GTP stays inside one NVLink domain.
        """
        key = (m, n, group_size)
        if key not in self._tp_mode_cache:
            self._tp_mode_cache[key] = _select_tp_mode(
                m,
                n,
                group_size,
                self.num_ns_steps,
                self.use_syrk,
                self.elem_size,
                communication_crosses_domain=False,
                profile=self._hw_profile,
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"muon tp_mode=auto (dense): ({m}, {n}) group_size={group_size} -> "
                f"{self._tp_mode_cache[key]}",
            )
        return self._tp_mode_cache[key]

    def _expert_batch_key(self, p: torch.Tensor) -> Optional[tuple]:
        """Key grouping expert weights that can share one all-gather; None to not batch."""
        if self.expert_batch_size <= 1 or not self.pg_collection:
            return None
        if not (getattr(p, 'expert_tp', False) and getattr(p, 'is_gtp_weight_remat', False)):
            return None
        if get_pg_size(self.pg_collection.expt_gtp_remat) <= 1:
            return None
        # Expert weights are always "duplicated" (see scaled_orthogonalize_fn_with_gtp_remat),
        # and only "duplicated" all-gathers per weight, so only it has a collective to share.
        return (tuple(p.shape), p.dtype)

    def _chunks_for(self, index: int, group: Dict[str, Any]) -> list:
        """Units of work for *group*: ``[p]`` normally, or a batch of expert weights.

        Batches only *adjacent* same-key params, which keeps a batch within one MoE layer.
        Cached on the group's index: the param list is fixed after construction, and the
        index is stable for the optimizer's life (``id()`` would be recyclable).
        """
        params = group["params"]
        cached = self._chunk_cache.get(index)
        if cached is not None and cached[0] == len(params):
            return cached[1]

        chunks: list = []
        for key, run in groupby(params, self._expert_batch_key):
            run = list(run)
            width = self.expert_batch_size if key is not None else 1
            chunks += [run[i : i + width] for i in range(0, len(run), width)]

        batched = [c for c in chunks if len(c) > 1]
        if batched:
            log_single_rank(
                logger,
                logging.INFO,
                f"muon expert batching: {len(batched)} batches, "
                f"widths={sorted({len(c) for c in batched})}, "
                f"{len(params)} params -> {len(chunks)} units",
            )
        self._chunk_cache[index] = (len(params), chunks)
        return chunks

    def _orthogonalize_expert_batch(self, params: list, grads: list) -> list:
        """Orthogonalize same-shaped expert momenta over one shared GTP all-gather.

        Stacks to ``[B, M/G, N]`` and gathers on dim 1 -- the shard dim, moved from 0 by
        the stack -- then runs one 3-D Newton-Schulz over the batch and re-shards. Cutting
        the kernel count is what removes the per-boundary GPU stall; the shared collective
        alone does not (measured, job 2665765).
        """
        gtp_group = self.pg_collection.expt_gtp_remat
        tp_group = self.pg_collection.expt_tp
        # The dense duplicated path strips/restores GTP alignment padding; this one does
        # not, so refuse a padded weight rather than orthogonalize the pad rows as data.
        assert all(getattr(p, "pad_length", 0) == 0 for p in params), (
            "muon expert batching does not support GTP alignment padding"
        )
        shard = grads[0].shape[0]
        lo = get_pg_rank(gtp_group) * shard
        rows = shard * get_pg_size(gtp_group)

        torch.cuda.nvtx.range_push(f"muon_ns_x{len(params)}:({rows}, {grads[0].shape[1]})")
        gathered = self._all_gather_tensor(torch.stack(grads), gtp_group, 1)

        if get_pg_size(tp_group) == 1:
            # GTP is already undone by the gather, so the batched call must not re-enter
            # the TP path: partition_dim=None.
            orth = self.scaled_orthogonalize_fn(
                gathered, tp_group, None, tp_mode_this_group="duplicated"
            )
            out = [orth[i, lo : lo + shard].contiguous() for i in range(len(params))]
        else:
            # newton_schulz_tp cannot express a TP gather on dim+1 for 3-D input, so the
            # batch shares the collective but orthogonalizes per expert.
            pdim = getattr(params[0], "partition_dim", None)
            pdim = None if pdim == -1 else pdim
            out = [
                self.scaled_orthogonalize_fn(
                    g, tp_group, pdim, tp_mode_this_group="duplicated"
                )[lo : lo + shard].contiguous()
                for g in gathered
            ]
        torch.cuda.nvtx.range_pop()
        return out

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Optimizer step; identical to the base class when ``expert_batch_size == 1``.

        Above 1, expert weights are orthogonalized in batches so one all-gather and one
        Newton-Schulz serve the batch. All per-weight work -- weight decay, momentum, the
        update -- is unchanged.
        """
        if closure is not None:
            raise ValueError("closure is not supported")

        for index, group in enumerate(self.param_groups):
            self._init_group(group)
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            lr, momentum = group["lr"], group["momentum"]

            for chunk in self._chunks_for(index, group):
                have = [p for p in chunk if p.grad is not None]
                if not have:
                    continue
                # Chunk width sets the collective's shape, so it must be rank-invariant:
                # a partially-gradded batch would desync the all-gather rather than error.
                assert len(have) == len(chunk) or len(chunk) == 1, (
                    "muon expert batch has partial gradients; chunking is not rank-invariant"
                )

                grads = []
                for p in chunk:
                    self._apply_weight_decay_inplace(p, p.grad, lr, group["weight_decay"])
                    buf = self.state[p]["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - momentum)
                    grads.append(p.grad.lerp(buf, momentum) if self.nesterov else buf)

                with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    if len(chunk) > 1:
                        orth_grads = self._orthogonalize_expert_batch(chunk, grads)
                    else:
                        orth_grads = [self.orthogonalize(chunk[0], grads[0], **group_kwargs)]

                for p, orth_grad in zip(chunk, orth_grads):
                    self.pre_weight_update_fn_inplace(p, orth_grad)
                    p.add_(orth_grad, alpha=-lr)
                    self.post_weight_update_fn_inplace(p)

        return None

    def scaled_orthogonalize_fn_with_gtp_remat(self, p, grad, tp_group, partition_dim):
        """Orthogonalize a (possibly GTP-sharded) momentum, then reshard.

        When GTP is inactive this is a plain passthrough to ``scaled_orthogonalize_fn``.
        Otherwise, ``mode`` (``self.tp_mode``, or resolved per-weight when
        ``self.tp_mode == "auto"``) controls how GTP sharding is handled:

        - **blockwise**: orthogonalize the local GTP shard independently, no collective.
        - **duplicated**: all-gather over GTP, run whole-matrix NS (TP-aware), reshard.
        - **distributed**: distribute NS over GTP via small-Gram all-reduce. When both
          GTP and TP are active, NS is distributed over the larger group to minimize
          redundant compute; the smaller group is all-gathered beforehand.

        GTP_remat may pad dim 0 for alignment (see gtp_remat_shard_dim0). blockwise and
        duplicated strip the padding before calling scaled_orthogonalize_fn and restore it
        after, since every rank holds a uniform, fully-reconstructed tensor by then.
        distributed does not: it stays row-sharded through its own collective, where
        stripping isn't safe (known limitation).
        """
        # TODO: Clean up code that determines if parameter is a MoE layer and which TP group to use
        is_expert = getattr(p, 'expert_tp', False)
        gtp_remat_group = (
            (self.pg_collection.expt_gtp_remat if is_expert else self.pg_collection.gtp_remat)
            if self.pg_collection
            else None
        )

        # Parameters with is_gtp_weight_remat=False are not sharded along the
        # GTP process group, and do not require all-gathering prior to
        # orthogonalization.
        gtp_active = (
            gtp_remat_group is not None
            and get_pg_size(gtp_remat_group) > 1
            and getattr(p, 'is_gtp_weight_remat', False)
        )
        gtp_remat_size = get_pg_size(gtp_remat_group) if gtp_active else 1

        mode = self.tp_mode
        if mode == "auto":
            # Scoped to dense (GTP) weights for now; expert weights keep today's default.
            mode = (
                self._resolve_tp_mode(p.shape[0] * gtp_remat_size, p.shape[1], gtp_remat_size)
                if gtp_active and not is_expert
                else "duplicated"
            )

        if not gtp_active:
            return self.scaled_orthogonalize_fn(
                grad, tp_group, partition_dim, tp_mode_this_group=mode
            )

        gtp_rank = get_pg_rank(gtp_remat_group)
        pad_length = getattr(p, 'pad_length', 0)

        if mode == "blockwise":
            # Local block NS on this rank's GTP row-shard (shape [M/gtp_remat_size, K]):
            # partition_dim=None makes scaled_orthogonalize_fn run a plain Newton-Schulz on
            # the shard with no GTP/TP collective. pad_length can exceed one shard's row
            # count, so only the overlap between this rank's shard and the trailing padded
            # rows of the full tensor is this rank's own padding.
            shard_size = grad.size(0)
            ranks_from_end = gtp_remat_size - 1 - gtp_rank
            local_pad_length = min(shard_size, max(0, pad_length - ranks_from_end * shard_size))
            if local_pad_length == shard_size:
                # Entirely padding: grad is exact zero, and NS(0) = 0.
                return torch.zeros_like(grad)
            result = self.scaled_orthogonalize_fn(
                self._strip_pad(grad, local_pad_length), tp_group, None, tp_mode_this_group=mode
            )
            return self._restore_pad(result, local_pad_length)

        if mode == "duplicated":
            # All-gather over GTP (dim 0), strip/restore padding exactly (every rank now
            # holds the same padded tensor), orthogonalize the whole matrix
            # (scaled_orthogonalize_fn handles any TP sharding per tp_mode), reshard dim 0.
            gathered_grad = self._all_gather_tensor(grad, gtp_remat_group, 0)
            result = self.scaled_orthogonalize_fn(
                self._strip_pad(gathered_grad, pad_length),
                tp_group,
                partition_dim,
                tp_mode_this_group=mode,
            )
            result = self._restore_pad(result, pad_length)
            reshard_size = result.size(0) // gtp_remat_size
            return result[gtp_rank * reshard_size : (gtp_rank + 1) * reshard_size].contiguous()

        # distributed: NS via the small-Gram all-reduce (no redundant full-matrix NS).
        # A momentum with both TP and GTP as sharding axes takes two communication steps: an
        # all-gather that eliminates one axis, then the Gram all-reduce that distributes NS over
        # the other. With GTP as the only sharding axis, the Gram all-reduce is the only
        # communication needed. partition_dim is what says whether TP is a sharding axis here,
        # the same signal scaled_orthogonalize_fn and newton_schulz_tp key off.
        #
        # GTP_remat's alignment padding is not corrected for here -- see the class docstring.
        needs_two_step_communication = (
            partition_dim is not None and tp_group is not None and get_pg_size(tp_group) > 1
        )

        if not needs_two_step_communication:
            # GTP is the only sharding axis: distribute NS over it on the local dim-0 row shard.
            #
            # partition_dim=0 forces transpose=True in newton_schulz_tp, so NS runs on
            # [N, M/G] and the Gram is [N, N]. That is only right when N is the SHORT dim;
            # when M < N it puts the Gram, and its replicated N^3 term, on the LONG dim --
            # the waste newton_schulz_tp's own docstring warns about. So pick the
            # orientation that lands the Gram on min(M, N), the same rule newton_schulz
            # already applies to non-TP input via transpose = x.size(-2) > x.size(-1):
            #   M > N  ->  partition_dim=0, transpose, Gram [N, N]  (unchanged)
            #   M < N  ->  all-to-all to column-sharded, partition_dim=1, Gram [M, M]
            # Only N % G matters: the return all-to-all splits dim 0 of [M, N/G], and
            # M = grad.size(0) * G is divisible by construction. Indivisible N falls back.
            full_rows = grad.size(0) * gtp_remat_size
            cols = grad.size(1)
            if full_rows < cols and cols % gtp_remat_size == 0:
                x = self._all_to_all_tensor(grad, gtp_remat_group, scatter_dim=1, gather_dim=0)
                x = self.scaled_orthogonalize_fn(x, gtp_remat_group, 1, tp_mode_this_group=mode)
                return self._all_to_all_tensor(x, gtp_remat_group, scatter_dim=0, gather_dim=1)
            return self.scaled_orthogonalize_fn(
                grad, gtp_remat_group, partition_dim=0, tp_mode_this_group=mode
            )

        # GTP + TP: distributed NS can only operate over one (group, dim) at a
        # time. Distribute over the larger group so that the NS GEMMs are sharded
        # across more ranks (less redundant compute), and all-gather the smaller
        # group to eliminate its sharding beforehand.
        tp_size = get_pg_size(tp_group)
        if gtp_remat_size >= tp_size:
            smaller_group, smaller_dim = tp_group, partition_dim
            larger_group, larger_dim = gtp_remat_group, 0
        else:
            smaller_group, smaller_dim = gtp_remat_group, 0
            larger_group, larger_dim = tp_group, partition_dim

        gathered_grad = self._all_gather_tensor(grad, smaller_group, smaller_dim)
        orthogonalized_grad = self.scaled_orthogonalize_fn(
            gathered_grad, larger_group, larger_dim, tp_mode_this_group=mode
        )
        shard_size = orthogonalized_grad.size(smaller_dim) // get_pg_size(smaller_group)
        reshard_rank = get_pg_rank(smaller_group)
        return orthogonalized_grad.narrow(
            smaller_dim, reshard_rank * shard_size, shard_size
        ).contiguous()

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. It is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            qkv_split_shapes = getattr(p, "qkv_split_shapes", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
            if qkv_split_shapes is None:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is not set")
            qkv_split_dim = sum(qkv_split_shapes)
            if grad_shape[0] % qkv_split_dim != 0:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: grad_shape={tuple(grad_shape)}, "
                    f"split_shapes={qkv_split_shapes}"
                )
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // qkv_split_dim
            qkv_grads = torch.split(
                grad.view(num_query_groups, qkv_split_dim, -1), qkv_split_shapes, dim=1
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn_with_gtp_remat(p, g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn_with_gtp_remat(p, grad, tp_group, partition_dim)
        return grad


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", "distributed", or "auto").
        use_syrk: Whether to use the Triton SYRK kernel for the Gram matrix in
            Newton-Schulz. Requires emerging_optimizers >= 0.4.0.
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed", "auto"] = "duplicated",
        use_syrk: bool = False,
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
            use_syrk=use_syrk,
        )
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(predicate=ParamPredicate(name="muon_excluded", fn=_is_muon_excluded)): {
                    'optimizer': 'adam'
                }
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(predicate=ParamPredicate(name="muon_excluded", fn=_is_muon_excluded)): {
                    'optimizer': 'adam'
                }
            },
        ),
    }
)

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name)
        )
