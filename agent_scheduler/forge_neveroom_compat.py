"""
NeverOOM / NO_VRAM inference-tensor compatibility for Forge Neo.

When NeverOOM sets ``VRAMState.NO_VRAM``, Forge casts weights under
``torch.inference_mode()`` via ``empty_like`` + ``copy_`` / ``.to()``.
Those allocations are inference tensors (no version counter). Ops like
``torch.embedding`` then raise:

    RuntimeError: Inference tensors do not track version counter.

This module monkey-patches Forge backend at Agent Scheduler load time so
queued tasks work **without** editing Forge core files (which package
updates would wipe).

Patches (idempotent — safe if core already has equivalent fixes):
  - ``backend.memory_management.cast_to`` — demote returned tensors
  - ``backend.utils.set_attr`` / ``tensor2parameter`` — safe Parameter wrap
  - ``torch.nn.functional.embedding`` — demote weight before embed (covers
    ForgeOperations, GGUF, mixed-precision Embedding paths)
  - expose ``materialize_inference_tensor`` on ``backend.operations``
"""

from __future__ import annotations

from typing import Any, Callable

from .helpers import log

_MARKER = "_agent_scheduler_neveroom_compat"
_APPLIED = False


def materialize_inference_tensor(t: Any) -> Any:
    """Clone inference-mode tensors outside inference mode → normal tensors."""
    if t is None:
        return t
    is_inf = getattr(t, "is_inference", None)
    if not (callable(is_inf) and is_inf()):
        return t
    import torch

    with torch.inference_mode(False):
        return t.clone()


def _mark(fn: Callable) -> Callable:
    setattr(fn, _MARKER, True)
    return fn


def _is_marked(fn: Any) -> bool:
    return bool(getattr(fn, _MARKER, False))


def _patch_cast_to(memory_management) -> None:
    orig = memory_management.cast_to
    if _is_marked(orig):
        return

    def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, *, context=None):
        result = orig(
            weight,
            dtype=dtype,
            device=device,
            non_blocking=non_blocking,
            copy=copy,
            context=context,
        )
        return materialize_inference_tensor(result)

    memory_management.cast_to = _mark(cast_to)


def _patch_utils(utils) -> None:
    import torch

    orig_set_attr = utils.set_attr
    if not _is_marked(orig_set_attr):

        def set_attr(obj, attr, value):
            if hasattr(value, "is_inference") and callable(value.is_inference) and value.is_inference():
                with torch.inference_mode(False):
                    value = value.clone()
            try:
                return utils.set_attr_raw(obj, attr, torch.nn.Parameter(value, requires_grad=False))
            except RuntimeError:
                with torch.inference_mode(False):
                    value = value.clone()
                return utils.set_attr_raw(obj, attr, torch.nn.Parameter(value, requires_grad=False))

        utils.set_attr = _mark(set_attr)

    orig_t2p = utils.tensor2parameter
    if not _is_marked(orig_t2p):

        def tensor2parameter(x):
            if isinstance(x, torch.nn.Parameter):
                return x
            if hasattr(x, "is_inference") and callable(x.is_inference) and x.is_inference():
                with torch.inference_mode(False):
                    x = x.clone()
            return torch.nn.Parameter(x, requires_grad=False)

        utils.tensor2parameter = _mark(tensor2parameter)


def _patch_functional_embedding() -> None:
    import torch

    orig = torch.nn.functional.embedding
    if _is_marked(orig):
        return

    def embedding(input, weight, *args, **kwargs):
        weight = materialize_inference_tensor(weight)
        return orig(input, weight, *args, **kwargs)

    torch.nn.functional.embedding = _mark(embedding)


def _ensure_operations_helper(operations) -> None:
    """Keep ``materialize_inference_tensor`` importable from operations."""
    existing = getattr(operations, "materialize_inference_tensor", None)
    if existing is None or not _is_marked(existing):
        operations.materialize_inference_tensor = _mark(materialize_inference_tensor)


def _try_optional_patches() -> None:
    """utils/operations need optional deps (gguf); retry after full WebUI init."""
    try:
        from backend import utils

        _patch_utils(utils)
    except Exception:
        pass

    try:
        from backend import operations

        _ensure_operations_helper(operations)
    except Exception:
        pass


def apply_neveroom_compat(*, force: bool = False) -> bool:
    """
    Apply all NeverOOM / inference-tensor monkey-patches once.

    Returns True if critical patches are active (``cast_to`` + ``F.embedding``).
    ``utils`` / ``operations`` helpers are retried on later calls when optional
    deps become available after full WebUI startup.
    """
    global _APPLIED
    if _APPLIED and not force:
        _try_optional_patches()
        return True

    try:
        from backend import memory_management
    except ImportError as e:
        log.warning(f"[AgentScheduler] NeverOOM compat skipped (backend not ready): {e}")
        return False

    try:
        _patch_cast_to(memory_management)
        _patch_functional_embedding()
        _try_optional_patches()

        _APPLIED = True
        log.info("[AgentScheduler] NeverOOM/inference-tensor compat patched")
        return True
    except Exception as e:
        log.error(f"[AgentScheduler] NeverOOM compat failed: {e}")
        return False


def ensure_patched() -> bool:
    """Idempotent entry point — safe to call on every queued task."""
    return apply_neveroom_compat()


def prepare_for_queued_inference() -> None:
    """
    Scheduler-side mitigation before ``__execute_ui_task``.

    Re-applies patches if needed. NeverOOM still owns VRAM state; we only
    guarantee inference-tensor demotion hooks are installed.
    """
    ensure_patched()
