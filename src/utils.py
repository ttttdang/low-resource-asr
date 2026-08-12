"""
Shared utilities for reproducibility, logging, checkpointing, and config paths.

Functions:
  set_seed(seed)                                - fix all random sources
  seed_worker(worker_id)                        - DataLoader worker_init_fn for deterministic workers
  get_logger(name, log_dir, log_file)           - console + file logger
  save_checkpoint(...)                          - persist model + optimizer state
  load_checkpoint(...)                          - restore state, return (epoch, step, metric)
  expand_user_paths(cfg)                        - in-place expand `~` in OmegaConf string values
  _peft_inner(model), _adapter_inner(model)     - PEFT helper functions
==================================================================
"""

import os
import random
import logging
import shutil
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from peft import PeftModel
from safetensors.torch import load_file
from models.adapter_fusion import LightWeightFusion


# ==================================================================
# SEED
# ==================================================================
def set_seed(seed: int) -> None:
    """Fix all sources of randomness for reproducible experiments."""

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed_all(seed)

    # CuDNN — disable non-deterministic ops and auto-tuning
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for deterministic worker RNG state.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==================================================================
# LOGGING
# ==================================================================
def get_logger(
    name     : str,
    log_dir  : str,
    log_file : str,
) -> logging.Logger:
    """
    Build a logger that writes to both stderr and a file.

    Args:
        name     : Logger name is experiment run name
        log_dir  : Directory to write the log file into
        log_file : Log filename e.g. "train.log"
    """

    # Create log_dir if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Get logger and set level to INFO
    logger = logging.getLogger(name)
    if logger.handlers:                  
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False                

    # Build a shared formatter
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler to stderr
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler to log_dir/log_file
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# ==================================================================
# CONFIG PATH EXPANSION
# ==================================================================
def expand_user_paths(cfg) -> None:
    """Recursively expand `~` in any string config value, in-place.

    OmegaConf stores `~/...` as a literal string and Python's path functions
    don't expand `~` either — only os.path.expanduser does. Without this,
    os.makedirs(cfg.log_dir) creates a directory literally named `~`.
    """
    if isinstance(cfg, DictConfig):
        for k, v in cfg.items():
            if isinstance(v, str) and v.startswith("~"):
                cfg[k] = os.path.expanduser(v)
            elif isinstance(v, (DictConfig, ListConfig)):
                expand_user_paths(v)
    elif isinstance(cfg, ListConfig):
        for i, v in enumerate(cfg):
            if isinstance(v, str) and v.startswith("~"):
                cfg[i] = os.path.expanduser(v)
            elif isinstance(v, (DictConfig, ListConfig)):
                expand_user_paths(v)

# ==================================================================
# CHECKPOINTING
# ==================================================================
def _peft_inner(model):
    """Detects peft-wrapped models (e.g. LoRA from apply_lora) for save/load dispatch.
    """
    if isinstance(model, PeftModel):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, PeftModel):
        return inner
    return None

def _adapter_inner(model):
    """Return the adapter-capable inner HF model if any, else None.
    """
    if hasattr(model, "adapter_summary"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "adapter_summary"):
        return inner
    return None

def _composition_from_spec(spec):
    """
    Reconstruct an active-adapters composition object from a serialized spec.
    
    spec = {"kind": "single" | "stack", "adapters": [str, ...]}
    Returns: a value suitable for direct assignment to `model.active_adapters`
            - a str for "single", a Stack object for "stack".
    """
    from adapters.composition import Stack
    
    kind    = spec["kind"]
    names   = spec["adapters"]
    
    if kind == "single":
        return names[0]
    # Iterable unpacking since Stack accepts positional args, e.g. (Stack("es", "ast"))
    if kind == "stack":
        return Stack(*names) # without *, we'd pass Stack(["es", "ast"])
    raise ValueError(f"Unknown composition kind: {kind!r}")

def save_checkpoint(
    model     : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : torch.optim.lr_scheduler.LRScheduler,
    epoch     : int,
    step      : int,
    metric    : float,                    # validation WER at this checkpoint
    path      : str,
    adapter_name: str | None = None,
    adapter_init_from: str | None = None,
    active_composition: dict | None = None,
) -> None:
    """Save model, optimizer, and scheduler state to path."""

    # Create parent directory if needed
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    peft_model = _peft_inner(model)
    adapter_model = _adapter_inner(model)
    
    # ==================================================================
    # PEFT PATH
    # ==================================================================
    if peft_model is not None:
        # PEFT path: write directory at path (rewriting .pt to no extension)
        # writes:  save_dir/adapter/{adapter_model.safetensors, adapter_config.json}
        #          save_dir/training_state.pt
        save_dir = path[:-3] if path.endswith(".pt") else path
        staging  = save_dir + ".tmp"

        # 1. Clean any stale staging from a previous interrupted save
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)

        # 2. Write everything into staging
        peft_model.save_pretrained(os.path.join(staging, "adapter"))
        torch.save({
            "epoch":  epoch, "step": step, "metric": metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "adapter_name": adapter_name,
            "adapter_init_from": adapter_init_from,
        }, os.path.join(staging, "training_state.pt"))

        # 3. Remove old save_dir, rename staging into its place
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.rename(staging, save_dir)
    
    # ==================================================================
    # ADAPTER PATH
    # ==================================================================
    elif adapter_model is not None:
        # writes:  save_dir/adapter/{pytorch_adapter.bin, adapter_config.json}
        #          save_dir/training_state.pt
        save_dir = path[:-3] if path.endswith(".pt") else path
        staging  = save_dir + ".tmp"
        
        # 1. Clean any stale staging from a previous interrupted save
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)
        
        # 2. Branch on what artifact to persist
        # Lightweight fusion: fusion-layer weights via a custom writer (see below).
        # Everything else (single adapter, stack): save the single trainable adapter via save_adapter
        if active_composition is not None and active_composition["kind"] == "lightweight_fuse":
            # custom save — library save_adapter_fusion would persist a config that
            # rebuilds Q/K at dense_size on load → shape mismatch with our d_k weights.
            os.makedirs(os.path.join(staging, "adapter"), exist_ok=True)
            fusion_state = {
                name: mod.state_dict()
                for name, mod in adapter_model.named_modules()
                if isinstance(mod, LightWeightFusion)
            }
            assert fusion_state, "lightweight_fuse: captured 0 fusion modules — LightWeightFusion class/name mismatch?"
            torch.save(
                {"d_k": active_composition.get("d_k"),
                "value": active_composition.get("value"),
                "fusion_state": fusion_state},
                os.path.join(staging, "adapter", "lightweight_fusion.pt"),
            )
        else:
            # with_head=False: only the adapter-layer weights are saved
            adapter_model.save_adapter(os.path.join(staging, "adapter"), adapter_name, with_head=False)
        
    
        # 3. Saves a serialized object to disk
        torch.save({
            "epoch":  epoch, "step": step, "metric": metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "adapter_name": adapter_name,
            "adapter_init_from": adapter_init_from,
            "active_composition": active_composition,
        }, os.path.join(staging, "training_state.pt"))
        
        # 4. Remove old save_dir, rename staging into its place
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.rename(staging, save_dir)
        
    # ==================================================================
    # FULL FT PATH
    # ==================================================================
    else:
        # FFT path: write single .pt file at path
        # writes: path (one .pt with full state_dict)
        tmp = path + ".tmp"
        # Build and save state dict
        torch.save({
            "epoch":                epoch,
            "step":                 step,
            "metric":               metric,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }, tmp)
        os.replace(tmp, path)


def load_checkpoint(
    path      : str,
    model     : torch.nn.Module,
    optimizer : torch.optim.Optimizer | None = None,
    scheduler = None,
    adapter_name: str | None = None,
) -> tuple[int, int, float]:
    """
    Load checkpoint from path into model.
    Returns (epoch, step, metric) so train.py knows where to resume.
    """
    
    peft_model = _peft_inner(model)
    adapter_model = _adapter_inner(model)
    # ==================================================================
    # PEFT PATH
    # ==================================================================
    if peft_model is not None:
        # PEFT path
        save_dir = path[:-3] if path.endswith(".pt") else path
        adapter_dir = os.path.join(save_dir, "adapter")

        # Load adapter state
        st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if os.path.exists(st_path):
            adapter_state = load_file(st_path)
        else:
            adapter_state = torch.load(
                os.path.join(adapter_dir, "adapter_model.bin"),
                map_location="cpu", weights_only=False,
            )
        # Load into the inner PeftModel so the adapter-only state_dict keys
        # match (outer WhisperASR wrapper would prepend "model." to all keys).
        peft_model.load_state_dict(adapter_state, strict=False)
        
        # Load training state
        ts_path = os.path.join(save_dir, "training_state.pt")
        if os.path.exists(ts_path):
            ts = torch.load(ts_path, map_location="cpu", weights_only=False)
            if optimizer is not None and ts.get("optimizer_state_dict"):
                optimizer.load_state_dict(ts["optimizer_state_dict"])
            if scheduler is not None and ts.get("scheduler_state_dict"):
                scheduler.load_state_dict(ts["scheduler_state_dict"])
            return ts["epoch"], ts["step"], ts["metric"]
        # Test-eval-only load (best.pt without resume): no training state on disk
        return 0, 0, float("inf")
    
    # ==================================================================
    # ADAPTER PATH
    # ==================================================================
    elif adapter_model is not None:
        # Adapter path
        save_dir = path[:-3] if path.endswith(".pt") else path
        adapter_dir = os.path.join(save_dir, "adapter")
        
        # Read training_state first so we know what artifact to expect on disk
        ts_path = os.path.join(save_dir, "training_state.pt")
        ts = None
        if os.path.exists(ts_path):
            ts = torch.load(ts_path, map_location="cpu", weights_only=False)
        
        # active_composition is None both when no training_state.pt is present and
        # for a plain single adapter (freeze_strategy: adapter, which stores no
        # spec); it is a dict for stacked adapters and lightweight fusion.
        active_composition = (ts or {}).get("active_composition")
        
        if active_composition is not None and active_composition["kind"] == "lightweight_fuse":
            # architecture already rebuilt by build_model -> apply_lightweight_fusion
            # (sources reloaded+frozen, stock fusion swapped). Just restore weights.
            blob = torch.load(os.path.join(adapter_dir, "lightweight_fusion.pt"),
                            map_location="cpu", weights_only=False)
            fstate, loaded = blob["fusion_state"], 0
            for name, mod in adapter_model.named_modules():
                if name in fstate:
                    mod.load_state_dict(fstate[name]); loaded += 1
            assert loaded == len(fstate), f"restored {loaded}/{len(fstate)} fusion modules"
        else:
            # Single adapter on disk (single or stacked-target-only)
            adapter_model.load_adapter(adapter_dir)
            # Re-establish the active composition
            if active_composition is not None:
                # Stack(...) or a single-name spec
                adapter_model.active_adapters = _composition_from_spec(active_composition)
            else:
                # Without set_active_adapters the adapter loads but stays inactive
                if adapter_name is None:
                    raise ValueError(
                        "adapter_name is required to activate a single-adapter checkpoint. "
                        "Pass the name from cfg.model.adapter.name (e.g. 'ast')."
                    )
                adapter_model.set_active_adapters(adapter_name)
        
        # Restore optimizer / scheduler state if present, return resume position
        if ts is not None:
            if optimizer is not None and ts.get("optimizer_state_dict"):
                optimizer.load_state_dict(ts["optimizer_state_dict"])
            if scheduler is not None and ts.get("scheduler_state_dict"):
                scheduler.load_state_dict(ts["scheduler_state_dict"])
            return ts["epoch"], ts["step"], ts["metric"]
        # Test-eval-only load (best.pt without resume): no training state on disk
        return 0, 0, float("inf")
        
    # ==================================================================
    # FULL FT PATH
    # ==================================================================
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
        # Restore model weights
        model.load_state_dict(ckpt["model_state_dict"])

        # Restore optimizer state if provided
        if optimizer is not None and ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Restore scheduler state if provided
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Return resume position and best metric
        return ckpt["epoch"], ckpt["step"], ckpt["metric"]