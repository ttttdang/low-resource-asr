"""
Pfeiffer bottleneck adapter setup via the `adapters` library.

Usage: Shared between adapter tuning and adapter composition methods.

"""

import os
import adapters
from adapters import SeqBnConfig

# ------------------------------------------------------------------
# SINGLE ADAPTER
# ------------------------------------------------------------------
def ensure_adapters_initialized(model):
    """
    Wrapper for adapters.init.
    
    Usage: To call multiple times across composed setup functions
    Uses hasattr on a method injected by adapters.init as a capability check
    """
    if not hasattr(model, "adapter_summary"):
        adapters.init(model)

def setup_pfeiffer(model, cfg):
    """
    Add or warm-init a Pfeiffer bottleneck adapter with configurable name
    (default 'vi') and put the model in train-adapter mode.

    If cfg.init_from is set, load existing adapter weights from disk;
    otherwise add a fresh adapter with the SeqBnConfig params.
    """
    adapter_name = cfg.get("name", "vi")
    ensure_adapters_initialized(model)

    init_from = cfg.get("init_from")
    if init_from:
        src_path = os.path.expanduser(init_from)
        model.load_adapter(src_path, load_as=adapter_name)
    else:
        # init_weights="mam_adapter" zeroes the up-projection (LoRA-equivalent init):
        # Adapter(x) near 0 at step 0 yields model output identical to base at init.
        # "bert" default compounds over 48 layers and breaks EOS prediction in zero-shot eval (verified empirically).
        config_kwargs = {
            "reduction_factor": cfg.reduction_factor,
            "non_linearity":    cfg.non_linearity,
            "dropout":          cfg.dropout,
            "init_weights":     cfg.init_weights,
        }
        # Pass a list of layer indices to skip (e.g. all decoder layers for enc-only).
        # Whisper-medium has global, encoder-first indices
        # Encoder: 0-23; Decoder: 24-47
        if cfg.get("leave_out") is not None:
            config_kwargs["leave_out"] = list(cfg.leave_out)
        # placement flags. None means SeqBnConfig default (output_adapter=True only).
        for placement_flag in ("output_adapter", "mh_adapter", "cross_adapter"):
            if cfg.get(placement_flag) is not None:
                config_kwargs[placement_flag] = cfg.get(placement_flag)
        config = SeqBnConfig(**config_kwargs)
        model.add_adapter(adapter_name, config=config)
    
    model.train_adapter(adapter_name)
    print(model.adapter_summary())

# ------------------------------------------------------------------
# SEQUENTIAL STACKING
# ------------------------------------------------------------------
def setup_stacked(model, cfg):
    """
    Add stacked adapter composition: load + freeze source adapter, add a
    fresh trainable target adapter, and set the composition to
    Stack(*cfg.composition) so the forward pass runs input -> source -> target -> next.

    Only the target adapter is trainable; source weights are loaded from disk
    and frozen. cfg shape:
        name              : ast                      # trainable target name
        composition       : [es, ast]                # forward order
        reduction_factor  : 4                        # target's SeqBnConfig
        non_linearity     : relu
        dropout           : 0.0
        init_weights      : mam_adapter
        leave_out         : null                     # full encoder+decoder placement
        sources:
          - name      : es
            init_from : ./checkpoints/<source_run>/best/adapter
    """
    from adapters.composition import Stack

    ensure_adapters_initialized(model)

    # 1. Load + freeze source adapter(s). 
    # load_adapter sets active = src_name, which we'll override at step 3.
    for src in cfg.sources:
        src_path = os.path.expanduser(src.init_from)
        model.load_adapter(src_path, load_as=src.name)

    # 2. Add the fresh trainable target adapter. 
    # Reuses setup_pfeiffer because the target needs the exact same SeqBnConfig path. 
    # setup_pfeiffer ends with train_adapter(target_name), which:
    #      - freezes base + source adapter 
    #      - unfreezes only the target adapter
    #      - sets active = target_name  (single — about to be overridden)
    setup_pfeiffer(model, cfg)

    # 3. Override active_adapters to the Stack composition. Without this, the
    #    forward pass would only run the target adapter and ignore the source.
    model.active_adapters = Stack(*cfg.composition)

    # 4. Defensive: explicitly freeze source adapter params. train_adapter
    #    should already have frozen them at step 2, but this ensures it
    #    and the result is verifiable in the adapter_summary() print below.
    source_names = {s.name for s in cfg.sources}
    for n, p in model.named_parameters():
        for src_name in source_names:
            if f"adapters.{src_name}." in n:
                p.requires_grad = False

    print(model.adapter_summary())
# ------------------------------------------------------------------
# LIGHTWEIGHT ADAPTER FUSION
# ------------------------------------------------------------------
def setup_lightweight_fusion(model, cfg):
    """
    Low-rank, value-free AdapterFusion over frozen source adapters.

    Loads each source adapter, then:
      (1) adds fusion with a Static config (value=cfg.fusion.value, default
          False) so the library builds no W_V, and
      (2) swaps each resulting BertFusion for a LightweightFusion(d_k) — see
          src/models/adapter_fusion.py.

    Only the fusion layer is trainable; the sources stay frozen.

    cfg shape:
        composition : [es, fr]            # load_as names; also the Fuse order
        fusion:
          d_k   : 256                     # low-rank attention width (default 256)
          value : false                   # keep W_V false = lightweight (default)
        sources:
          - { name: es, init_from: ./checkpoints/<es_run>/best/adapter }
          - { name: fr, init_from: ./checkpoints/<fr_run>/best/adapter }

    Ref: https://github.com/adapter-hub/adapters/blob/53a1ea164c07498be7d522aed96c9e600aa3b38b/src/adapters/configuration/adapter_fusion_config.py#L48
    """
    from adapters.composition import Fuse
    from adapters import StaticAdapterFusionConfig
    from .adapter_fusion import replace_with_lightweight_fusion
    
    ensure_adapters_initialized(model)
    
    # 1. Load each source adapter
    for src in cfg.sources:
        model.load_adapter(os.path.expanduser(src.init_from), load_as=src.name)
    
    fuse        = Fuse(*cfg.composition)
    fcfg        = cfg.get("fusion", {}) or {}
    want_val    = bool(fcfg.get("value", False))
    d_k         = int(fcfg.get("d_k", 256))
    
    # 2. Library builds stock fusion modules
    model.add_adapter_fusion(fuse, config=StaticAdapterFusionConfig(value=want_val))
    
    # 3. Swap stock modules to LightWeightFusion(d_k)
    # Sanity check placement count
    n = replace_with_lightweight_fusion(model, d_k=d_k)
    print(f"[lightweight_fusion] swapped {n} modules -> d_k={d_k}, value={want_val}")
    
    # 4. Re-assert trainable fusion and frozen sources (fresh modules after swap)
    model.train_adapter_fusion(fuse) # freeze base + sources, train fusion
    source_names = {s.name for s in cfg.sources}
    for pname, p in model.named_parameters():
        if any(f"adapters.{s}." in pname for s in source_names):
            p.requires_grad = False
    
    print(model.adapter_summary())
