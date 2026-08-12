"""
Lightweight AdapterFusion - low-rank, value-free (W_V) AdapterFusion.

The `adapters` library BertFusion uses full Q/K/V at d_k = hidden (1024)
resulting to 151M trainable params over 48 Whisper Medium adapter locations.
This module subclasses BertFusion to:
(a) Project Q/K to a reduced d_k
(b) Drop W_V, bringing trainable params to ~2 x H x d_k x 48.

With d_k = 256, this yields ~25.1M, equal to the reduction-factor-4 Pfeiffer
parameter count — so fusion is compute-matched to the single-adapter and
stacked-adapter methods by construction.

Notes:
- forward() is not overwritten. BertFusion.forward contracts Q's last dim against K's,
    so it's agnostic to d_k
    Ref: https://github.com/adapter-hub/adapters/blob/53a1ea164c07498be7d522aed96c9e600aa3b38b/src/adapters/methods/modeling.py#L359
- W_V removal is expressed via the library's static fusion config (value=False), which
    the inherited forward already handles.
- Only __init__ changes (q/k projection width + 1/sqrt(d_k) scaling via the 
    temperature field)
- Fusion is trained from scratch, so swapped-in modules use fresh (bert) init
"""

import torch.nn as nn
from adapters.methods.modeling import BertFusion, Adapter

class LightWeightFusion(BertFusion):
    """
    Low-rank (d_k), value-free AdapterFusion block.
    
    Args:
        config                          :   the AdapterFusionConfig the library built the stock module
                                            with (must have value=False for the headline lightweight variant)
        dense_size                      :   model hidden size H (1024 for Whisper Medium)
        attention_probs_dropout_prob    :   dropout on attention scores
        d_k                             :   reduced query/key width (256 -> R-4 matched)
    
    Trainable params: W_Q (H x d_k) + W_K (H x d_k) per location. No W_V.    
    """
    
    def __init__(self, config, dense_size, attention_probs_dropout_prob, d_k=256):
        super().__init__(config, dense_size, attention_probs_dropout_prob)
        self.d_k = d_k
        
        # Re-create Q & K at the reduced width (override the dense_size versions)
        if self.config["query"]:
            self.query = nn.Linear(dense_size, d_k)
            self.query.apply(Adapter.init_bert_weights)
        if self.config["key"]:
            self.key = nn.Linear(dense_size, d_k)
            self.key.apply(Adapter.init_bert_weights)
        
        # Temperature scaling
        # with the standard scaled-dot-production attention softmax(QK^T / sqrt(k_d))
        # to keep the scores variance ~1 so softmax doesn't saturate when d_k changes
        self.T = float(d_k) ** 0.5 # sqrt(256) = 16 -> softmax(scores/sqrt(d_k))
        self.reduction = 0.0 # constant scale so that T is not annealed each fwd pass
        
def replace_with_lightweight_fusion(model, d_k=256):
    """
    Swap every stock BertFusion in `model` for a LightweightFusion(d_k).
        
    add_adapter_fusion() has already placed a stock BertFusion at each adapter
    location (in BottleneckLayer.adapter_fusion_layer, a ModuleDict keyed by
    fusion name)
    We replace each in-placed, preserving its config/dropout but reducing d_k.
    This must run before model.to(device), so that modules inherit dtype from 
    the module they replace and get moved with the model.
        
    Returns the number of modules swapped — assert this equals the number of
    adapter locations (48 for whisper-medium, full enc+dec placement).
        
    Ref:
    - add_adapter_fusion(): https://github.com/adapter-hub/adapters/blob/53a1ea164c07498be7d522aed96c9e600aa3b38b/src/adapters/model_mixin.py#L904
    - add_fusion_layer(): https://github.com/adapter-hub/adapters/blob/53a1ea164c07498be7d522aed96c9e600aa3b38b/src/adapters/methods/bottleneck.py#L120C9-L120C25
    """
    n = 0
        
    # 1. Check every sub-module
    for module in model.modules():
        fdict = getattr(module, "adapter_fusion_layer", None)
        # 2. Keep only BottleneckLayers that have a fusion dict
        if not isinstance(fdict, nn.ModuleDict):
            continue
        for name, fusion in list(fdict.items()):
            # 3. Swap a stock BertFusion
            if not isinstance(fusion, BertFusion) or isinstance(fusion, LightWeightFusion):
                continue
            ref = next(fusion.parameters()) # read device + dtype off the existing module
            new = LightWeightFusion(fusion.config, fusion.dense_size, fusion.dropout.p, d_k=d_k,
                                    ).to(device=ref.device, dtype=ref.dtype) # build the drop-in with config + dense_size the library used, only d_k differs
            # 4. Re-assign a ModuleDict entry to register the submodule
            # Ensure the layer's fwd now calls LightWeightFusion instead of BertFusion
            fdict[name] = new
            n += 1
    return n
                