"""
Whisper model wrapper for ASR fine-tuning.

Wraps WhisperForConditionalGeneration with:
  - Language/task configuration
  - Selective layer freezing for zero-shot / full fine-tune / PEFT
  - Trainable parameter counting for logging

==================================================================
"""

import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

class WhisperASR(nn.Module):
    """
    Whisper wrapper for multilingual ASR.

    Args:
        model_name  : HuggingFace model id "openai/whisper-medium"
        language    : Target language for forced decoding e.g. "vietnamese"
        task        : "transcribe" or "translate"
        dropout     : Dropout rate applied to attention layers (default 0.0)
        clear_suppress_tokens : If True (default) empty Whisper's token suppress list.
                                If False, keep the checkpoint's list active.

    """

    def __init__(
        self,
        model_name : str = "openai/whisper-medium",
        language   : str = "vietnamese",
        task       : str = "transcribe",
        dropout    : float = 0.0,
        clear_suppress_tokens: bool = True,
    ):
        super().__init__()

        # 1. Load pretrained model
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, dropout=dropout)

        # 2. Load processor (bundles feature extractor + tokenizer)
        self.processor = WhisperProcessor.from_pretrained(model_name)

        # 3. Whisper `special tokens`
        self.model.generation_config.language = language
        self.model.generation_config.task = task
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None

        # 4. Whisper `suppress_tokens`
        if clear_suppress_tokens:
            self.model.config.suppress_tokens = []
            self.model.generation_config.suppress_tokens = []

    # ------------------------------------------------------------------
    # LORA
    # ------------------------------------------------------------------
    def apply_lora(self, lora_cfg):
        """Wrap self.model with PEFT LoRA adapters."""
    
        config = LoraConfig(
            r               = lora_cfg.r,
            lora_alpha      = lora_cfg.alpha,
            lora_dropout    = lora_cfg.dropout,
            target_modules  = list(lora_cfg.target_modules),
            bias            = "none",
        )
        
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
    
    # ------------------------------------------------------------------
    # ADAPTERS
    # ------------------------------------------------------------------
    def apply_adapter(self, adapter_cfg):
        """Wrap self.model with a Pfeiffer bottleneck adapter."""
        from .bottleneck_adapter import setup_pfeiffer
        setup_pfeiffer(self.model, adapter_cfg)

    def apply_stacked_adapter(self, cfg):
        """
        Wrap self.model with a stacked adapter composition.
        Frozen source adapter(s) + fresh trainable target adapter, with active
        composition set to Stack(*cfg.composition).
        """
        from .bottleneck_adapter import setup_stacked
        setup_stacked(self.model, cfg)
    
    def apply_lightweight_fusion(self, cfg):
        """
        Wrap self.model with low-rank, value-free adapter fusion.

        Frozen source adapters + a trainable LightweightFusion (Q/K → d_k, no
        W_V) over them.
        """
        from .bottleneck_adapter import setup_lightweight_fusion
        setup_lightweight_fusion(self.model, cfg)

    # ------------------------------------------------------------------
    # FREEZE CONTROLS
    # ------------------------------------------------------------------
    def freeze_encoder(self):
        """Freeze all encoder parameters. Used for decoder-only fine-tuning."""
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        """Freeze all decoder parameters. Used for encoder-only fine-tuning."""
        for param in self.model.model.decoder.parameters():
            param.requires_grad = False

    def freeze_all(self):
        """Freeze entire model. Used for zero-shot evaluation."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze entire model. Used for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # PARAMETER COUNTING
    # ------------------------------------------------------------------

    def trainable_params(self):
        """
        Return (trainable, total) parameter counts, for logging at the start
        of every run.
        """
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------
    def generate(self, input_features, **kwargs):
        """
        Delegate to WhisperForConditionalGeneration.generate().
        Seq-to-seq generation during inference, taking encoder outputs (log-mel spectrograms produced by processor)
        and generate predicted token IDs.
        """
        return self.model.generate(input_features, **kwargs)
    
    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(self, input_features, labels=None, **kwargs):
        return self.model(input_features=input_features, labels=labels, **kwargs)
