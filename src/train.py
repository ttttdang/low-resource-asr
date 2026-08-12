"""
src/train.py

Unified training loop for low-resource ASR fine-tuning.
Config-driven via YAML (OmegaConf). Optional wandb logging.
Supports zero-shot eval, full fine-tuning, and adapter-based fine-tuning.

Run from the repository root — config paths are relative and resolved at runtime.
==================================================================
"""

import os
import time
import random
import torch
import wandb
import argparse
import math
from torch.utils.data       import DataLoader, Sampler
from omegaconf              import OmegaConf
from transformers           import get_linear_schedule_with_warmup
from accelerate             import Accelerator

from peft                   import PeftModel

from dataset                import ASRDataset
from collator               import WhisperCollator
from models.whisper_base    import WhisperASR
from text_normalizer        import get_normalizer
from evaluate               import evaluate, export_trn
from utils                  import set_seed, seed_worker, get_logger, save_checkpoint, load_checkpoint, expand_user_paths

# ==================================================================
# STEP 0. SAMPLER
# ==================================================================
class EpochAnchoredSampler(Sampler):
    """
    Target-Anchored Epoch (TAE) sampler for multilingual training.
    
    Epoch composition:
    - Anchor language       : all indices, shuffled, no repetition (each clip is seen once per epoch)
    - Auxiliary languages   : |anchor| draws with replacement (per epoch fresh draws via
                              set_epoch + seed_advance so cross-epoch coverage of aux. data is diverse)
    Total per-epoch samples : |anchor| x (1 + n_aux_langs)
    
    Rationale               : Anchoring epoch length to the smallest language means the model sees
                              every anchor clip exactly once per epoch, preventing the smallest language
                              from being under-sampled when training on heavily imbalanced multi-lang mix.
    """
    
    def __init__(self, df, anchor_lang: str = "vi_vn", seed: int = 42):
        self.seed               = seed
        self.epoch              = 0
        indices_by_lang         = {
            lang: df.index[df["language"] == lang].tolist() 
            for lang in df["language"].unique() # idx retrieved from ASRDataset instance
            }
        
        if anchor_lang not in indices_by_lang:
            raise ValueError(
                f"anchor_language={anchor_lang!r} not found in dataset languages "
                f"{sorted(indices_by_lang)}. Did filter_languages drop it?"
            )
        
        self.anchor_idx         = indices_by_lang[anchor_lang]
        self.aux_idx_by_lang    = {
            l: idx for l, idx in indices_by_lang.items()
            if l != anchor_lang            
        }
        
        self.n_anchor           = len(self.anchor_idx)
    
    def set_epoch(self, epoch: int):
        """
        Called at the top of each training epoch so per-epoch RNG advances
        Allowing aux-language draws are fresh cross-epoch (different sample 
        of Zh/Fr each pass through the loop), while anchor language stays exhaustive.
        """
        self.epoch = epoch
        
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idxs = list(self.anchor_idx)
        # vi-only shuffled
        rng.shuffle(idxs)
        for aux_lang_idx in self.aux_idx_by_lang.values():
            # select n_anchor random indx for aux: vi_size with replacement
            idxs += rng.choices(aux_lang_idx, k=self.n_anchor)
            
        # interleaves Vi/Zh/Fr so batches are mixed-lang (not all-Vi-then-all-Zh-then-all-Fr blocks)
        rng.shuffle(idxs)
        # print(f"[TAE] epoch={self.epoch} anchor={self.n_anchor} aux={list(self.aux_idx_by_lang.keys())} total={len(idxs)}", flush=True)
        return iter(idxs)
    
    def __len__(self):
        return self.n_anchor * (1 + len(self.aux_idx_by_lang))

# ==================================================================
# STEP 1. SETUP
# ==================================================================
def setup(cfg):
    """Initialize seed, logger, wandb, device. Return (logger, device, accelerator)."""

    # 0. Expand `~` in all config paths in-place
    expand_user_paths(cfg)

    # 1. Initialize seed
    set_seed(cfg.seed)
    
    # Instantiate Accelerator
    accelerator = Accelerator(mixed_precision=cfg.training.get("mixed_precision", "no"))

    # 2. Build logger
    logger = get_logger(
        name     = cfg.run_name,
        log_dir  = cfg.log_dir,
        log_file = "train.log"
    )

    # 3. Initialize wandb (opt-in: set `wandb.enabled: true` in the config)
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", False):
        os.environ["WANDB_MODE"] = "disabled"
    if accelerator.is_main_process:
        wandb.init(
            project = wb.get("project", "low-resource-asr"),
            name    = cfg.run_name,
            id      = cfg.run_name,                                       # set unique ID for resume
            resume  = "allow" if cfg.get("resume_from") else "never",     # resume only when explicitly requested
            notes   = cfg.get("description", ""),
            tags    = cfg.get("tags", []),                                # tags in wandb run table
            config  = OmegaConf.to_container(cfg, resolve=True),
        )

    # 4. Device
    device = accelerator.device
    logger.info(f"Device: {device}")

    return logger, device, accelerator

# ==================================================================
# STEP 2. DATA LOADERS
# ==================================================================
def build_dataloaders(cfg, collator):
    """Build train and val DataLoaders from manifest paths in cfg.

    Each loader gets its own seeded torch.Generator so shuffle order is
    deterministic across runs with the same cfg.seed.
    """

    max_label_tokens = cfg.data.get("max_label_tokens", 448)

    train_dataset = ASRDataset(
        manifest_paths   = cfg.data.train_manifest,
        norm_condition   = cfg.norm_condition.train,
        max_duration_sec = cfg.data.max_duration_sec,
        min_duration_sec = cfg.data.min_duration_sec,
        max_samples      = cfg.data.get("max_samples", None),
        filter_datasets  = cfg.data.get("train_datasets", None),
        filter_languages = cfg.data.get("train_languages", None),
        tokenizer        = collator.tokenizer,
        max_label_tokens = max_label_tokens,
    )
    val_dataset = ASRDataset(
        manifest_paths   = cfg.data.val_manifest,
        norm_condition   = cfg.norm_condition.eval,
        max_duration_sec = cfg.data.max_duration_sec,
        min_duration_sec = cfg.data.min_duration_sec,
        max_samples      = cfg.data.get("max_samples", None),
        filter_datasets  = cfg.data.get("val_datasets", None),
        filter_languages = cfg.data.get("val_languages", None),
        tokenizer        = collator.tokenizer,
        max_label_tokens = max_label_tokens,
    )

    train_gen = torch.Generator().manual_seed(cfg.seed)
    val_gen   = torch.Generator().manual_seed(cfg.seed)

    # Target-Anchored Epoch (TAE) sampler when enabled in YAML,
    # otherwise plain shuffling.
    train_sampler = None
    if cfg.data.get("epoch_anchored_sampling", False):
        train_sampler = EpochAnchoredSampler(
            train_dataset.df, # ASRDataset instance convert each lang dataset to DF and concat them
            anchor_lang = cfg.data.get("anchor_language", "vi_vn"),
            seed        = cfg.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size     = cfg.training.batch_size,
        shuffle        = (train_sampler is None),   # mutually exclusive with sampler
        sampler        = train_sampler,
        num_workers    = cfg.data.num_workers,
        pin_memory     = True,
        collate_fn     = collator,
        worker_init_fn = seed_worker,
        generator      = train_gen,
        persistent_workers = cfg.data.num_workers > 0, # keep loaders alive across epochs
        prefetch_factor    = 4 if cfg.data.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size     = cfg.training.batch_size,
        shuffle        = False,
        num_workers    = cfg.data.num_workers,
        pin_memory     = True,
        collate_fn     = collator,
        worker_init_fn = seed_worker,
        generator      = val_gen,
        persistent_workers = cfg.data.num_workers > 0,
        prefetch_factor    = 4 if cfg.data.num_workers > 0 else None,
    )

    return train_loader, val_loader

# ==================================================================
# STEP 3. MODEL
# ==================================================================
def build_model(cfg, device, logger, accelerator):
    """Instantiate WhisperASR, apply freeze strategy, move to device."""

    model = WhisperASR(
        model_name = cfg.model.name,
        language   = cfg.model.language,
        task       = cfg.model.task,
        dropout    = cfg.model.dropout,
        clear_suppress_tokens = cfg.model.get("clear_suppress_tokens", True),
    )
    
    _st = model.model.generation_config.suppress_tokens
    logger.info(f"suppress_tokens: n={0 if _st is None else len(_st)} "
                f"(clear_suppress_tokens={cfg.model.get('clear_suppress_tokens', True)})")
    
    # Apply freeze strategy from config
    #   "none" | "freeze_encoder" | "freeze_decoder" | "freeze_all"
    #   | "lora" | "adapter" | "stacked_adapter" | "lightweight_fusion"
    strategy = cfg.model.freeze_strategy
    if strategy == "none":
        model.unfreeze_all()
    elif strategy == "freeze_encoder":
        model.freeze_encoder()
    elif strategy == "freeze_decoder":
        model.freeze_decoder()
    elif strategy == "freeze_all":
        model.freeze_all()
    elif strategy == "lora":
        model.freeze_all()                      # freeze the base first
        model.apply_lora(cfg.model.lora)       
    elif strategy == "adapter":
        model.freeze_all()
        model.apply_adapter(cfg.model.adapter)
        init_from = cfg.model.adapter.get("init_from")
        if init_from:
            logger.info(f"Warm init adapter from {os.path.expanduser(init_from)}")
        logger.info(f"Active adapter after apply_adapter: {model.model.active_adapters}")
    elif strategy == "stacked_adapter":
        model.freeze_all()
        model.apply_stacked_adapter(cfg.model.adapter)
        for src in cfg.model.adapter.sources:
            logger.info(
                f"Loaded frozen source adapter '{src.name}' from "
                f"{os.path.expanduser(src.init_from)}"
            )
        logger.info(f"Active composition after apply_stacked_adapter: {model.model.active_adapters}")
    elif strategy == "lightweight_fusion":
        model.freeze_all()
        model.apply_lightweight_fusion(cfg.model.adapter)
        for src in cfg.model.adapter.sources:
            logger.info(f"Loaded frozen source adapter '{src.name}' from "
                        f"{os.path.expanduser(src.init_from)}")
        logger.info(f"Active composition after apply_lightweight_fusion: "
                    f"{model.model.active_adapters}")
    else:
        raise ValueError(f"Unknown freeze_strategy: {strategy!r}")

    model.to(device)

    # Log parameter counts to wandb and logger
    trainable, total = model.trainable_params()
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")
    if accelerator.is_main_process:
        wandb.log({"model/trainable_params": trainable, "model/total_params": total})

    return model

# ==================================================================
# HELPER FUNCTION
# ==================================================================
def _resolve_adapter_metadata(cfg):
    """
    Derive (adapter_name, adapter_init_from, active_composition) from cfg.
    
    Centralizes the freeze-strategy-aware metadata resolution so train()
    and run_test_eval() stay in sync.
    """
    if cfg.model.freeze_strategy == "adapter":
        return (
            cfg.model.adapter.get("name", "vi"),
            cfg.model.adapter.get("init_from"),
            None,
        )
    if cfg.model.freeze_strategy == "stacked_adapter":
        return (
            cfg.model.adapter.get("name", "vi"),
            None,
            {"kind": "stack", "adapters": list(cfg.model.adapter.composition)},
        )
    if cfg.model.freeze_strategy == "lightweight_fusion":
        fcfg = cfg.model.adapter.get("fusion", {}) or {}
        return (None, None, {
            "kind":     "lightweight_fuse",          # distinct kind allows custom save/load
            "adapters": list(cfg.model.adapter.composition),
            "d_k":      int(fcfg.get("d_k", 256)),
            "value":    bool(fcfg.get("value", False)),
        })
    # Non-adapter strategies (FFT, LoRA, freeze_all, etc.) - values are unused downstream
    return None, None, None

# ==================================================================
# STEP 4. TRAINING LOOP
# ==================================================================
def train(cfg, model, train_loader, val_loader,
          optimizer, scheduler, normalizer, logger, device, accelerator):

    # Adapter metadata for save/load_checkpoint kwargs (no-op for non-adapter strategies)
    adapter_name, adapter_init_from, active_composition = _resolve_adapter_metadata(cfg)
        
    accum_steps = cfg.training.grad_accum_steps
    ckpt_metric = cfg.training.get("checkpoint_metric", "wer")
    best_metric = float("inf")
    global_step = 0
    start_epoch = 0
    train_start = time.time()

    # Resume from checkpoint if specified.
    # start_epoch is the epoch to BEGIN AT — i.e. the first un-trained epoch.
    # This was saved as `epoch + 1` after the previous run completed that epoch.
    if cfg.get("resume_from"):
        start_epoch, global_step, best_metric = load_checkpoint(
            cfg.resume_from, model, optimizer, scheduler, adapter_name=adapter_name,
        )
        logger.info(f"Resumed from {cfg.resume_from} at epoch {start_epoch}")

    patience = cfg.training.get("early_stop_patience")   # None or int
    epochs_since_best = 0

    for epoch in range(start_epoch, cfg.training.epochs):

        # Refresh the TAE sampler's per-epoch RNG so the aux-language draws
        # differ each epoch. No-op when no custom sampler is active.
        _sampler = getattr(train_loader, "sampler", None)
        if _sampler is not None and hasattr(_sampler, "set_epoch"):
            _sampler.set_epoch(epoch)

        # ----------------------------------------------------------
        # 4.1 TRAIN
        # ----------------------------------------------------------
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        running_loss        = 0.0   # mean raw loss across the current accum window
        epoch_loss_sum      = 0.0   # sum of per-effective-batch losses across the epoch
        n_effective_batches = 0
        
        n_batches = len(train_loader)

        for step, batch in enumerate(train_loader):

            # 1. Move to device
            input_features = batch["input_features"].to(device)
            labels         = batch["labels"].to(device)

            # 2. Forward pass
            output = model(input_features=input_features, labels=labels)

            # 3. Scale loss for gradient accumulation
            window_size = min(accum_steps,
                              n_batches - (step // accum_steps) * accum_steps)
            loss = output.loss / window_size
            accelerator.backward(loss)
            running_loss += loss.item()

            # 4. Optimizer step every accum_steps mini-batches
            is_accum_step = (step + 1) % accum_steps == 0
            is_last_step  = step + 1 == n_batches       

            if is_accum_step or is_last_step:                 

                # Clip gradients
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.training.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log mean raw loss across the effective batch
                # i.e. sum of scaled losses across windows since accum_steps is fixed
                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss": running_loss,
                        "train/lr":   scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item(),
                        "epoch":      epoch,
                    }, step=global_step)
                epoch_loss_sum      += running_loss
                n_effective_batches += 1
                running_loss = 0.0

        # ----------------------------------------------------------
        # 4.2 VALIDATE
        # ----------------------------------------------------------
        val_metrics, _ = evaluate(
            model          = model,
            dataloader     = val_loader,
            tokenizer      = model.processor.tokenizer,
            normalizer     = normalizer,
            norm_condition = cfg.norm_condition.eval,
            device         = device,
            num_beams      = cfg.generation.num_beams,
            max_new_tokens = cfg.generation.max_new_tokens,
            no_repeat_ngram_size = cfg.generation.get("no_repeat_ngram_size", 0),
            desc           = f"Val epoch {epoch}",
        )

        # Restore training mode after evaluate() for next epoch
        model.train()

        val_metric = val_metrics[ckpt_metric]
        val_wer    = val_metrics["wer"]
        val_cer = val_metrics["cer"]
        
        epoch_seconds = time.time() - epoch_start
        logger.info(f"Epoch {epoch}  val/wer={val_wer:.4f}  val/cer={val_cer:.4f}  "
                    f"SUB={val_metrics['subs_rate']:.4f}  "
                    f"INS={val_metrics['ins_rate']:.4f}  "
                    f"DEL={val_metrics['del_rate']:.4f}  "
                    f"avg_gen_len={val_metrics['avg_gen_len']:.0f}  "
                    f"pct_at_cap={val_metrics['pct_at_cap']:.2%}  "
                    f"took {epoch_seconds:.1f}s ({epoch_seconds/60:.1f} min)")

        epoch_log = {
            "val/wer":                val_wer,
            "val/cer":                val_cer,
            "val/subs_rate":          val_metrics["subs_rate"],
            "val/ins_rate":           val_metrics["ins_rate"],
            "val/del_rate":           val_metrics["del_rate"],
            "epoch":                  epoch,
            "runtime/epoch_seconds":  epoch_seconds,
        }
        if n_effective_batches > 0:
            epoch_log["train/epoch_loss"] = epoch_loss_sum / n_effective_batches
        if accelerator.is_main_process:
            wandb.log(epoch_log, step=global_step)

        # Save latest checkpoint every epoch.
        # epoch + 1 = the epoch to start FROM on resume (current epoch is already complete).
        if accelerator.is_main_process:
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            global_step, best_metric,
                            os.path.join(cfg.checkpoint_dir, "latest.pt"),
                            adapter_name=adapter_name,
                            adapter_init_from=adapter_init_from,
                            active_composition=active_composition,
                            )

        # Save best checkpoint when val_metric improves
        if val_metric < best_metric:
            best_metric = val_metric
            epochs_since_best = 0
            if accelerator.is_main_process:
                save_checkpoint(model, optimizer, scheduler, epoch + 1,
                                global_step, best_metric,
                                os.path.join(cfg.checkpoint_dir, "best.pt"),
                                adapter_name=adapter_name,
                                adapter_init_from=adapter_init_from,
                                active_composition=active_composition,
                                )
            logger.info(f"  New best {ckpt_metric.upper()}: {best_metric:.4f} — saved best.pt")
        else:
            epochs_since_best += 1
            if patience is not None and epochs_since_best >= patience:
                logger.info(f"Early stop: val {ckpt_metric.upper()} hasn't improved for {patience} epochs (best={best_metric:.4f}). Breaking.")
                break
    
    train_total = time.time() - train_start
    logger.info(f"Training phase total: {train_total:.1f}s ({train_total/3600:.2f} hrs)")
    if accelerator.is_main_process:
        wandb.summary["runtime/train_total_hours"] = train_total / 3600

    return global_step


# ==================================================================
# FINAL TEST EVALUATION
# ==================================================================
def run_test_eval(cfg, model, collator, normalizer, logger, device, accelerator, global_step=0):
    """Evaluate on each test dataset.

    If best.pt exists (after fine-tuning) load it.
    Otherwise evaluate whatever weights `model` currently holds — for
    freeze_all this is the pretrained Whisper checkpoint (zero-shot baseline).
    """
    adapter_name, _, _ = _resolve_adapter_metadata(cfg)
        
    best_ckpt = os.path.join(cfg.checkpoint_dir, "best.pt")
    # PEFT save_checkpoint writes to a directory (stripping .pt), FFT writes to a file.
    # Check both forms so test eval loads the best-epoch adapter rather than silently
    # falling through to whatever in-memory state remains post-training.
    best_ckpt_dir = best_ckpt[:-3] if best_ckpt.endswith(".pt") else best_ckpt
    if os.path.exists(best_ckpt) or os.path.isdir(best_ckpt_dir):
        load_checkpoint(best_ckpt, model, adapter_name=adapter_name)
        logger.info(f"Loaded best checkpoint from {best_ckpt}")
        if cfg.model.freeze_strategy in ("adapter", "stacked_adapter", "lightweight_fusion"):
            logger.info(f"Active adapter after load_checkpoint: {model.model.active_adapters}")
    else:
        logger.info(
            f"No checkpoint at {best_ckpt} — evaluating current model weights "
            f"(zero-shot if freeze_strategy=freeze_all)"
        )

    # For PEFT runs, merge LoRA adapter into base for faster test eval.
    # This eliminates the per-attention-module BAx contribution overhead
    if isinstance(model.model, PeftModel):
        model.model = model.model.merge_and_unload()
        model.to(device)
        logger.info("Merged LoRA adapter into base model for test eval (faster inference)")

    results_dir = os.path.join(cfg.results_dir, cfg.run_name)
    os.makedirs(results_dir, exist_ok=True)

    for ds_name, manifest_path in cfg.data.test_manifests.items():
        test_dataset = ASRDataset(
            manifest_paths   = manifest_path,
            norm_condition   = cfg.norm_condition.eval,
            max_duration_sec = cfg.data.max_duration_sec,
            min_duration_sec = cfg.data.min_duration_sec,
            filter_datasets  = [ds_name],
            max_samples      = cfg.data.get("max_samples"),
        )
        test_gen = torch.Generator().manual_seed(cfg.seed)
        # Generation batch size can differ from training (beam search needs ~num_beams × memory).
        # Falls back to training batch size when not set in config.
        test_batch_size = cfg.generation.get("batch_size", cfg.training.batch_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size     = test_batch_size,
            shuffle        = False,
            num_workers    = cfg.data.num_workers,
            pin_memory     = True,
            collate_fn     = collator,
            worker_init_fn = seed_worker,
            generator      = test_gen,
            persistent_workers = cfg.data.num_workers > 0,
            prefetch_factor    = 4 if cfg.data.num_workers > 0 else None,
        )

        metrics, results_df = evaluate(
            model          = model,
            dataloader     = test_loader,
            tokenizer      = model.processor.tokenizer,
            normalizer     = normalizer,
            norm_condition = cfg.norm_condition.eval,
            device         = device,
            num_beams      = cfg.generation.num_beams,
            max_new_tokens = cfg.generation.max_new_tokens,
            no_repeat_ngram_size = cfg.generation.get("no_repeat_ngram_size", 0),  # 0=OFF default; set 3 for Latin loop-suppression (NEVER for Indic byte-fallback)
            desc           = f"Test {ds_name}",
        )

        # Save results CSV for post-hoc analysis and bootstrap test (optional)
        csv_path = os.path.join(results_dir, f"{ds_name}_test.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(
            f"{ds_name}: WER={metrics['wer']:.4f}  CER={metrics['cer']:.4f}  "
            f"SUB={metrics['subs_rate']:.4f}  INS={metrics['ins_rate']:.4f}  "
            f"DEL={metrics['del_rate']:.4f}  (N={metrics['n_ref_words']})"
        )

        # Export trn files for MAPSSWE significance testing
        export_trn(
            results_df,
            ref_path = os.path.join(results_dir, f"{ds_name}_ref.trn"),
            hyp_path = os.path.join(results_dir, f"{ds_name}_hyp.trn"),
        )
        if accelerator.is_main_process:
            wandb.log({
                f"test/{ds_name}/wer":        metrics["wer"],
                f"test/{ds_name}/cer":        metrics["cer"],
                f"test/{ds_name}/subs_rate":  metrics["subs_rate"],
                f"test/{ds_name}/ins_rate":   metrics["ins_rate"],
                f"test/{ds_name}/del_rate":   metrics["del_rate"],
            }, step=global_step)


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":

    run_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True, help="Path to YAML config")
    parser.add_argument("--overrides", nargs="*",     default=[],
                        help="Dot-list overrides e.g. training.lr=1e-5")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    logger, device, accelerator = setup(cfg)
    
    logger.info(f"Run: {cfg.run_name}")
    if cfg.get("description"):
        logger.info(f"Description: {cfg.description}")
    if cfg.get("tags"):
        logger.info(f"Tags: {', '.join(cfg.tags)}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Build model and collator
    model = build_model(cfg, device, logger, accelerator)
    collator = WhisperCollator(
        feature_extractor = model.processor.feature_extractor,
        tokenizer         = model.processor.tokenizer,
        language          = cfg.model.language,
        task              = cfg.model.task,
    )
    # data.languages selects the text normalizers used for validation and test
    # scoring; it must cover every language code present in the manifests, since
    # evaluate() indexes this dict by the batch's language.
    langs = cfg.data.get("languages")
    if not langs:
        raise ValueError(
            "cfg.data.languages is required — it selects the text normalizers used "
            "for validation and test scoring. Set e.g. `languages: [ast_es]`."
        )
    normalizer = {lang: get_normalizer(lang) for lang in langs}

    # Zero-shot: skip training, go straight to test eval
    if cfg.model.freeze_strategy == "freeze_all" or cfg.training.epochs == 0:
        logger.info("Zero-shot mode — skipping training, running test eval directly")
        test_start = time.time()
        run_test_eval(cfg, model, collator, normalizer, logger, device, accelerator)
        test_total = time.time() - test_start
        logger.info(f"Test eval phase total: {test_total:.1f}s ({test_total/60:.1f} min)")
        if accelerator.is_main_process:
            wandb.summary["runtime/test_total_hours"] = test_total / 3600
    else:
        # Build dataloaders, optimizer, scheduler only when training
        train_loader, val_loader = build_dataloaders(cfg, collator)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr           = cfg.training.lr,
            weight_decay = cfg.training.weight_decay,
        )

        # Update total steps for scheduler for trailing partial accmumulation window
        steps_per_epoch = math.ceil(len(train_loader) / cfg.training.grad_accum_steps)
        total_steps     = steps_per_epoch * cfg.training.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = cfg.training.warmup_steps,
            num_training_steps = total_steps,
        )

        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
            )

        global_step = train(cfg, model, train_loader, val_loader,
                            optimizer, scheduler, normalizer, logger, device, accelerator)
        test_start = time.time()
        run_test_eval(cfg, model, collator, normalizer, logger, device, accelerator,
                      global_step=global_step)
        test_total = time.time() - test_start
        logger.info(f"Test eval phase total: {test_total:.1f}s ({test_total/60:.1f} min)")
        if accelerator.is_main_process:
            wandb.summary["runtime/test_total_hours"] = test_total / 3600

    run_total = time.time() - run_start
    logger.info(f"Total run wall clock: {run_total:.1f}s ({run_total/3600:.2f} hrs)")
    if accelerator.is_main_process:
        wandb.summary["runtime/total_hours"] = run_total / 3600

    if accelerator.is_main_process:
        wandb.finish()