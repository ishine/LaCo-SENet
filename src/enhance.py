import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchaudio
import logging
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader

from src.utils import LogProgress
from src.stft import mag_pha_stft, mag_pha_istft

# Constants
DEFAULT_SAMPLE_RATE = 16_000


def save_wavs(wavs_dict: Dict[str, torch.Tensor], filepath: str, sr: int = DEFAULT_SAMPLE_RATE) -> None:
    """Save multiple waveforms to separate files.

    Args:
        wavs_dict: Dictionary mapping suffixes to waveform tensors
        filepath: Base filepath (suffixes will be appended)
        sr: Sample rate in Hz
    """
    for key, wav in wavs_dict.items():
        try:
            torchaudio.save(filepath + f"_{key}.wav", wav, sr)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save {filepath}_{key}.wav: {e}")


def write(wav: torch.Tensor, filename: str, sr: int = DEFAULT_SAMPLE_RATE) -> None:
    """Write a single waveform to file with normalization.

    Args:
        wav: Waveform tensor
        filename: Output filename
        sr: Sample rate in Hz
    """
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    try:
        torchaudio.save(filename, wav.cpu(), sr)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save {filename}: {e}")


def enhance_multiple_snr(
    args: Any,
    model: torch.nn.Module,
    dataloader_list: Dict[int, DataLoader],
    logger: logging.Logger,
    epoch: Optional[int] = None,
    local_out_dir: str = "samples"
) -> None:
    """Run enhancement on multiple SNR levels.

    Args:
        args: Configuration object
        model: Enhancement model
        dataloader_list: Dictionary mapping SNR values to dataloaders
        logger: Logger instance
        epoch: Training epoch number (for naming)
        local_out_dir: Output directory
    """
    for snr, data_loader in dataloader_list.items():
        enhance(args, model, data_loader, logger, snr, epoch, local_out_dir)


def enhance(
    args: Any,
    model: torch.nn.Module,
    data_loader: DataLoader,
    logger: logging.Logger,
    snr: int,
    epoch: Optional[int] = None,
    local_out_dir: str = "samples",
    stft_args: Optional[Dict[str, Any]] = None
) -> None:
    """Run enhancement on a dataset and save results.

    Args:
        args: Configuration object with model and device settings
        model: Enhancement model
        data_loader: DataLoader for test dataset
        logger: Logger instance
        snr: Signal-to-noise ratio in dB
        epoch: Training epoch number (for naming outputs)
        local_out_dir: Output directory for enhanced samples
        stft_args: STFT parameters for frequency-domain models

    Raises:
        ValueError: If stft_args is None for frequency-domain models
    """
    model.eval()

    if stft_args is None:
        raise ValueError("stft_args must be provided")

    suffix = f"_epoch{epoch+1}" if epoch is not None else ""
    outdir_wavs = os.path.join(local_out_dir, f"wavs" + suffix + f"_{snr}dB")
    os.makedirs(outdir_wavs, exist_ok=True)

    failed_samples = []

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        for batch_idx, data in enumerate(iterator):
            try:
                # Get batch data (batch, channel, time)
                noisy, clean, id, _ = data

                # STFT and model inference
                noisy_com = mag_pha_stft(noisy, **stft_args)[2].to(args.device)
                clean_mag_hat, clean_pha_hat, _ = model(noisy_com)
                enhanced = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

                # Move to CPU and squeeze channel dimension
                clean = clean.squeeze(1).cpu()
                noisy = noisy.squeeze(1).cpu()
                enhanced = enhanced.squeeze(1).cpu()

                wavs_dict = {
                    "noisy": noisy,
                    "clean": clean,
                    "enhanced": enhanced,
                }

                save_wavs(wavs_dict, os.path.join(outdir_wavs, id[0]))

            except Exception as e:
                logger.error(f"Failed to process sample {batch_idx} (id={id[0] if id else 'unknown'}): {e}")
                failed_samples.append((batch_idx, id[0] if id else 'unknown', str(e)))
                continue

    # Report failures if any
    if failed_samples:
        logger.warning(f"Failed to process {len(failed_samples)}/{len(data_loader)} samples")
        failure_log_path = os.path.join(outdir_wavs, "failed_samples.txt")
        try:
            with open(failure_log_path, 'w') as f:
                for batch_idx, sample_id, error in failed_samples:
                    f.write(f"Batch {batch_idx} | ID: {sample_id} | Error: {error}\n")
            logger.info(f"Failure log saved to {failure_log_path}")
        except Exception as e:
            logger.error(f"Failed to save failure log: {e}")


if __name__ == "__main__":
    import logging
    import logging.config
    import argparse
    from src.data import VoiceBankDataset
    from src.utils import load_model, load_checkpoint, get_stft_args_from_config
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory.")
    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint file name.")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory for enhanced samples.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu).")

    # Stateful Convolution options
    parser.add_argument(
        "--use_stateful_conv",
        action="store_true",
        default=False,
        help="Use stateful convolutions for streaming inference."
    )

    args = parser.parse_args()
    chkpt_dir = args.chkpt_dir
    device = args.device
    local_out_dir = args.output_dir

    conf = OmegaConf.load(os.path.join(chkpt_dir, '.hydra', "config.yaml"))
    hydra_conf = OmegaConf.load(os.path.join(chkpt_dir, '.hydra', "hydra.yaml"))
    del hydra_conf.hydra.job_logging.handlers.file
    hydra_conf.hydra.job_logging.root.handlers = ['console']
    logging_conf = OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)

    logging.config.dictConfig(logging_conf)
    logger = logging.getLogger(__name__)
    conf.device = device

    # Load model and checkpoint
    model = load_model(conf.model, device)
    model = load_checkpoint(model, chkpt_dir, args.chkpt_file, device)

    # Apply stateful convolutions if requested
    if args.use_stateful_conv:
        from src.models.streaming.converters import (
            convert_to_stateful,
            set_streaming_mode,
            get_stateful_layer_count,
        )

        logger.info("Applying stateful convolutions...")
        model = convert_to_stateful(model, verbose=False, inplace=True)
        model.to(device)
        model.eval()
        set_streaming_mode(model, True)

        layer_counts = get_stateful_layer_count(model)
        logger.info(f"Stateful conversion complete: {layer_counts['total']} layers")
        if layer_counts["StatefulCausalConv1d"] > 0:
            logger.info(f"  - StatefulCausalConv1d: {layer_counts['StatefulCausalConv1d']}")
        if layer_counts["StatefulAsymmetricConv2d"] > 0:
            logger.info(f"  - StatefulAsymmetricConv2d: {layer_counts['StatefulAsymmetricConv2d']}")
        if layer_counts["StatefulCausalConv2d"] > 0:
            logger.info(f"  - StatefulCausalConv2d: {layer_counts['StatefulCausalConv2d']}")

    # Load VoiceBank+DEMAND test set
    hf_dataset_id = conf.dset.get("hf_dataset_id", "JacobLinCool/VoiceBank-DEMAND-16k")
    testset = load_dataset(hf_dataset_id, split="test")

    use_pcs400 = conf.dset.get("use_pcs400", False) if hasattr(conf, "dset") else False
    tt_dataset = VoiceBankDataset(testset, segment=None, with_id=True, with_text=True, use_pcs400=use_pcs400)

    tt_loader = DataLoader(
        dataset=tt_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    # Prepare STFT args from model config
    stft_args = get_stft_args_from_config(conf.model)

    logger.info(f"Dataset: VoiceBank-DEMAND")
    logger.info(f"Model: Backbone")
    logger.info(f"Checkpoint: {chkpt_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Stateful conv: {args.use_stateful_conv}")
    logger.info(f"Output directory: {local_out_dir}")
    os.makedirs(local_out_dir, exist_ok=True)

    enhance(model=model,
            data_loader=tt_loader,
            args=conf,
            snr="mixed",
            epoch=None,
            logger=logger,
            local_out_dir=local_out_dir,
            stft_args=stft_args
            )
