import os
import time
import shutil
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from .stft import mag_pha_istft, mag_pha_stft
from .utils import copy_state, batch_pesq, anti_wrapping_function, \
                  phase_losses, get_stft_args_from_config


class Solver(object):
    def __init__(
        self,
        data,
        model,
        discriminator,
        optim,
        optim_disc,
        scheduler,
        scheduler_disc,
        args,
        logger,
        device=None
    ):
        # Dataloaders and samplers
        self.tr_loader = data['tr_loader']      # Training DataLoader
        self.va_loader = data['va_loader']      # Validation DataLoader
        self.tt_loader_list = data['tt_loader_list']      # Test Time Evaluation DataLoader

        self.model = model
        self.discriminator = discriminator
        self.optim = optim
        self.optim_disc = optim_disc
        self.scheduler = scheduler
        self.scheduler_disc = scheduler_disc

        # loss weights
        self.loss = args.loss

        # logger
        self.logger = logger

        # dataset & STFT
        self.segment = args.segment
        self.stft_args = get_stft_args_from_config(args.model)
        # Basic config
        self.device = device or torch.device(args.device)

        self.max_steps = args.max_steps

        self.validation_interval = args.validation_interval
        self.summary_interval = args.summary_interval
        self.log_interval = args.log_interval
        self.best_models_num = args.best_models_num
        self.scheduler_step_interval = getattr(args, 'scheduler_step_interval', None)

        # Checkpoint settings
        self.continue_from = args.continue_from

        self.writer = None
        self.best_models = []
        self.log_dir = args.log_dir
        self.num_workers = args.num_workers
        self.args = args

        self.step_start = 0

        # Initialize or resume (checkpoint loading)
        self._reset()

    def _save_model_checkpoint(self, steps, state_dict):
        """ Save model checkpoint. """
        package_model = {}
        package_model['model'] = copy_state(state_dict)

        tmp_path = "model.tmp"
        save_path = f"model_{steps}.th"
        torch.save(package_model, tmp_path)
        os.rename(tmp_path, save_path)

    def _save_states_checkpoint(self, step):
        """ Save states checkpoint. """
        package = {}
        package['model'] = copy_state(self.model.state_dict())
        package['best_models'] = self.best_models
        package['discriminator'] = copy_state(self.discriminator.state_dict())
        package['optimizer'] = self.optim.state_dict()
        package['optimizer_disc'] = self.optim_disc.state_dict()
        package['scheduler'] = self.scheduler.state_dict() if self.scheduler is not None else None
        package['scheduler_disc'] = self.scheduler_disc.state_dict() if self.scheduler_disc is not None else None
        package['args'] = self.args
        package['step'] = step
        # Write to a temporary file first
        tmp_path = "states.tmp"
        save_path = "states.th"
        torch.save(package, tmp_path)
        os.rename(tmp_path, save_path)

    def _update_best_models(self, steps, valid_pesq_value):
        """Maintain top-k models by validation PESQ. """
        entry = {
            "steps": steps,
            "valid_pesq_value": valid_pesq_value,
            "model": copy_state(self.model.state_dict())
        }
        self.best_models.append(entry)
        # Keep only top k by PESQ in descending order
        self.best_models.sort(key=lambda x: x["valid_pesq_value"], reverse=True)
        if len(self.best_models) > self.best_models_num:
            for evicted in self.best_models[self.best_models_num:]:
                old_path = f"model_{evicted['steps']}.th"
                if os.path.exists(old_path):
                    os.remove(old_path)
            self.best_models = self.best_models[:self.best_models_num]

    def _reset(self):
        """Load checkpoint if 'continue_from' is specified, or create a fresh writer if not."""
        if self.continue_from is not None:
            self.logger.info(f'Loading checkpoint model: {self.continue_from}')
            if not os.path.exists(self.continue_from):
                raise FileNotFoundError(f"Checkpoint directory {self.continue_from} not found.")

            # Attempt to copy the 'tensorbd' directory (TensorBoard logs) if it exists
            src_tb_dir = os.path.join(self.continue_from, 'tensorbd')
            dst_tb_dir = self.log_dir

            if os.path.exists(src_tb_dir):
                if not os.path.exists(dst_tb_dir):
                    shutil.copytree(src_tb_dir, dst_tb_dir)
                else:
                    self.logger.warning(f"TensorBoard log dir {dst_tb_dir} already exists. Skipping copy.")
            self.writer = SummaryWriter(log_dir=dst_tb_dir)

            # loads the checkpoint file from disk
            ckpt_path = os.path.join(self.continue_from, 'states.th')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            package = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            model_state = package['model']
            model_disc_state = package.get('discriminator', None)
            optim_state = package['optimizer']
            optim_disc_state = package.get('optimizer_disc', None)
            scheduler_state = package.get('scheduler', None)
            scheduler_disc_state = package.get('scheduler_disc', None)
            self.best_models = package.get('best_models', [])
            self.step_start = package.get('step', 0)

            self.model.load_state_dict(model_state)
            self.optim.load_state_dict(optim_state)

            if model_disc_state is not None:
                self.discriminator.load_state_dict(model_disc_state)
            if optim_disc_state is not None:
                self.optim_disc.load_state_dict(optim_disc_state)

            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            if self.scheduler_disc is not None and scheduler_disc_state is not None:
                self.scheduler_disc.load_state_dict(scheduler_disc_state)

        else:
            # If there's no checkpoint to resume from, just create a fresh SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)


    def _infinite_loader(self):
        """Yield batches from tr_loader infinitely."""
        while True:
            for data in self.tr_loader:
                yield data

    def train(self):
        self.logger.info("Training for %d steps", self.max_steps)

        if self.step_start != 0:
            self.logger.info("Resuming training from step %d", self.step_start + 1)

        loader_iter = self._infinite_loader()

        for step in range(self.step_start + 1, self.max_steps + 1):
            start = time.time()

            data = next(loader_iter)
            noisy, clean = data[0], data[1]
            loss_dict = self._run_one_step(noisy, clean)

            if step % self.log_interval == 0:
                lr = self.optim.param_groups[0]['lr']
                info = " | ".join(f"{k} {v:.4f}" for k, v in loss_dict.items())
                self.logger.info(f"Train | Step {step}/{self.max_steps} | LR {lr:.6f} | {1/(time.time() - start):.1f} iters/s | {info}")

            if step % self.summary_interval == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"Train/{key}_Loss", value, step)

            if step % self.validation_interval == 0:
                val_pesq_score = self._run_validation(step)
                self._update_best_models(step, val_pesq_score)
                if any(m['steps'] == step for m in self.best_models):
                    self._save_model_checkpoint(step, self.model.state_dict())
                self._save_states_checkpoint(step)

            if self.scheduler_step_interval and step % self.scheduler_step_interval == 0:
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.scheduler_disc is not None:
                    self.scheduler_disc.step()

        self.logger.info("-" * 70)
        self.logger.info("Training Completed")
        if self.best_models:
            best = self.best_models[0]
            self.logger.info(f"Best Model | Steps: {best['steps']}, Valid PESQ: {best['valid_pesq_value']:.4f}")
        self.logger.info("-" * 70)
        self.writer.close()

    def _run_one_step(self, noisy, clean):

        self.model.train()
        self.discriminator.train()

        noisy = noisy.to(self.device)
        clean = clean.to(self.device)
        one_labels = torch.ones(noisy.shape[0]).to(self.device)

        noisy_com = mag_pha_stft(noisy, **self.stft_args)[2]
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean, **self.stft_args)

        clean_mag_hat, clean_pha_hat, clean_com_hat = self.model(noisy_com)

        clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **self.stft_args)
        clean_mag_hat_con, clean_pha_hat_con, clean_com_hat_con = mag_pha_stft(clean_hat, **self.stft_args)

        # Discriminator training
        clean_list, clean_list_hat = list(clean.cpu().numpy()), list(clean_hat.detach().cpu().numpy())
        batch_pesq_score = batch_pesq(clean_list, clean_list_hat, workers=self.num_workers)

        self.optim_disc.zero_grad()

        metric_r = self.discriminator(clean_mag.unsqueeze(1), clean_mag.unsqueeze(1))
        metric_g = self.discriminator(clean_mag.unsqueeze(1), clean_mag_hat_con.detach().unsqueeze(1))

        loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

        if batch_pesq_score is not None:
            loss_disc_g = F.mse_loss(batch_pesq_score.to(self.device), metric_g.flatten())
        else:
            loss_disc_g = 0

        loss_disc = loss_disc_r + loss_disc_g

        loss_disc.backward()

        max_grad_norm = getattr(self.args, 'max_grad_norm', 5.0)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=max_grad_norm)

        self.optim_disc.step()

        # Generator training
        self.optim.zero_grad()

        loss_magnitude = F.mse_loss(clean_mag, clean_mag_hat)
        loss_phase = phase_losses(clean_pha, clean_pha_hat)
        loss_complex = F.mse_loss(clean_com, clean_com_hat) * 2
        loss_consistency = F.mse_loss(clean_com_hat, clean_com_hat_con) * 2

        metric_g = self.discriminator(clean_mag.unsqueeze(1), clean_mag_hat_con.unsqueeze(1))
        loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

        loss_gen = loss_metric * self.loss.metric + \
                   loss_complex * self.loss.complex + \
                   loss_consistency * self.loss.consistency + \
                   loss_magnitude * self.loss.magnitude + \
                   loss_phase * self.loss.phase

        loss_gen.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

        self.optim.step()

        loss_dict = {
            "Metric": loss_metric.item(),
            "Complex": loss_complex.item(),
            "Consistency": loss_consistency.item(),
            "Phase": loss_phase.item(),
            "Magnitude": loss_magnitude.item(),
            "Disc": loss_disc.item() if isinstance(loss_disc, torch.Tensor) else loss_disc,
            "Gen": loss_gen.item()
        }

        return loss_dict


    def _run_validation(self, steps):
        self.model.eval()
        self.discriminator.eval()

        val_err_complex = 0
        val_err_mag = 0
        val_err_phase = 0
        clean_list, clean_hat_list = [], []

        with torch.no_grad():
            for data in self.va_loader:
                noisy, clean = data[0], data[1]
                noisy = noisy.squeeze(0).to(self.device)
                clean = clean.squeeze(0).to(self.device)

                # Full-length inference (no segmentation)
                noisy_in = noisy.unsqueeze(0)   # [1, length]
                clean_in = clean.unsqueeze(0)    # [1, length]

                noisy_com = mag_pha_stft(noisy_in, **self.stft_args)[2]
                clean_mag, clean_pha, clean_com = mag_pha_stft(clean_in, **self.stft_args)

                clean_mag_hat, clean_pha_hat, clean_com_hat = self.model(noisy_com)

                clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **self.stft_args)

                # Align lengths (center=False may produce different output length)
                min_len = min(clean.shape[-1], clean_hat.shape[-1])
                clean = clean[..., :min_len]
                clean_hat = clean_hat[..., :min_len]

                clean_list.append(clean.squeeze().detach().cpu().numpy())
                clean_hat_list.append(clean_hat.squeeze().detach().cpu().numpy())

                val_err_complex += F.l1_loss(clean_com, clean_com_hat).item()
                val_err_mag += F.l1_loss(clean_mag, clean_mag_hat).item()
                val_err_phase += torch.mean(anti_wrapping_function(clean_pha - clean_pha_hat)).item()


        val_err_complex /= len(self.va_loader)
        val_err_mag /= len(self.va_loader)
        val_err_phase /= len(self.va_loader)
        val_pesq_result = batch_pesq(clean_list, clean_hat_list, workers=self.num_workers, normalize=False)
        if val_pesq_result is not None:
            val_pesq_score = val_pesq_result.mean().item()
        else:
            val_pesq_score = 0

        self.logger.info("-" * 70)
        self.logger.info(
            f"Validation | Step {steps} | Complex Diff {val_err_complex:.5f} | Magnitude Diff {val_err_mag:.5f} | Phase Diff {val_err_phase:.5f} | Valid PESQ {val_pesq_score:.5f}"
        )
        self.logger.info("-" * 70)
        self.writer.add_scalar("Validation/Complex_Loss", val_err_complex, steps)
        self.writer.add_scalar("Validation/Magnitude_Loss", val_err_mag, steps)
        self.writer.add_scalar("Validation/Phase_Loss", val_err_phase, steps)
        self.writer.add_scalar("Validation/Validation_PESQ_Score", val_pesq_score, steps)

        return val_pesq_score
