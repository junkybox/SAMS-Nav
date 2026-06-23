if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import contextlib
import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_node_policy import DiffusionTransformerNodePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
# from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from classes.navigation_metrics import training_proxy_metrics
from parameter import UPDATING_MAP_SIZE

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _DDPLossWrapper(torch.nn.Module):
    """Routes compute_loss through forward so DDP hooks gradient sync.

    DistributedDataParallel only synchronizes gradients for calls made via
    the wrapped module's forward; the policy exposes training through
    compute_loss, so this shim makes that the forward.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


class TrainDiffusionTransformerNodeWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerNodePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerNodePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.final_navigation_metrics = None

    def save_training_gif(self, pred_action, gt_action):
        """Render one predicted/expert action sequence for this epoch."""
        gif_dir = pathlib.Path(self.output_dir).joinpath('training_gifs')
        frame_dir = gif_dir.joinpath('frames')
        gif_dir.mkdir(parents=True, exist_ok=True)
        frame_dir.mkdir(parents=True, exist_ok=True)

        pred = pred_action[0].detach().cpu().numpy()
        expert = gt_action[0].detach().cpu().numpy()
        pred_path = np.vstack((np.zeros(2), np.cumsum(pred, axis=0)))
        expert_path = np.vstack((np.zeros(2), np.cumsum(expert, axis=0)))

        all_points = np.vstack((pred_path, expert_path))
        padding = max(1.0, float(np.ptp(all_points, axis=0).max()) * 0.15)
        x_limits = (all_points[:, 0].min() - padding, all_points[:, 0].max() + padding)
        y_limits = (all_points[:, 1].min() - padding, all_points[:, 1].max() + padding)

        frame_paths = []
        frame_count = max(len(pred_path), len(expert_path))
        for frame_index in range(1, frame_count):
            figure, axis = plt.subplots(figsize=(6, 6))
            pred_end = min(frame_index + 1, len(pred_path))
            expert_end = min(frame_index + 1, len(expert_path))
            axis.plot(
                expert_path[:expert_end, 0],
                expert_path[:expert_end, 1],
                'o-',
                color='tab:blue',
                label='expert',
            )
            axis.plot(
                pred_path[:pred_end, 0],
                pred_path[:pred_end, 1],
                'o-',
                color='tab:orange',
                label='prediction',
            )
            axis.scatter(0.0, 0.0, marker='s', color='black', label='start')
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_aspect('equal', adjustable='box')
            axis.grid(True, alpha=0.3)
            axis.legend(loc='best')
            axis.set_title(f'Training epoch {self.epoch}: action sequence')
            axis.set_xlabel('relative x')
            axis.set_ylabel('relative y')
            figure.tight_layout()

            frame_path = frame_dir.joinpath(
                f'epoch_{self.epoch:04d}_{frame_index:03d}.png'
            )
            figure.savefig(frame_path, dpi=120)
            plt.close(figure)
            frame_paths.append(frame_path)

        gif_path = gif_dir.joinpath(f'epoch_{self.epoch:04d}.gif')
        with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))

        for frame_path in frame_paths:
            frame_path.unlink()
        return str(gif_path)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # distributed setup: active when launched via torchrun, e.g.
        #   torchrun --standalone --nproc_per_node=2 train.py --config-dir=. \
        #     --config-name=train_navigation_transformer_node.yaml
        use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        if use_ddp:
            dist.init_process_group(backend='nccl')
            ddp_rank = dist.get_rank()
            ddp_world_size = dist.get_world_size()
            local_rank = int(os.environ.get('LOCAL_RANK', ddp_rank))
        else:
            ddp_rank, ddp_world_size, local_rank = 0, 1, 0
        # rank 0 owns logging, validation, sampling, and checkpoints; other
        # ranks only contribute gradients
        is_main = ddp_rank == 0

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_sampler = None
        if use_ddp:
            dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=ddp_world_size,
                rank=ddp_rank,
                shuffle=dataloader_cfg.pop('shuffle', True),
            )
            train_dataloader = DataLoader(
                dataset, sampler=train_sampler, **dataloader_cfg
            )
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset (validated on rank 0 only)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        if is_main:
            print(f"Training dataset size: {len(train_dataloader)}")
            print(f"Validation dataset size: {len(val_dataloader)}")
            print(f"Max train steps: {cfg.training.max_train_steps}")
            print(f"Max val steps: {cfg.training.max_val_steps}")
        train_steps_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            train_steps_per_epoch = min(train_steps_per_epoch, cfg.training.max_train_steps)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                train_steps_per_epoch * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = None
        if is_main:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = None
        if is_main:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # device transfer
        ddp_loss_model = None
        if use_ddp:
            device = torch.device('cuda', local_rank)
            torch.cuda.set_device(device)
            print(f'Training device: {device} (DDP rank {ddp_rank}/{ddp_world_size})')
            self.model.to(device)
            # the encoder has parameters unused by compute_loss
            # (q_values_layer, current_embedding), so DDP must tolerate them
            ddp_loss_model = torch.nn.parallel.DistributedDataParallel(
                _DDPLossWrapper(self.model),
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            device = torch.device(cfg.training.device)
            device_ids = cfg.training.get('device_ids', None)
            if device_ids is not None:
                device_ids = [int(device_id) for device_id in device_ids]
                if len(device_ids) > 1:
                    if device.type != 'cuda':
                        raise ValueError('training.device_ids requires a CUDA device')
                    primary_device = torch.device(f'cuda:{device_ids[0]}')
                    if device != primary_device:
                        print(
                            f'Training device changed from {device} to '
                            f'{primary_device} for DataParallel.'
                        )
                        device = primary_device
            print(f'Training device: {device}')
            self.model.to(device)
            if hasattr(self.model, 'configure_data_parallel'):
                self.model.configure_data_parallel(device_ids)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # save batch for sampling
        train_sampling_batch = None
        # fixed held-out batch for offline sampling metrics, so action MSE and
        # proxy rates measure generalization instead of training-set recall
        val_sampling_batch = None
        if is_main and len(val_dataloader) > 0:
            val_sampling_batch = next(iter(val_dataloader))

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        log_context = JsonLogger(log_path) if is_main else contextlib.nullcontext()
        with log_context as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, disable=not is_main,
                        mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        if ddp_loss_model is not None:
                            raw_loss = ddp_loss_model(batch)
                        else:
                            raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # Calculate total gradient norm
                        total_grad_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_grad_norm = p.grad.data.norm(2).item()
                                total_grad_norm += param_grad_norm ** 2
                        total_grad_norm = total_grad_norm ** 0.5

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'grad_norm': total_grad_norm
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if is_main:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation (rank 0 only; weights are identical on all
                # ranks after gradient sync). RNG is forked and reseeded so
                # every epoch evaluates the same diffusion timesteps and noise:
                # val_loss differences then reflect the model, not sampling
                # luck, which makes top-k checkpoint selection trustworthy
                if is_main and (self.epoch % cfg.training.val_every) == 0:
                    fork_devices = [device] if device.type == 'cuda' else []
                    with torch.random.fork_rng(devices=fork_devices):
                        torch.manual_seed(cfg.training.seed)
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss

                # run diffusion sampling on a held-out validation batch
                # (rank 0 only): these metrics track generalization, unlike
                # the previous training-batch proxies which track recall
                if is_main and (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        sampling_batch = (
                            val_sampling_batch
                            if val_sampling_batch is not None
                            else train_sampling_batch
                        )
                        batch = dict_apply(sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        # Training diagnostics remain single-sample even when
                        # the navigation policy defaults to best-of-K rollout.
                        if hasattr(policy, 'num_samples'):
                            result = policy.predict_action(obs_dict, num_samples=1)
                        else:
                            # Other node-policy backends retain their original
                            # predict_action signature.
                            result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['val_action_mse_error'] = mse.item()
                        navigation_metrics = training_proxy_metrics(
                            pred_action=pred_action,
                            expert_action=gt_action,
                            obs_dict=obs_dict,
                            n_obs_steps=self.model.n_obs_steps,
                            node_resolution=cfg.task.shape_meta.obs.node_resolution,
                            coordinate_scale=UPDATING_MAP_SIZE / 2,
                        )
                        self.final_navigation_metrics = navigation_metrics
                        step_log.update({
                            f'val_proxy_{key}': value
                            for key, value in navigation_metrics.items()
                        })
                        save_gifs = cfg.training.get('save_gifs', True)
                        gif_every = cfg.training.get(
                            'gif_every', cfg.training.sample_every
                        )
                        if save_gifs and (self.epoch % gif_every) == 0:
                            gif_path = self.save_training_gif(
                                pred_action, gt_action
                            )
                            step_log['training_gif_path'] = gif_path
                            wandb_run.log(
                                {
                                    'training_gif': wandb.Video(
                                        gif_path, format='gif'
                                    )
                                },
                                step=self.global_step,
                            )
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint (rank 0 only)
                if is_main and (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if is_main:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        if use_ddp:
            dist.barrier()
            dist.destroy_process_group()
            if not is_main:
                return

        print('\n=== Offline Proxy Navigation Metrics (validation batch) ===')
        if self.final_navigation_metrics is None:
            print('No sampled trajectory was available for metric calculation.')
        else:
            print(
                '| Collision Rate:',
                self.final_navigation_metrics['collision_rate'],
            )
            print(
                '| Success Rate:',
                self.final_navigation_metrics['success_rate'],
            )
            print(
                '| Smoothness Cost:',
                self.final_navigation_metrics['smoothness_cost'],
            )
        print(
            'Note: training values are offline trajectory proxies; '
            'closed-loop metrics are reported by test_driver.py.'
        )

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerNodeWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
