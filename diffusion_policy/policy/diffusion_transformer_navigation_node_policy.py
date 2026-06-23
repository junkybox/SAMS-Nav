from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.encoder.navigation_node_encoder import NavigationNodeEncoder
from diffusion_policy.common.pytorch_util import dict_apply

# node feature layout: [rel_x, rel_y, utility, guidepost, occupancy, goal_dx, goal_dy]
GOAL_FEATURE_SLICE = slice(5, 7)


class DiffusionTransformerNavigationNodePolicy(BaseImagePolicy):
    """Goal-conditioned continuous waypoint-delta diffusion policy.

    Differences from the discrete one-hot backend it replaces:
    - diffuses normalized continuous 2-D waypoint deltas instead of one-hot
      class logits, so samples keep diffusion's multimodal expressiveness and
      no argmax decoding is needed;
    - supports classifier-free goal guidance: goal features are dropped with
      probability ``goal_drop_prob`` during training, and inference blends
      goal-conditioned and goal-free predictions with ``goal_guidance_scale``;
    - ``predict_action(..., num_samples=K)`` draws K candidate trajectories in
      one batched denoising pass for test-time best-of-K selection.
    """

    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: NavigationNodeEncoder,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            num_samples=8,
            # goal classifier-free guidance
            goal_drop_prob=0.1,
            goal_guidance_scale=1.0,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # parameters passed to scheduler.step
            **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = shape_meta['obs']['embedding_dim']
        # kept for test-time graph snapping
        self.node_resolution = shape_meta['obs']['node_resolution']

        assert obs_as_cond, (
            'Goal guidance requires observation conditioning (obs_as_cond=true).'
        )
        input_dim = action_dim
        output_dim = input_dim
        cond_dim = obs_feature_dim

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.goal_drop_prob = float(goal_drop_prob)
        self.goal_guidance_scale = float(goal_guidance_scale)
        self.num_samples = int(num_samples)
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def configure_data_parallel(self, device_ids=None):
        if device_ids is not None and len(device_ids) > 1:
            self.parallel_obs_encoder = nn.DataParallel(
                self.obs_encoder,
                device_ids=device_ids,
                output_device=device_ids[0],
            )
            self.parallel_model = nn.DataParallel(
                self.model,
                device_ids=device_ids,
                output_device=device_ids[0],
            )
        else:
            self.parallel_obs_encoder = self.obs_encoder
            self.parallel_model = self.model

    # ========= inference  ============
    def _encode_obs(self, obs, drop_goal=False):
        """Encode the first n_obs_steps observations into (B, To, Do) cond."""
        To = self.n_obs_steps
        value = next(iter(obs.values()))
        B = value.shape[0]
        this_obs = dict_apply(
            obs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        node_inputs = this_obs['node_inputs']
        if drop_goal:
            node_inputs = node_inputs.clone()
            node_inputs[..., GOAL_FEATURE_SLICE] = 0.0
        obs_encoder = getattr(self, 'parallel_obs_encoder', self.obs_encoder)
        obs_features = obs_encoder(node_inputs,
                                   this_obs['node_padding_mask'],
                                   this_obs['edge_mask'],
                                   this_obs['current_index'],
                                   this_obs['current_edge'],
                                   this_obs['edge_padding_mask'])
        return obs_features.reshape(B, To, -1)

    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, uncond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        guidance = self.goal_guidance_scale

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output (with classifier-free goal guidance)
            model_output = model(trajectory, t, cond)
            if uncond is not None and guidance != 1.0:
                uncond_output = model(trajectory, t, uncond)
                model_output = uncond_output + guidance * (
                    model_output - uncond_output
                )

            # 3. x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], num_samples: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: stacked node-graph observations, leading dims (B, To).
        num_samples: candidate trajectories per observation. Defaults to the
            configured ``self.num_samples`` (8). Denoising is vectorized over
            B*num_samples. Legacy ``action`` and ``action_pred`` contain the
            first candidate with the original B-leading shape; all samples are
            exposed through the new candidate keys.
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        if num_samples is None:
            num_samples = self.num_samples
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        # node graph observations are already normalized features
        obs = obs_dict
        value = next(iter(obs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        cond = self._encode_obs(obs)
        uncond = None
        if self.goal_guidance_scale != 1.0:
            uncond = self._encode_obs(obs, drop_goal=True)

        if num_samples > 1:
            cond = cond.repeat_interleave(num_samples, dim=0)
            if uncond is not None:
                uncond = uncond.repeat_interleave(num_samples, dim=0)

        n_batch = B * num_samples
        shape = (n_batch, T, Da)
        if self.pred_action_steps_only:
            shape = (n_batch, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            uncond=uncond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        action_candidates = action.reshape(
            B, num_samples, *action.shape[1:]
        )
        trajectory_candidates = action_pred.reshape(
            B, num_samples, *action_pred.shape[1:]
        )
        # A policy-only call has no environment validity signal.  Scores are
        # initialized neutrally and are replaced by NavigationWaypointRanker
        # in closed-loop evaluation.
        candidate_scores = torch.zeros(
            (B, num_samples), device=action.device, dtype=action.dtype
        )
        result = {
            # Backward-compatible keys and shapes.
            'action': action_candidates[:, 0],
            'action_pred': trajectory_candidates[:, 0],
            # Explicit multi-sample API.
            'waypoint_candidates': action_candidates,
            'trajectory_candidates': trajectory_candidates,
            'candidate_scores': candidate_scores,
            'selected_waypoint': action_candidates[:, 0],
            'selected_index': torch.zeros(
                B, device=action.device, dtype=torch.long
            ),
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        To = self.n_obs_steps

        obs = batch['obs']
        # goal dropout for classifier-free guidance: zero the goal-relative
        # node features for a random subset of samples
        if self.training and self.goal_drop_prob > 0.0:
            node_inputs = obs['node_inputs']
            drop = (
                torch.rand(batch_size, device=node_inputs.device)
                < self.goal_drop_prob
            )
            if drop.any():
                node_inputs = node_inputs.clone()
                node_inputs[drop, ..., GOAL_FEATURE_SLICE] = 0.0
                obs = dict(obs)
                obs['node_inputs'] = node_inputs

        cond = self._encode_obs(obs)
        trajectory = nactions
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        model = getattr(self, 'parallel_model', self.model)
        pred = model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
