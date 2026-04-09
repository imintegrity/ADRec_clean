import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from common import SiLU, TransformerEncoder
from utils import _extract_into_tensor,exponential_mapping
from step_sample import *
from mamba_denoiser import MambaDenoiser, TimestepConditionedMambaDenoiser, StatePreservingConditionalMambaDenoiser, AdaLNConditionalMambaDenoiser

class DenoisedModel(nn.Module):
    def __init__(self, args):
        super(DenoisedModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.decoder_type = args.dif_decoder
        self.diffusion_steps = args.diffusion_steps
        self.rescale_timesteps = args.rescale_timesteps
        if args.dif_decoder =='mlp':
            self.decoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                        SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size),
                                        nn.LayerNorm(self.hidden_size),
                                        )
        elif args.dif_decoder == 'mamba':
            self.decoder = MambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2))
        elif args.dif_decoder == 'mamba_tcond':
            self.decoder = TimestepConditionedMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), placement='full')
        elif args.dif_decoder == 'mamba_tcond_ssm':
            self.decoder = TimestepConditionedMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), placement='ssm')
        elif args.dif_decoder == 'mamba_tcond_ffn':
            self.decoder = TimestepConditionedMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), placement='ffn')
        elif args.dif_decoder == 'mamba_tcond_input':
            self.decoder = TimestepConditionedMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), placement='input')
        elif args.dif_decoder == 'spc_mamba_nogate':
            self.decoder = StatePreservingConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='nogate')
        elif args.dif_decoder == 'spc_mamba':
            self.decoder = StatePreservingConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='gated')
        elif args.dif_decoder == 'mamba_adaln_only':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='adaln_only')
        elif args.dif_decoder == 'mamba_tcond_input_adaln':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_localattn_last':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_localattn_last')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_localattn_all':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_localattn_all')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_stage_route_m':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_stage_route_m')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_stage_route_f':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_stage_route_f')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_stage_route_both':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_stage_route_both')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_ffn_stage_adapter':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_ffn_stage_adapter')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_ffn_tsm_adapter':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_ffn_tsm_adapter')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_ffn_tsm_adapter_noglobal':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln_ffn_tsm_adapter_noglobal')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_item_consistency_all':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_item_consistency_snr':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_stationary_latent_all':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln')
        elif args.dif_decoder == 'mamba_tcond_input_adaln_stationary_latent_snr':
            self.decoder = AdaLNConditionalMambaDenoiser(args, num_blocks=getattr(args, 'dif_blocks', 2), mode='tcond_input_adaln')
        else:
            self.decoder = TransformerEncoder(args,num_blocks=2,norm_first=False,hidden_size=self.hidden_size)

        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                        SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size)
                                        )

        self.lambda_uncertainty = args.lambda_uncertainty


    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        assert dim % 2 == 0
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward_cfg(self,c, x, t, mask_seq,mask_tgt,cfg_scale=1.0):
        cond_eps = self.forward(c,x, t,mask_seq,mask_tgt)
        uncond_eps = self.forward(c,x, t,mask_seq,mask_tgt,condition=False)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return eps


    def forward(self, rep_item, x_t, t, mask_seq,mask_tgt,condition=True):
        if condition is not True:  #CFG
            rep_item = torch.zeros_like(rep_item)
            # mask = torch.rand_like(mask_seq) > 0.5
            # rep_item = torch.where(mask.unsqueeze(-1), torch.zeros_like(rep_item), rep_item)
        t=t.reshape(x_t.shape[0],-1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        raw_t = t.float()
        if self.rescale_timesteps:
            raw_t = raw_t / (1000.0 / self.diffusion_steps)
        raw_t = raw_t.clamp(min=0, max=self.diffusion_steps - 1)
        stage_num_buckets = getattr(self.decoder, 'stage_num_buckets', getattr(self.decoder, 'route_num_stages', 1))
        stage_ids = torch.div(raw_t * stage_num_buckets, self.diffusion_steps, rounding_mode='floor').long().clamp(min=0, max=stage_num_buckets - 1)
        lambda_uncertainty = self.lambda_uncertainty  ### fixed

        rep_diffu = rep_item + lambda_uncertainty * (x_t + time_emb)
        state_input = x_t + time_emb

        if self.decoder_type == 'mlp':
            rep_diffu = self.decoder(rep_diffu)
        elif self.decoder_type in {'spc_mamba_nogate', 'spc_mamba'}:
            rep_diffu = self.decoder(state_input, rep_item, mask_seq, time_emb)
        elif self.decoder_type == 'mamba_adaln_only':
            rep_diffu = self.decoder(state_input, rep_item, mask_seq, time_emb)
        elif self.decoder_type in {
            'mamba_tcond_input_adaln',
            'mamba_tcond_input_adaln_localattn_last',
            'mamba_tcond_input_adaln_localattn_all',
            'mamba_tcond_input_adaln_stage_route_m',
            'mamba_tcond_input_adaln_stage_route_f',
            'mamba_tcond_input_adaln_stage_route_both',
            'mamba_tcond_input_adaln_ffn_stage_adapter',
            'mamba_tcond_input_adaln_ffn_tsm_adapter',
            'mamba_tcond_input_adaln_ffn_tsm_adapter_noglobal',
            'mamba_tcond_input_adaln_item_consistency_all',
            'mamba_tcond_input_adaln_item_consistency_snr',
            'mamba_tcond_input_adaln_stationary_latent_all',
            'mamba_tcond_input_adaln_stationary_latent_snr',
        }:
            rep_diffu = self.decoder(rep_diffu, rep_item, mask_seq, time_emb, stage_ids=stage_ids)
        elif self.decoder_type in {'mamba_tcond', 'mamba_tcond_ssm', 'mamba_tcond_ffn', 'mamba_tcond_input'}:
            rep_diffu = self.decoder(rep_diffu, mask_seq, time_emb)
        else:
            rep_diffu = self.decoder(rep_diffu, mask_seq)

        return rep_diffu

class AdRec(nn.Module):
    def __init__(self, args,):
        super(AdRec, self).__init__()

        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        betas = get_named_beta_schedule(args)
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        # print(self.posterior_mean_coef1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        # self.xstart_model = self.dif_model(args)
        self.net = DenoisedModel(args)
        self.independent_diffusion = args.independent
        self.cfg_scale = args.cfg_scale
        self.geodesic = args.geodesic
        self.ag_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False)
        self.stationary_latent_mode = {
            'mamba_tcond_input_adaln_stationary_latent_all': 'all',
            'mamba_tcond_input_adaln_stationary_latent_snr': 'snr',
        }.get(args.dif_decoder, 'none')
        self.stationary_anchor_scale = getattr(args, 'stationary_anchor_scale', 0.25)
        self.stationary_anchor_max_scale = getattr(args, 'stationary_anchor_max_scale', self.stationary_anchor_scale)
        if self.stationary_anchor_max_scale is None:
            self.stationary_anchor_max_scale = self.stationary_anchor_scale
        self.stationary_shift_norm_cap = getattr(args, 'stationary_shift_norm_cap', 0.0)
        self.item_consistency_warmup_epochs = getattr(args, 'item_consistency_warmup_epochs', 0)
        self.item_consistency_ramp_epochs = getattr(args, 'item_consistency_ramp_epochs', 0)
        self.use_positive_negative_guidance = getattr(args, 'use_positive_negative_guidance', False)
        self.negative_condition_source = getattr(args, 'negative_condition_source', 'confusing_topk_from_main_ce')
        self.negative_condition_topk = max(int(getattr(args, 'negative_condition_topk', 10)), 1)
        self.png_guidance_scale = float(getattr(args, 'png_guidance_scale', 0.0))
        self.negative_condition_chunk_size = max(int(getattr(args, 'item_consistency_chunk_size', 2048)), 1)
        self.prediction_target_mode = getattr(args, 'prediction_target_mode', 'item_emb')
        self.pref_teacher_topk = max(int(getattr(args, 'pref_teacher_topk', 20)), 1)
        self.pref_teacher_temperature = float(getattr(args, 'pref_teacher_temperature', 1.0))
        self.pref_mix_topk_alpha = float(getattr(args, 'pref_mix_topk_alpha', 0.30))
        self.pref_mix_stationary_beta = float(getattr(args, 'pref_mix_stationary_beta', 0.20))
        self.generative_process_mode = getattr(args, 'generative_process_mode', 'diffusion')
        self.flow_source_mode = getattr(args, 'flow_source_mode', 'stationary_anchor')
        self.flow_source_hist_blend_rho = float(getattr(args, 'flow_source_hist_blend_rho', 0.0))
        self.flow_num_steps = max(int(getattr(args, 'flow_num_steps', self.diffusion_steps)), 1)
        self.flow_time_schedule = getattr(args, 'flow_time_schedule', 'linear')
        self.flow_loss_weight = float(getattr(args, 'flow_loss_weight', 1.0))
        self.trajectory_consistency_mode = getattr(args, 'trajectory_consistency_mode', 'td_adjacent')
        self.td_delta_step = max(int(getattr(args, 'td_delta_step', 1)), 1)
        self.td_loss_weight = float(getattr(args, 'td_loss_weight', 0.05))
        self.td_weighting_mode = getattr(args, 'td_weighting_mode', 'snr')
        self.current_epoch = 0
        self.current_stationary_anchor_scale = 0.0
        self.latest_stats = {}
        self.set_curriculum_epoch(0)
        if self.negative_condition_source not in {'confusing_topk_from_main_ce'}:
            raise ValueError(f"unsupported negative_condition_source: {self.negative_condition_source}")
        if self.generative_process_mode not in {'diffusion', 'flow_matching'}:
            raise ValueError(f"unsupported generative_process_mode: {self.generative_process_mode}")
        if self.flow_source_mode not in {'stationary_anchor', 'stationary_anchor_blend_last'}:
            raise ValueError(f"unsupported flow_source_mode: {self.flow_source_mode}")
        if self.flow_time_schedule not in {'linear'}:
            raise ValueError(f"unsupported flow_time_schedule: {self.flow_time_schedule}")
        if not (0.0 <= self.flow_source_hist_blend_rho < 1.0):
            raise ValueError("flow_source_hist_blend_rho must be in [0, 1)")
        if self.flow_loss_weight <= 0:
            raise ValueError("flow_loss_weight must be positive")
        if self.trajectory_consistency_mode not in {'none', 'td_adjacent'}:
            raise ValueError(f"unsupported trajectory_consistency_mode: {self.trajectory_consistency_mode}")
        if self.td_weighting_mode not in {'none', 'snr'}:
            raise ValueError(f"unsupported td_weighting_mode: {self.td_weighting_mode}")
        if self.td_loss_weight < 0:
            raise ValueError("td_loss_weight must be non-negative")
        if self.prediction_target_mode not in {'item_emb', 'pref_state'}:
            raise ValueError(f"unsupported prediction_target_mode: {self.prediction_target_mode}")
        if self.pref_teacher_temperature <= 0:
            raise ValueError("pref_teacher_temperature must be positive")
        if self.pref_mix_topk_alpha < 0 or self.pref_mix_stationary_beta < 0:
            raise ValueError("preference-state mixing weights must be non-negative")
        if self.pref_mix_topk_alpha + self.pref_mix_stationary_beta >= 1.0:
            raise ValueError("pref_mix_topk_alpha + pref_mix_stationary_beta must be < 1.0")

    def _compute_curriculum_weight(self, epoch, max_weight, warmup_epochs, ramp_epochs):
        max_weight = float(max(max_weight, 0.0))
        if max_weight == 0.0:
            return 0.0, 0.0
        if epoch < warmup_epochs:
            return 0.0, 0.0
        if ramp_epochs <= 0:
            return max_weight, 1.0
        progress = min(max((epoch - warmup_epochs + 1) / ramp_epochs, 0.0), 1.0)
        return max_weight * progress, progress

    def set_curriculum_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self.current_stationary_anchor_scale, ramp_progress = self._compute_curriculum_weight(
            epoch=self.current_epoch,
            max_weight=self.stationary_anchor_max_scale,
            warmup_epochs=self.item_consistency_warmup_epochs,
            ramp_epochs=self.item_consistency_ramp_epochs,
        )
        self.latest_stats.update({
            'stationary_curr_epoch': float(self.current_epoch),
            'stationary_effective_anchor_scale': float(self.current_stationary_anchor_scale),
            'stationary_ramp_progress': float(ramp_progress),
            'stationary_shift_norm_cap': float(self.stationary_shift_norm_cap),
        })

    def _normalize_with_mask(self, tensor, mask):
        return F.normalize(tensor, dim=-1) * mask.unsqueeze(-1).to(tensor.dtype)

    def _masked_mean(self, tensor, mask):
        mask = mask.float()
        denom = mask.sum().clamp_min(1.0)
        return float((tensor * mask).sum().detach().item() / denom.detach().item())

    def _build_stationary_anchor(self, hist_rep, mask_seq):
        anchor_denom = mask_seq.sum(1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        return (hist_rep * mask_seq.unsqueeze(-1)).sum(1, keepdim=True) / anchor_denom

    def _compute_stationary_components(self, hist_rep, item_tag, mask_seq):
        stationary_anchor = self._build_stationary_anchor(hist_rep, mask_seq)
        anchor_shift = stationary_anchor - item_tag
        raw_shift_norm = anchor_shift.norm(dim=-1)
        if self.stationary_shift_norm_cap and self.stationary_shift_norm_cap > 0:
            capped_shift_norm = raw_shift_norm.clamp(max=self.stationary_shift_norm_cap)
            anchor_shift = anchor_shift * (capped_shift_norm / raw_shift_norm.clamp_min(1e-6)).unsqueeze(-1)
        return stationary_anchor, anchor_shift, raw_shift_norm

    def _build_flow_source_target(self, hist_rep, item_tag, mask_seq, mask_tag):
        stationary_anchor = self._build_stationary_anchor(hist_rep, mask_seq)
        hist_last = hist_rep[:, -1:, :]
        if self.flow_source_mode == 'stationary_anchor':
            source_latent = stationary_anchor
        else:
            source_latent = (1.0 - self.flow_source_hist_blend_rho) * stationary_anchor + self.flow_source_hist_blend_rho * hist_last
        source_latent = self._normalize_with_mask(source_latent.expand_as(item_tag), mask_tag)
        target_latent = self._normalize_with_mask(item_tag, mask_tag)
        valid_mask = mask_tag.float()
        source_to_target_cos = (source_latent * target_latent).sum(dim=-1)
        source_norm = source_latent.norm(dim=-1)
        target_norm = target_latent.norm(dim=-1)
        stationary_anchor_norm = stationary_anchor.norm(dim=-1)
        self.latest_stats = {
            'generative_process_mode': self.generative_process_mode,
            'flow_source_mode': self.flow_source_mode,
            'flow_source_hist_blend_rho': float(self.flow_source_hist_blend_rho),
            'flow_num_steps': int(self.flow_num_steps),
            'flow_time_schedule': self.flow_time_schedule,
            'flow_loss_weight': float(self.flow_loss_weight),
            'trajectory_consistency_mode': self.trajectory_consistency_mode,
            'td_delta_step': int(self.td_delta_step),
            'td_loss_weight': float(self.td_loss_weight),
            'td_weighting_mode': self.td_weighting_mode,
            'prediction_target_mode': self.prediction_target_mode,
            'pref_teacher_topk': int(self.pref_teacher_topk),
            'pref_teacher_temperature': float(self.pref_teacher_temperature),
            'pref_mix_topk_alpha': float(self.pref_mix_topk_alpha),
            'pref_mix_stationary_beta': float(self.pref_mix_stationary_beta),
            'stationary_latent_mode': self.stationary_latent_mode,
            'stationary_anchor_scale': float(self.stationary_anchor_scale),
            'stationary_anchor_max_scale': float(self.stationary_anchor_max_scale),
            'stationary_effective_anchor_scale': float(self.current_stationary_anchor_scale),
            'stationary_shift_norm_cap': float(self.stationary_shift_norm_cap),
            'stationary_anchor_norm_mean': self._masked_mean(stationary_anchor_norm.expand_as(mask_tag), valid_mask),
            'stationary_anchor_norm_std': float(stationary_anchor_norm.std(unbiased=False).detach().item()),
            'stationary_raw_shift_norm_mean': 0.0,
            'stationary_raw_shift_norm_std': 0.0,
            'stationary_target_shift_norm_mean': 0.0,
            'stationary_target_shift_norm_std': 0.0,
            'pref_teacher_target_rank_in_topk': 0.0,
            'pref_teacher_entropy': 0.0,
            'pref_teacher_norm': 0.0,
            'pref_target_shift_norm': 0.0,
            'z_pref_hat_to_target_cos': 0.0,
            'z_pref_hat_to_stationary_cos': 0.0,
            'x0_item_top1_acc': 0.0,
            'source_to_target_cos': self._masked_mean(source_to_target_cos, valid_mask),
            'source_norm': self._masked_mean(source_norm, valid_mask),
            'flow_target_norm': self._masked_mean(target_norm, valid_mask),
            'velocity_target_norm': 0.0,
            'velocity_pred_norm': 0.0,
            'endpoint_to_target_cos': 0.0,
            'endpoint_to_stationary_cos': 0.0,
            'x1_item_top1_acc': 0.0,
            'td_loss': 0.0,
            'x0_hat_high_to_target_cos': 0.0,
            'x0_hat_low_to_target_cos': 0.0,
            'td_teacher_to_low_cos': 0.0,
            'td_high_low_gap': 0.0,
        }
        return source_latent, target_latent, stationary_anchor

    def build_stationary_target(self, hist_rep, item_tag, mask_seq, mask_tag):
        if self.stationary_latent_mode == 'none':
            valid_mask = mask_tag.float()
            self.latest_stats = {
                'generative_process_mode': self.generative_process_mode,
                'flow_source_mode': self.flow_source_mode,
                'flow_source_hist_blend_rho': float(self.flow_source_hist_blend_rho),
                'flow_num_steps': int(self.flow_num_steps),
                'flow_time_schedule': self.flow_time_schedule,
                'flow_loss_weight': float(self.flow_loss_weight),
                'trajectory_consistency_mode': self.trajectory_consistency_mode,
                'td_delta_step': int(self.td_delta_step),
                'td_loss_weight': float(self.td_loss_weight),
                'td_weighting_mode': self.td_weighting_mode,
                'prediction_target_mode': self.prediction_target_mode,
                'pref_teacher_topk': int(self.pref_teacher_topk),
                'pref_teacher_temperature': float(self.pref_teacher_temperature),
                'pref_mix_topk_alpha': float(self.pref_mix_topk_alpha),
                'pref_mix_stationary_beta': float(self.pref_mix_stationary_beta),
                'stationary_latent_mode': self.stationary_latent_mode,
                'stationary_anchor_scale': float(self.stationary_anchor_scale),
                'stationary_anchor_max_scale': float(self.stationary_anchor_max_scale),
                'stationary_effective_anchor_scale': float(self.current_stationary_anchor_scale),
                'stationary_shift_norm_cap': float(self.stationary_shift_norm_cap),
                'stationary_anchor_norm_mean': 0.0,
                'stationary_anchor_norm_std': 0.0,
                'stationary_raw_shift_norm_mean': 0.0,
                'stationary_raw_shift_norm_std': 0.0,
                'stationary_target_shift_norm_mean': 0.0,
                'stationary_target_shift_norm_std': 0.0,
                'pref_teacher_target_rank_in_topk': 0.0,
                'pref_teacher_entropy': 0.0,
                'pref_teacher_norm': self._masked_mean(item_tag.norm(dim=-1), valid_mask),
                'pref_target_shift_norm': 0.0,
                'z_pref_hat_to_target_cos': 0.0,
                'z_pref_hat_to_stationary_cos': 0.0,
                'x0_item_top1_acc': 0.0,
                'source_to_target_cos': 0.0,
                'source_norm': 0.0,
                'flow_target_norm': 0.0,
                'velocity_target_norm': 0.0,
                'velocity_pred_norm': 0.0,
                'endpoint_to_target_cos': 0.0,
                'endpoint_to_stationary_cos': 0.0,
                'x1_item_top1_acc': 0.0,
                'td_loss': 0.0,
                'x0_hat_high_to_target_cos': 0.0,
                'x0_hat_low_to_target_cos': 0.0,
                'td_teacher_to_low_cos': 0.0,
                'td_high_low_gap': 0.0,
            }
            return item_tag
        stationary_anchor, anchor_shift, raw_shift_norm = self._compute_stationary_components(hist_rep, item_tag, mask_seq)
        target_latent = item_tag + self.current_stationary_anchor_scale * anchor_shift
        target_latent = target_latent * mask_tag.unsqueeze(-1)
        anchor_norm = stationary_anchor.norm(dim=-1)
        shift_norm = (target_latent - item_tag).norm(dim=-1)
        self.latest_stats = {
            'generative_process_mode': self.generative_process_mode,
            'flow_source_mode': self.flow_source_mode,
            'flow_source_hist_blend_rho': float(self.flow_source_hist_blend_rho),
            'flow_num_steps': int(self.flow_num_steps),
            'flow_time_schedule': self.flow_time_schedule,
            'flow_loss_weight': float(self.flow_loss_weight),
            'trajectory_consistency_mode': self.trajectory_consistency_mode,
            'td_delta_step': int(self.td_delta_step),
            'td_loss_weight': float(self.td_loss_weight),
            'td_weighting_mode': self.td_weighting_mode,
            'prediction_target_mode': self.prediction_target_mode,
            'pref_teacher_topk': int(self.pref_teacher_topk),
            'pref_teacher_temperature': float(self.pref_teacher_temperature),
            'pref_mix_topk_alpha': float(self.pref_mix_topk_alpha),
            'pref_mix_stationary_beta': float(self.pref_mix_stationary_beta),
            'stationary_latent_mode': self.stationary_latent_mode,
            'stationary_anchor_scale': float(self.stationary_anchor_scale),
            'stationary_anchor_max_scale': float(self.stationary_anchor_max_scale),
            'stationary_effective_anchor_scale': float(self.current_stationary_anchor_scale),
            'stationary_shift_norm_cap': float(self.stationary_shift_norm_cap),
            'stationary_anchor_norm_mean': float(anchor_norm.mean().detach().item()),
            'stationary_anchor_norm_std': float(anchor_norm.std(unbiased=False).detach().item()),
            'stationary_raw_shift_norm_mean': float(raw_shift_norm.mean().detach().item()),
            'stationary_raw_shift_norm_std': float(raw_shift_norm.std(unbiased=False).detach().item()),
            'stationary_target_shift_norm_mean': float(shift_norm.mean().detach().item()),
            'stationary_target_shift_norm_std': float(shift_norm.std(unbiased=False).detach().item()),
            'pref_teacher_target_rank_in_topk': 0.0,
            'pref_teacher_entropy': 0.0,
            'pref_teacher_norm': 0.0,
            'pref_target_shift_norm': 0.0,
            'z_pref_hat_to_target_cos': 0.0,
            'z_pref_hat_to_stationary_cos': 0.0,
            'x0_item_top1_acc': 0.0,
            'source_to_target_cos': 0.0,
            'source_norm': 0.0,
            'flow_target_norm': 0.0,
            'velocity_target_norm': 0.0,
            'velocity_pred_norm': 0.0,
            'endpoint_to_target_cos': 0.0,
            'endpoint_to_stationary_cos': 0.0,
            'x1_item_top1_acc': 0.0,
            'td_loss': 0.0,
            'x0_hat_high_to_target_cos': 0.0,
            'x0_hat_low_to_target_cos': 0.0,
            'td_teacher_to_low_cos': 0.0,
            'td_high_low_gap': 0.0,
        }
        return target_latent

    def _build_pref_teacher_topk(self, teacher_state, target_labels, item_embedding_weight, valid_mask):
        item_weight_detached = item_embedding_weight.detach()
        item_emb_for_logits = F.normalize(item_weight_detached, dim=-1)
        teacher_state = F.normalize(teacher_state.detach(), dim=-1)
        flat_valid_mask = valid_mask.reshape(-1)
        flat_teacher_state = teacher_state.reshape(-1, teacher_state.size(-1))[flat_valid_mask]
        flat_target_labels = target_labels.reshape(-1)[flat_valid_mask]
        flat_prototype = teacher_state.new_zeros((flat_teacher_state.size(0), teacher_state.size(-1)))
        flat_entropy = teacher_state.new_zeros(flat_teacher_state.size(0))
        flat_rank = teacher_state.new_zeros(flat_teacher_state.size(0))
        if flat_teacher_state.size(0) == 0:
            return flat_prototype, flat_entropy, flat_rank

        max_available = max(item_emb_for_logits.size(0) - 1, 1)
        teacher_topk = min(self.pref_teacher_topk, max_available)
        for start in range(0, flat_teacher_state.size(0), self.negative_condition_chunk_size):
            end = min(start + self.negative_condition_chunk_size, flat_teacher_state.size(0))
            chunk_state = flat_teacher_state[start:end]
            chunk_labels = flat_target_labels[start:end]
            chunk_logits = torch.matmul(chunk_state, item_emb_for_logits.t())
            if chunk_logits.size(-1) > 0:
                chunk_logits[:, 0] = float('-inf')
            chunk_topk_logits, chunk_topk_indices = torch.topk(chunk_logits, k=teacher_topk, dim=-1)
            target_in_topk = chunk_topk_indices.eq(chunk_labels.unsqueeze(-1))
            missing_mask = ~target_in_topk.any(dim=-1)
            if missing_mask.any():
                chunk_topk_indices[missing_mask, -1] = chunk_labels[missing_mask]
                chunk_topk_logits[missing_mask, -1] = chunk_logits[missing_mask, chunk_labels[missing_mask]]
            sorted_logits, sorted_order = torch.sort(chunk_topk_logits, dim=-1, descending=True)
            sorted_indices = chunk_topk_indices.gather(-1, sorted_order)
            teacher_probs = F.softmax(sorted_logits / self.pref_teacher_temperature, dim=-1)
            flat_prototype[start:end] = (teacher_probs.unsqueeze(-1) * item_weight_detached[sorted_indices]).sum(dim=1)
            flat_entropy[start:end] = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1)
            flat_rank[start:end] = sorted_indices.eq(chunk_labels.unsqueeze(-1)).float().argmax(dim=-1).float() + 1.0
        return flat_prototype, flat_entropy, flat_rank

    def build_diffusion_target(self, hist_rep, item_tag, mask_seq, mask_tag, item_embedding_weight=None, labels=None):
        stationary_target = self.build_stationary_target(hist_rep, item_tag, mask_seq, mask_tag)
        stationary_anchor, _, _ = self._compute_stationary_components(hist_rep, item_tag, mask_seq)
        valid_mask = mask_tag > 0
        if self.prediction_target_mode != 'pref_state':
            return stationary_target, stationary_anchor

        if item_embedding_weight is None or labels is None:
            raise ValueError("pref_state mode requires item_embedding_weight and labels")

        teacher_state = hist_rep
        flat_prototype, flat_entropy, flat_rank = self._build_pref_teacher_topk(
            teacher_state=teacher_state,
            target_labels=labels,
            item_embedding_weight=item_embedding_weight,
            valid_mask=valid_mask,
        )
        p_topk = item_tag.new_zeros(item_tag.shape)
        p_topk.reshape(-1, p_topk.size(-1))[valid_mask.reshape(-1)] = flat_prototype

        target_weight = 1.0 - self.pref_mix_topk_alpha - self.pref_mix_stationary_beta
        z_pref_teacher = (
            target_weight * item_tag
            + self.pref_mix_topk_alpha * p_topk
            + self.pref_mix_stationary_beta * stationary_anchor
        )
        z_pref_teacher = F.normalize(z_pref_teacher, dim=-1)
        z_pref_teacher = z_pref_teacher * mask_tag.unsqueeze(-1)
        z_pref_teacher = z_pref_teacher.detach()

        pref_shift_norm = (z_pref_teacher - item_tag).norm(dim=-1)
        pref_teacher_norm = z_pref_teacher.norm(dim=-1)
        valid_mask_float = valid_mask.float()
        rank_tensor = item_tag.new_zeros(labels.shape)
        rank_tensor = rank_tensor.masked_scatter(valid_mask, flat_rank.to(rank_tensor.dtype))
        entropy_tensor = item_tag.new_zeros(labels.shape)
        entropy_tensor = entropy_tensor.masked_scatter(valid_mask, flat_entropy.to(entropy_tensor.dtype))
        self.latest_stats.update({
            'pref_teacher_target_rank_in_topk': self._masked_mean(rank_tensor, valid_mask_float),
            'pref_teacher_entropy': self._masked_mean(entropy_tensor, valid_mask_float),
            'pref_teacher_norm': self._masked_mean(pref_teacher_norm, valid_mask_float),
            'pref_target_shift_norm': self._masked_mean(pref_shift_norm, valid_mask_float),
        })
        return z_pref_teacher, stationary_anchor

    def _compute_x0_item_top1_acc(self, x0_seq, labels, item_embedding_weight, valid_mask):
        if item_embedding_weight is None or labels is None or not valid_mask.any():
            return 0.0
        flat_x0 = F.normalize(x0_seq, dim=-1).reshape(-1, x0_seq.size(-1))[valid_mask.reshape(-1)]
        flat_labels = labels.reshape(-1)[valid_mask.reshape(-1)]
        item_emb_norm = F.normalize(item_embedding_weight, dim=-1)
        total_correct = 0.0
        total_examples = max(int(flat_labels.numel()), 1)
        chunk_size = max(int(self.negative_condition_chunk_size), 1)
        for start in range(0, flat_labels.size(0), chunk_size):
            end = min(start + chunk_size, flat_labels.size(0))
            chunk_logits = torch.matmul(flat_x0[start:end], item_emb_norm.t())
            total_correct += float((chunk_logits.argmax(dim=-1) == flat_labels[start:end]).float().sum().detach().item())
        return total_correct / total_examples

    def _update_prediction_geometry_stats(self, denoised_seq, item_tag, stationary_anchor, mask_tag, item_embedding_weight=None, labels=None):
        valid_mask = mask_tag > 0
        if not valid_mask.any():
            self.latest_stats.update({
                'z_pref_hat_to_target_cos': 0.0,
                'z_pref_hat_to_stationary_cos': 0.0,
                'x0_item_top1_acc': 0.0,
            })
            return
        x0_norm = F.normalize(denoised_seq, dim=-1)
        target_norm = F.normalize(item_tag, dim=-1)
        stationary_norm = F.normalize(stationary_anchor, dim=-1)
        target_cos = (x0_norm * target_norm).sum(dim=-1)
        stationary_cos = (x0_norm * stationary_norm).sum(dim=-1)
        valid_mask_float = valid_mask.float()
        self.latest_stats.update({
            'z_pref_hat_to_target_cos': self._masked_mean(target_cos, valid_mask_float),
            'z_pref_hat_to_stationary_cos': self._masked_mean(stationary_cos, valid_mask_float),
            'x0_item_top1_acc': float(self._compute_x0_item_top1_acc(denoised_seq, labels, item_embedding_weight, valid_mask)),
        })

    def _update_flow_stats(self, endpoint_seq, velocity_pred, velocity_target, item_tag, stationary_anchor, mask_tag, item_embedding_weight=None, labels=None):
        valid_mask = mask_tag > 0
        if not valid_mask.any():
            self.latest_stats.update({
                'velocity_target_norm': 0.0,
                'velocity_pred_norm': 0.0,
                'endpoint_to_target_cos': 0.0,
                'endpoint_to_stationary_cos': 0.0,
                'x1_item_top1_acc': 0.0,
            })
            return
        valid_mask_float = valid_mask.float()
        endpoint_norm = F.normalize(endpoint_seq, dim=-1)
        target_norm = F.normalize(item_tag, dim=-1)
        stationary_norm = F.normalize(stationary_anchor.expand_as(endpoint_seq), dim=-1)
        endpoint_to_target_cos = (endpoint_norm * target_norm).sum(dim=-1)
        endpoint_to_stationary_cos = (endpoint_norm * stationary_norm).sum(dim=-1)
        self.latest_stats.update({
            'velocity_target_norm': self._masked_mean(velocity_target.norm(dim=-1), valid_mask_float),
            'velocity_pred_norm': self._masked_mean(velocity_pred.norm(dim=-1), valid_mask_float),
            'endpoint_to_target_cos': self._masked_mean(endpoint_to_target_cos, valid_mask_float),
            'endpoint_to_stationary_cos': self._masked_mean(endpoint_to_stationary_cos, valid_mask_float),
            'x1_item_top1_acc': float(self._compute_x0_item_top1_acc(endpoint_seq, labels, item_embedding_weight, valid_mask)),
            'z_pref_hat_to_target_cos': self._masked_mean(endpoint_to_target_cos, valid_mask_float),
            'z_pref_hat_to_stationary_cos': self._masked_mean(endpoint_to_stationary_cos, valid_mask_float),
            'x0_item_top1_acc': float(self._compute_x0_item_top1_acc(endpoint_seq, labels, item_embedding_weight, valid_mask)),
        })

    def _sample_flow_times(self, mask_tag, device, dtype):
        flow_t = torch.rand(mask_tag.shape, device=device, dtype=dtype) * mask_tag.to(dtype)
        flow_t_index = flow_t * float(max(self.num_timesteps - 1, 1))
        return flow_t, flow_t_index

    def _reshape_timesteps_like_target(self, t, target_shape):
        if t.dim() == 1:
            if t.numel() == target_shape[0] * target_shape[1]:
                return t.reshape(target_shape[0], target_shape[1])
            if t.numel() == target_shape[0]:
                return t.unsqueeze(-1).expand(-1, target_shape[1])
        if t.dim() == 2 and t.shape == target_shape[:2]:
            return t
        raise ValueError(f"unexpected timestep shape {tuple(t.shape)} for target shape {tuple(target_shape)}")

    def _apply_td_step_from_high(self, x_t_high, x0_hat_high_detached, t_high_matrix):
        x_curr = x_t_high.detach()
        for step_idx in range(self.td_delta_step):
            current_t = (t_high_matrix - step_idx).clamp(min=0)
            posterior_mean = self.q_posterior_mean_variance(x_start=x0_hat_high_detached, x_t=x_curr, t=current_t)
            update_mask = (current_t > 0).unsqueeze(-1)
            x_curr = torch.where(update_mask, posterior_mean, x0_hat_high_detached)
        return x_curr

    def _td_weight_tensor(self, t_low_matrix, mask_tag, dtype, device):
        if self.td_weighting_mode == 'none':
            return mask_tag.to(dtype)
        alpha_bar = torch.as_tensor(self.alphas_cumprod, device=device, dtype=dtype)
        return alpha_bar[t_low_matrix.long().clamp(min=0, max=self.num_timesteps - 1)] * mask_tag.to(dtype)

    def _compute_td_consistency_loss(self, item_rep, diff_target, x_t_high, t_high, x0_hat_high, mask_seq, mask_tag):
        zero_loss = x0_hat_high.new_zeros(())
        if self.trajectory_consistency_mode == 'none' or self.td_loss_weight <= 0:
            self.latest_stats.update({
                'td_loss': 0.0,
                'x0_hat_high_to_target_cos': 0.0,
                'x0_hat_low_to_target_cos': 0.0,
                'td_teacher_to_low_cos': 0.0,
                'td_high_low_gap': 0.0,
            })
            return zero_loss

        t_high_matrix = self._reshape_timesteps_like_target(t_high, diff_target.shape)
        t_low_matrix = (t_high_matrix - self.td_delta_step).clamp(min=0)
        x_t_low = self.q_sample(
            diff_target.reshape(-1, diff_target.shape[-1]),
            t_low_matrix.reshape(-1),
            mask=mask_tag.reshape(-1),
        ).reshape_as(diff_target)
        x0_hat_low = self.net(item_rep, x_t_low, self._scale_timesteps(t_low_matrix), mask_seq, mask_tag)

        with torch.no_grad():
            x0_hat_high_detached = x0_hat_high.detach()
            x_tdlow_teacher = self._apply_td_step_from_high(x_t_high, x0_hat_high_detached, t_high_matrix)
            x0_tdlow_from_high = self.net(item_rep, x_tdlow_teacher, self._scale_timesteps(t_low_matrix), mask_seq, mask_tag).detach()

        td_weights = self._td_weight_tensor(t_low_matrix, mask_tag, x0_hat_high.dtype, x0_hat_high.device)
        td_weights = td_weights / td_weights.sum(1, keepdim=True).clamp_min(1e-6)
        td_loss = (F.mse_loss(x0_hat_low, x0_tdlow_from_high, reduction='none') * td_weights.unsqueeze(-1)).sum(1).mean()
        td_loss = td_loss * self.td_loss_weight

        valid_mask = mask_tag > 0
        valid_mask_float = valid_mask.float()
        x0_high_norm = F.normalize(x0_hat_high, dim=-1)
        x0_low_norm = F.normalize(x0_hat_low, dim=-1)
        td_teacher_norm = F.normalize(x0_tdlow_from_high, dim=-1)
        diff_target_norm = F.normalize(diff_target, dim=-1)
        self.latest_stats.update({
            'td_loss': float(td_loss.detach().item()),
            'x0_hat_high_to_target_cos': self._masked_mean((x0_high_norm * diff_target_norm).sum(dim=-1), valid_mask_float),
            'x0_hat_low_to_target_cos': self._masked_mean((x0_low_norm * diff_target_norm).sum(dim=-1), valid_mask_float),
            'td_teacher_to_low_cos': self._masked_mean((td_teacher_norm * x0_low_norm).sum(dim=-1), valid_mask_float),
            'td_high_low_gap': self._masked_mean((x0_hat_high - x0_hat_low).norm(dim=-1), valid_mask_float),
        })
        return td_loss

    def _flow_step_schedule(self, device, dtype):
        if self.flow_time_schedule == 'linear':
            return torch.linspace(0.0, 1.0, self.flow_num_steps + 1, device=device, dtype=dtype)
        raise ValueError(f"unsupported flow_time_schedule: {self.flow_time_schedule}")

    def get_latest_stats(self):
        return dict(self.latest_stats)

    def _build_confusing_negative_condition(self, positive_condition, x0_pos, item_embedding_weight, valid_mask, labels=None):
        negative_condition = torch.zeros_like(positive_condition)
        stats = {
            'confusing_negative_score_mean': 0.0,
            'negative_condition_source': self.negative_condition_source,
            'negative_condition_topk': int(self.negative_condition_topk),
        }
        if item_embedding_weight is None or not valid_mask.any():
            return negative_condition, stats

        # Stop gradients through the negative-condition builder so GAL/negative branch
        # only trains the denoiser under the derived repulsive condition.
        item_emb_norm = F.normalize(item_embedding_weight.detach(), dim=-1)
        x0_norm = F.normalize(x0_pos.detach(), dim=-1)
        flat_valid_mask = valid_mask.reshape(-1)
        flat_x0 = x0_norm.reshape(-1, x0_norm.size(-1))[flat_valid_mask]
        all_negative_vectors = []
        all_confusing_scores = []

        if labels is not None:
            flat_labels = labels.reshape(-1)[flat_valid_mask]
        else:
            flat_labels = None

        max_candidates = item_emb_norm.size(0) - 1
        if max_candidates <= 0:
            return negative_condition, stats

        for start in range(0, flat_x0.size(0), self.negative_condition_chunk_size):
            end = min(start + self.negative_condition_chunk_size, flat_x0.size(0))
            chunk_x0 = flat_x0[start:end]
            chunk_logits = torch.matmul(chunk_x0, item_emb_norm.t())
            if chunk_logits.size(-1) > 0:
                chunk_logits[:, 0] = float('-inf')

            if flat_labels is not None:
                chunk_labels = flat_labels[start:end]
                chunk_logits.scatter_(1, chunk_labels.unsqueeze(-1), float('-inf'))
                topk = max(1, min(self.negative_condition_topk, max(chunk_logits.size(-1) - 2, 1)))
                topk_scores, topk_indices = torch.topk(chunk_logits, k=topk, dim=-1)
            else:
                topk = max(1, min(self.negative_condition_topk + 1, max_candidates))
                topk_scores, topk_indices = torch.topk(chunk_logits, k=topk, dim=-1)
                if topk_scores.size(-1) > 1:
                    topk_scores = topk_scores[:, 1:]
                    topk_indices = topk_indices[:, 1:]

            attn_weights = F.softmax(topk_scores, dim=-1)
            negative_vectors = (attn_weights.unsqueeze(-1) * item_emb_norm[topk_indices]).sum(dim=1)
            all_negative_vectors.append(negative_vectors)
            all_confusing_scores.append(topk_scores.mean(dim=-1))

        flat_negative_condition = negative_condition.reshape(-1, negative_condition.size(-1))
        flat_negative_condition[flat_valid_mask] = torch.cat(all_negative_vectors, dim=0).detach()
        confusing_scores = torch.cat(all_confusing_scores, dim=0)
        stats['confusing_negative_score_mean'] = float(confusing_scores.mean().detach().item())
        return negative_condition, stats

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick
        if self.geodesic:
            # exp_x[v] = cos(||v||) * x + sin(||v||) * (v / ||v||)
            x_t = exponential_mapping(x_start, x_t)
        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        # print(t)
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        # print(t[0,-10:])
        # print(posterior_mean[0,-1,0])
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq,mask_tag, item_embedding_weight=None):
        # print("func p_mean_variance", rep_item.shape,x_t.shape)
        if self.cfg_scale==1.:
            x0_pos = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq,mask_tag)
        else:
            x0_pos = self.net.forward_cfg(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag,self.cfg_scale)
        x_0 = x0_pos
        if self.use_positive_negative_guidance and self.png_guidance_scale > 0 and item_embedding_weight is not None:
            valid_mask = mask_tag > 0
            negative_condition, negative_stats = self._build_confusing_negative_condition(
                positive_condition=rep_item,
                x0_pos=x0_pos,
                item_embedding_weight=item_embedding_weight,
                valid_mask=valid_mask,
                labels=None,
            )
            x0_neg = self.net(negative_condition, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
            x_0 = x0_pos + self.png_guidance_scale * (x0_pos - x0_neg)
            self.latest_stats.update(negative_stats)
        # x_0 = model_output.unsqueeze(1)  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict
        # x_0 = x_0.clamp_(-1., 1.)
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq,mask_tag, item_embedding_weight=None):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq,mask_tag, item_embedding_weight=item_embedding_weight)
        noise = th.randn_like(noise_x_t)
        # print("noise shape in func p_sample",noise.shape)
        nonzero_mask = (t != 0).float().unsqueeze(-1)  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        if self.geodesic:
            sample_xt = F.normalize(sample_xt,p=2,dim=-1)
        return sample_xt

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag, item_embedding_weight=None, labels=None):
        seq = self.ag_encoder(seq, mask_seq)
        if self.generative_process_mode == 'flow_matching':
            source_latent, target_latent, stationary_anchor = self._build_flow_source_target(seq, tgt, mask_seq, mask_tag)
            flow_times = self._flow_step_schedule(seq.device, seq.dtype)
            x_curr = source_latent
            last_velocity = torch.zeros_like(x_curr)
            for step_index in range(self.flow_num_steps):
                t_value = flow_times[step_index]
                dt_value = flow_times[step_index + 1] - flow_times[step_index]
                t_tensor = torch.full(mask_tag.shape, float(t_value.item() * max(self.num_timesteps - 1, 1)), device=seq.device, dtype=seq.dtype)
                velocity = self.net(seq, x_curr, self._scale_timesteps(t_tensor), mask_seq, mask_tag)
                x_curr = (x_curr + dt_value * velocity) * mask_tag.unsqueeze(-1)
                last_velocity = velocity
            self._update_flow_stats(
                endpoint_seq=x_curr,
                velocity_pred=last_velocity,
                velocity_target=target_latent - source_latent,
                item_tag=target_latent,
                stationary_anchor=stationary_anchor,
                mask_tag=mask_tag,
                item_embedding_weight=item_embedding_weight,
                labels=labels,
            )
            return x_curr
        item_target = tgt
        tgt, stationary_anchor = self.build_diffusion_target(
            seq,
            tgt,
            mask_seq,
            mask_tag,
            item_embedding_weight=item_embedding_weight,
            labels=labels,
        )
        # return self.xstart_model(item_rep, noise_x_t, th.tensor([1] * item_rep.shape[0], device=item_rep.device), mask_seq)[0]
        noise_x_t = th.randn_like(tgt)
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([0]*(seq.shape[1]-1) + [i], device=seq.device).unsqueeze(0).repeat(seq.shape[0],1)
            noise_x_t = torch.concat([tgt[:, :-1], noise_x_t[:, -1:]], dim=1)
            # noise_x_t = torch.concat([torch.zeros_like(tgt[:, :-1]),noise_x_t[:, -1:]], dim=1)
            noise_x_t = self.p_sample(seq, noise_x_t, t, mask_seq, mask_tag, item_embedding_weight=item_embedding_weight)
        # print(noise_x_t[0,-1,:10])
        self._update_prediction_geometry_stats(
            denoised_seq=noise_x_t,
            item_tag=item_target,
            stationary_anchor=stationary_anchor,
            mask_tag=mask_tag,
            item_embedding_weight=item_embedding_weight,
            labels=labels,
        )
        return noise_x_t

    def independent_diffuse(self, tgt, mask, is_independent=False):
        if is_independent:
            t, weights = self.schedule_sampler.sample(tgt.shape[0] * tgt.shape[1], tgt.device)
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)).reshape(*tgt.shape)
        else:
            t, weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t,t
    def forward(self, item_rep, item_tag, mask_seq,mask_tag, item_embedding_weight=None, labels=None):
        item_rep = self.ag_encoder(item_rep, mask_seq)
        if self.generative_process_mode == 'flow_matching':
            source_latent, target_latent, stationary_anchor = self._build_flow_source_target(item_rep, item_tag, mask_seq, mask_tag)
            flow_t, flow_t_index = self._sample_flow_times(mask_tag, item_rep.device, item_rep.dtype)
            x_t = ((1.0 - flow_t).unsqueeze(-1) * source_latent + flow_t.unsqueeze(-1) * target_latent) * mask_tag.unsqueeze(-1)
            velocity_target = (target_latent - source_latent) * mask_tag.unsqueeze(-1)
            velocity_pred = self.net(item_rep, x_t, self._scale_timesteps(flow_t_index), mask_seq,mask_tag)
            endpoint_seq = x_t + (1.0 - flow_t).unsqueeze(-1) * velocity_pred
            endpoint_seq = endpoint_seq * mask_tag.unsqueeze(-1)
            self._update_flow_stats(
                endpoint_seq=endpoint_seq,
                velocity_pred=velocity_pred,
                velocity_target=velocity_target,
                item_tag=target_latent,
                stationary_anchor=stationary_anchor,
                mask_tag=mask_tag,
                item_embedding_weight=item_embedding_weight,
                labels=labels,
            )
            flow_weights = (mask_tag / mask_tag.sum(1, keepdim=True).clamp_min(1.0)).unsqueeze(-1)
            losses = F.mse_loss(velocity_pred, velocity_target, reduction='none') * flow_weights
            losses = losses.sum(1).mean() * self.flow_loss_weight
            rounded_t = flow_t_index.round().long().clamp(min=0, max=self.num_timesteps - 1)
            return endpoint_seq, losses, rounded_t, None
        diff_target, stationary_anchor = self.build_diffusion_target(
            item_rep,
            item_tag,
            mask_seq,
            mask_tag,
            item_embedding_weight=item_embedding_weight,
            labels=labels,
        )
        x_t,t = self.independent_diffuse(diff_target, mask_tag, self.independent_diffusion)
        if self.cfg_scale != 1:
            mask = torch.rand([mask_seq.shape[0],1,1],device=item_rep.device) > 0.7
            item_rep = torch.where(mask,torch.zeros_like(item_rep),item_rep)
        denoised_seq = self.net(item_rep, x_t, self._scale_timesteps(t), mask_seq,mask_tag)  ##output predict
        td_loss = self._compute_td_consistency_loss(
            item_rep=item_rep,
            diff_target=diff_target,
            x_t_high=x_t,
            t_high=t,
            x0_hat_high=denoised_seq,
            mask_seq=mask_seq,
            mask_tag=mask_tag,
        )
        png_payload = None
        if self.use_positive_negative_guidance and item_embedding_weight is not None:
            negative_condition, negative_stats = self._build_confusing_negative_condition(
                positive_condition=item_rep,
                x0_pos=denoised_seq,
                item_embedding_weight=item_embedding_weight,
                valid_mask=mask_tag > 0,
                labels=labels,
            )
            denoised_seq_neg = self.net(negative_condition, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
            self.latest_stats.update(negative_stats)
            png_payload = {
                'x0_neg': denoised_seq_neg,
                'negative_condition': negative_condition,
                'confusing_negative_score_mean': negative_stats['confusing_negative_score_mean'],
            }
        self._update_prediction_geometry_stats(
            denoised_seq=denoised_seq,
            item_tag=item_tag,
            stationary_anchor=stationary_anchor,
            mask_tag=mask_tag,
            item_embedding_weight=item_embedding_weight,
            labels=labels,
        )
        # print(denoised_seq.shape,item_tag.shape,mask_tag.shape)
        losses = F.mse_loss(denoised_seq,diff_target, reduction='none')* (mask_tag / mask_tag.sum(1,keepdim=True)).unsqueeze(-1)
        losses = losses.sum(1).mean() + td_loss
        return denoised_seq, losses, t, png_payload





