import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from diffurec import DiffuRec
from adrec import AdRec
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th
import einops
import os
from common import *
from dreamrec import DreamRec
class Att_Diffuse_model(nn.Module):
    def __init__(self, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.args=args
        self.item_num = args.item_num
        self.item_embedding = self.embed_item(pretrained=args.pretrained)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        # self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        # self.position_embeddings = RotaryPositionalEmbeddings(args.hidden_size)
        self.hist_norm = LayerNorm(args.hidden_size, eps=1e-12)
        # self.tgt_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = create_model_diffu(args)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)
        # self.loss_mse = nn.MSELoss()
        # self.per_token_ag = args.per_token_ag

        self.geodesic = args.geodesic
        self.item_consistency_mode = {
            'mamba_tcond_input_adaln_item_consistency_all': 'all',
            'mamba_tcond_input_adaln_item_consistency_snr': 'snr',
            'mamba_tcond_input_adaln_stationary_latent_all': 'all',
            'mamba_tcond_input_adaln_stationary_latent_snr': 'snr',
        }.get(args.dif_decoder, 'none')
        self.lambda_item = getattr(args, 'lambda_item', 0.0)
        self.item_consistency_max_weight = getattr(args, 'item_consistency_max_weight', self.lambda_item)
        if self.item_consistency_max_weight is None:
            self.item_consistency_max_weight = self.lambda_item
        self.item_consistency_warmup_epochs = getattr(args, 'item_consistency_warmup_epochs', 0)
        self.item_consistency_ramp_epochs = getattr(args, 'item_consistency_ramp_epochs', 0)
        self.item_alignment_mode = getattr(args, 'item_alignment_mode', 'ce')
        self.item_alignment_temperature = getattr(
            args,
            'item_alignment_temperature',
            getattr(args, 'item_consistency_temperature', 0.07),
        )
        self.item_alignment_topk = getattr(args, 'item_alignment_topk', 50)
        self.item_alignment_kd_temperature = getattr(args, 'item_alignment_kd_temperature', 1.0)
        self.item_alignment_teacher_source = getattr(args, 'item_alignment_teacher_source', 'main_ce_head')
        self.item_alignment_margin = getattr(args, 'item_alignment_margin', 0.10)
        self.item_alignment_num_negatives = getattr(args, 'item_alignment_num_negatives', 32)
        self.item_alignment_negative_source = getattr(args, 'item_alignment_negative_source', 'inbatch_then_random')
        self.item_consistency_temperature = getattr(args, 'item_consistency_temperature', self.item_alignment_temperature)
        self.item_consistency_snr_power = getattr(args, 'item_consistency_snr_power', 1.0)
        self.item_consistency_chunk_size = getattr(args, 'item_consistency_chunk_size', 2048)
        self.use_positive_negative_guidance = getattr(args, 'use_positive_negative_guidance', False)
        self.negative_condition_source = getattr(args, 'negative_condition_source', 'confusing_topk_from_main_ce')
        self.negative_condition_topk = max(int(getattr(args, 'negative_condition_topk', 10)), 1)
        self.png_guidance_scale = float(getattr(args, 'png_guidance_scale', 0.0))
        self.gal_margin = float(getattr(args, 'gal_margin', 0.1))
        self.gal_weight = float(getattr(args, 'gal_weight', 0.05))
        self.prediction_target_mode = getattr(args, 'prediction_target_mode', 'item_emb')
        self.pref_teacher_topk = max(int(getattr(args, 'pref_teacher_topk', 20)), 1)
        self.pref_teacher_temperature = float(getattr(args, 'pref_teacher_temperature', 1.0))
        self.pref_mix_topk_alpha = float(getattr(args, 'pref_mix_topk_alpha', 0.30))
        self.pref_mix_stationary_beta = float(getattr(args, 'pref_mix_stationary_beta', 0.20))
        self.generative_process_mode = getattr(args, 'generative_process_mode', 'flow_matching')
        self.flow_source_mode = getattr(args, 'flow_source_mode', 'stationary_anchor')
        self.flow_source_hist_blend_rho = float(getattr(args, 'flow_source_hist_blend_rho', 0.0))
        self.flow_num_steps = max(int(getattr(args, 'flow_num_steps', getattr(args, 'diffusion_steps', 32))), 1)
        self.flow_time_schedule = getattr(args, 'flow_time_schedule', 'linear')
        self.flow_loss_weight = float(getattr(args, 'flow_loss_weight', 1.0))
        if self.item_alignment_mode not in {'ce', 'topk_kd', 'pref_ratio', 'cosine_margin_ce'}:
            raise ValueError(f"unsupported item_alignment_mode: {self.item_alignment_mode}")
        if self.item_alignment_teacher_source not in {'main_ce_head'}:
            raise ValueError(f"unsupported item_alignment_teacher_source: {self.item_alignment_teacher_source}")
        if self.item_alignment_negative_source not in {'inbatch', 'random', 'inbatch_then_random'}:
            raise ValueError(f"unsupported item_alignment_negative_source: {self.item_alignment_negative_source}")
        self.current_epoch = 0
        self.current_item_consistency_weight = 0.0
        self.latest_item_consistency_stats = {}
        self.latest_png_stats = {}
        self.set_curriculum_epoch(0)
    def load_pretrained_emb_weight(self):

        path = os.path.join('saved','pretrain',self.args.dataset, 'pretrain.pth')
        # path = path_dict[dataset_name]
        saved = torch.load(path, map_location='cpu',weights_only=False)
        pretrained_emb_weight = saved['item_embedding.weight']
        return pretrained_emb_weight
    def embed_item(self,pretrained=False):
        if pretrained:
            embedding = nn.Embedding.from_pretrained(
                self.load_pretrained_emb_weight(), padding_idx=0, freeze=self.args.freeze_emb
            )
        else:
            embedding = nn.Embedding(self.item_num+1, self.emb_dim, padding_idx=0)
        return embedding


    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embedding.weight.t())
        scores_pos = scores.gather(1 , labels)  ## labels: b x 1
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1)-scores_pos)/(scores.shape[1]-1)

        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
       
        # if isinstance(self.diffu.schedule_sampler, LossAwareSampler):
        #     self.diffu.schedule_sampler.update_with_all_losses(t, loss.detach())
        # loss = (loss * weights).mean()
        return loss

    def calculate_loss_minibatch(self, out_seq, labels, batch_size=128):
        """
        计算批量损失，按批次维度计算，以节省显存。

        :param out_seq: Tensor, 形状 (B, L, K) - 用户的输出序列表示
        :param labels: Tensor, 形状 (B, L) - 用户与物品的标签
        :param batch_size: int, 每批次的大小
        :return: loss
        """
        # 获取物品嵌入矩阵的转置
        item_embeddings = self.item_embedding.weight.t()

        # 获取样本数量 (即批次大小 B)
        num_batches = out_seq.shape[0]  # B (批次维度)

        # 损失变量
        total_loss = 0.0
        num = num_batches//batch_size
        # 按批次计算损失
        for i in range(0, num_batches, batch_size):
            # 获取当前批次的切片
            batch_out_seq = out_seq[i:i + batch_size]  # 形状 (B', L, K)
            batch_labels = labels[i:i + batch_size]  # 形状 (B', L)

            # 计算当前批次的分数（B' x L x K）与物品嵌入的矩阵乘积
            scores = torch.matmul(batch_out_seq, item_embeddings)  # 形状 (B', L, num_items)

            # 计算损失：需要将 `scores` 和 `batch_labels` 展平以计算交叉熵损失
            loss = self.loss_ce(scores.reshape(-1, scores.shape[-1]), batch_labels.reshape(-1))

            # 累加损失
            total_loss += loss

        # 返回平均损失
        return total_loss / num
    def calculate_loss(self, out_seq, labels):
        index = labels>0
        out_seq = out_seq[index]
        labels = labels[index]
        # if self.args.dataset == 'yelp':
        #     loss = self.calculate_loss_minibatch(out_seq, labels)
        # else:
        scores = torch.matmul(out_seq, self.item_embedding.weight.t()) #B,L,K
        loss = self.loss_ce(scores.reshape(-1, scores.shape[-1]), labels.reshape(-1))

        #
        # else:
        #     scores = torch.matmul(last_item, self.item_embeddings.weight.t()) #B,K labels: B
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        return loss
        # return self.loss_ce(scores, labels.squeeze(-1))

    def calculate_score(self, item):
        scores = torch.matmul(item.reshape(-1, item.shape[-1]), self.item_embedding.weight.t())
        return scores

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
        self.current_item_consistency_weight, ramp_progress = self._compute_curriculum_weight(
            epoch=self.current_epoch,
            max_weight=self.item_consistency_max_weight,
            warmup_epochs=self.item_consistency_warmup_epochs,
            ramp_epochs=self.item_consistency_ramp_epochs,
        )
        if hasattr(self.diffu, 'set_curriculum_epoch'):
            self.diffu.set_curriculum_epoch(self.current_epoch)
        self.latest_item_consistency_stats.update({
            'item_consistency_curr_epoch': float(self.current_epoch),
            'item_consistency_effective_weight': float(self.current_item_consistency_weight),
            'item_consistency_ramp_progress': float(ramp_progress),
        })

    def get_current_item_consistency_weight(self):
        return float(self.current_item_consistency_weight)

    def get_gal_weight(self):
        return float(self.gal_weight if self.use_positive_negative_guidance else 0.0)

    def _reset_png_stats(self):
        self.latest_png_stats = {
            'use_positive_negative_guidance': bool(self.use_positive_negative_guidance),
            'negative_condition_source': self.negative_condition_source,
            'negative_condition_topk': int(self.negative_condition_topk),
            'png_guidance_scale': float(self.png_guidance_scale),
            'gal_margin': float(self.gal_margin),
            'gal_weight': float(self.gal_weight),
            'gal_loss': 0.0,
            'pos_branch_target_cos': 0.0,
            'neg_branch_target_cos': 0.0,
            'pos_minus_neg_target_gap': 0.0,
            'confusing_negative_score_mean': 0.0,
            'x0_pos_item_top1_acc': 0.0,
            'x0_neg_item_top1_acc': 0.0,
            'prediction_target_mode': self.prediction_target_mode,
            'pref_teacher_topk': int(self.pref_teacher_topk),
            'pref_teacher_temperature': float(self.pref_teacher_temperature),
            'pref_mix_topk_alpha': float(self.pref_mix_topk_alpha),
            'pref_mix_stationary_beta': float(self.pref_mix_stationary_beta),
            'generative_process_mode': self.generative_process_mode,
            'flow_source_mode': self.flow_source_mode,
            'flow_source_hist_blend_rho': float(self.flow_source_hist_blend_rho),
            'flow_num_steps': int(self.flow_num_steps),
            'flow_time_schedule': self.flow_time_schedule,
            'flow_loss_weight': float(self.flow_loss_weight),
        }

    def _compute_branch_top1_acc(self, branch_seq, labels):
        valid_mask = labels > 0
        if not valid_mask.any():
            return 0.0
        branch_norm = F.normalize(branch_seq, dim=-1)
        item_emb_norm = F.normalize(self.item_embedding.weight, dim=-1)
        flat_branch = branch_norm[valid_mask]
        flat_labels = labels[valid_mask]
        total_correct = 0.0
        total_examples = int(flat_labels.numel())
        chunk_size = max(int(self.item_consistency_chunk_size), 1)
        for start in range(0, flat_labels.size(0), chunk_size):
            end = min(start + chunk_size, flat_labels.size(0))
            logits = torch.matmul(flat_branch[start:end], item_emb_norm.t())
            preds = logits.argmax(dim=-1)
            total_correct += float((preds == flat_labels[start:end]).float().sum().detach().item())
        return total_correct / max(total_examples, 1)

    def compute_positive_negative_guidance_loss(self, x0_pos, png_payload, target_embeddings, labels):
        self._reset_png_stats()
        if not self.use_positive_negative_guidance or png_payload is None or 'x0_neg' not in png_payload:
            return x0_pos.new_zeros(())

        valid_mask = labels > 0
        if not valid_mask.any():
            return x0_pos.new_zeros(())

        x0_neg = png_payload['x0_neg']
        target_norm = F.normalize(target_embeddings, dim=-1)
        pos_norm = F.normalize(x0_pos, dim=-1)
        neg_norm = F.normalize(x0_neg, dim=-1)

        pos_cos = (pos_norm * target_norm).sum(dim=-1)[valid_mask]
        neg_cos = (neg_norm * target_norm).sum(dim=-1)[valid_mask]
        gal_loss = F.relu(self.gal_margin - (pos_cos - neg_cos)).mean()

        self.latest_png_stats.update({
            'gal_loss': float(gal_loss.detach().item()),
            'pos_branch_target_cos': float(pos_cos.mean().detach().item()),
            'neg_branch_target_cos': float(neg_cos.mean().detach().item()),
            'pos_minus_neg_target_gap': float((pos_cos - neg_cos).mean().detach().item()),
            'confusing_negative_score_mean': float(png_payload.get('confusing_negative_score_mean', 0.0)),
            'x0_pos_item_top1_acc': float(self._compute_branch_top1_acc(x0_pos, labels)),
            'x0_neg_item_top1_acc': float(self._compute_branch_top1_acc(x0_neg, labels)),
        })
        return gal_loss

    def _sample_random_negatives(self, positive_labels, num_negatives):
        if num_negatives <= 0:
            return positive_labels.new_empty((positive_labels.size(0), 0))
        neg_ids = torch.randint(
            1,
            self.item_num + 1,
            (positive_labels.size(0), num_negatives),
            device=positive_labels.device,
            dtype=positive_labels.dtype,
        )
        if self.item_num <= 1:
            return neg_ids.fill_(0)
        conflict_mask = neg_ids.eq(positive_labels.unsqueeze(1))
        max_resample_rounds = 8
        for _ in range(max_resample_rounds):
            if not conflict_mask.any():
                break
            neg_ids[conflict_mask] = torch.randint(
                1,
                self.item_num + 1,
                (int(conflict_mask.sum().item()),),
                device=positive_labels.device,
                dtype=positive_labels.dtype,
            )
            conflict_mask = neg_ids.eq(positive_labels.unsqueeze(1))
        if conflict_mask.any():
            neg_ids[conflict_mask] = ((positive_labels.unsqueeze(1).expand_as(neg_ids)[conflict_mask]) % self.item_num) + 1
        return neg_ids

    def _sample_pref_ratio_negatives(self, positive_labels, num_negatives):
        num_negatives = max(int(num_negatives), 0)
        if num_negatives == 0:
            return positive_labels.new_empty((positive_labels.size(0), 0))

        negative_ids = positive_labels.new_empty((positive_labels.size(0), num_negatives))
        filled_counts = torch.zeros(positive_labels.size(0), device=positive_labels.device, dtype=torch.long)
        unique_labels = torch.unique(positive_labels.detach())
        use_inbatch = self.item_alignment_negative_source in {'inbatch', 'inbatch_then_random'}

        if use_inbatch and unique_labels.numel() > 1:
            for idx in range(positive_labels.size(0)):
                candidate_ids = unique_labels[unique_labels != positive_labels[idx]]
                if candidate_ids.numel() == 0:
                    continue
                take_num = min(num_negatives, candidate_ids.numel())
                perm = torch.randperm(candidate_ids.numel(), device=positive_labels.device)[:take_num]
                sampled_ids = candidate_ids[perm]
                negative_ids[idx, :take_num] = sampled_ids
                filled_counts[idx] = take_num

        if self.item_alignment_negative_source in {'random', 'inbatch_then_random'}:
            need_random = filled_counts < num_negatives
            if need_random.any():
                random_neg_ids = self._sample_random_negatives(positive_labels[need_random], num_negatives)
                for row_idx, sample_idx in enumerate(torch.nonzero(need_random, as_tuple=False).flatten()):
                    start_idx = int(filled_counts[sample_idx].item())
                    negative_ids[sample_idx, start_idx:] = random_neg_ids[row_idx, :(num_negatives - start_idx)]

        if self.item_alignment_negative_source == 'inbatch' and (filled_counts < num_negatives).any():
            need_fill = filled_counts < num_negatives
            fallback_neg_ids = self._sample_random_negatives(positive_labels[need_fill], num_negatives)
            for row_idx, sample_idx in enumerate(torch.nonzero(need_fill, as_tuple=False).flatten()):
                start_idx = int(filled_counts[sample_idx].item())
                negative_ids[sample_idx, start_idx:] = fallback_neg_ids[row_idx, :(num_negatives - start_idx)]

        return negative_ids

    def _build_teacher_topk_candidates(self, teacher_logits, target_labels, topk):
        topk = max(int(topk), 1)
        teacher_logits = teacher_logits.clone()
        if teacher_logits.size(-1) > 0:
            teacher_logits[:, 0] = float('-inf')
        k = min(topk, teacher_logits.size(-1) - 1 if teacher_logits.size(-1) > 1 else 1)
        topk_logits, topk_indices = torch.topk(teacher_logits, k=k, dim=-1)

        target_in_topk = topk_indices.eq(target_labels.unsqueeze(-1))
        missing_mask = ~target_in_topk.any(dim=-1)
        if missing_mask.any():
            topk_indices[missing_mask, -1] = target_labels[missing_mask]
            topk_logits[missing_mask, -1] = teacher_logits[missing_mask, target_labels[missing_mask]]

        candidate_teacher_logits = teacher_logits.gather(-1, topk_indices)
        sorted_teacher_logits, sorted_order = torch.sort(candidate_teacher_logits, dim=-1, descending=True)
        sorted_indices = topk_indices.gather(-1, sorted_order)
        target_rank = sorted_indices.eq(target_labels.unsqueeze(-1)).float().argmax(dim=-1) + 1
        teacher_probs = F.softmax(sorted_teacher_logits / self.item_alignment_kd_temperature, dim=-1)
        teacher_entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1)
        return sorted_indices, sorted_teacher_logits, teacher_entropy, target_rank

    def compute_item_consistency_loss(self, denoised_seq, labels, t):
        effective_weight = self.get_current_item_consistency_weight()
        if self.item_consistency_mode == 'none' or effective_weight <= 0:
            self.latest_item_consistency_stats = {
                'item_consistency_loss': 0.0,
                'lambda_item': float(self.lambda_item),
                'item_consistency_effective_weight': float(effective_weight),
                'item_consistency_curr_epoch': float(self.current_epoch),
                'item_alignment_mode': self.item_alignment_mode,
                'item_alignment_temperature': float(self.item_alignment_temperature),
                'item_alignment_topk': int(self.item_alignment_topk),
                'item_alignment_kd_temperature': float(self.item_alignment_kd_temperature),
                'item_alignment_teacher_source': self.item_alignment_teacher_source,
                'item_alignment_margin': float(self.item_alignment_margin),
                'item_alignment_num_negatives': int(self.item_alignment_num_negatives),
                'item_alignment_negative_source': self.item_alignment_negative_source,
                'pref_ratio_loss': 0.0,
                'kd_loss': 0.0,
                'teacher_topk_entropy': 0.0,
                'teacher_target_rank_in_topk': 0.0,
                'teacher_topk_logit_mean': 0.0,
                'student_topk_logit_mean': 0.0,
                'pos_score_mean': 0.0,
                'neg_score_mean': 0.0,
                'pos_minus_neg_gap_mean': 0.0,
                'x0_item_top1_acc': 0.0,
                'item_consistency_ramp_progress': float(
                    min(max((self.current_epoch - self.item_consistency_warmup_epochs + 1) / max(self.item_consistency_ramp_epochs, 1), 0.0), 1.0)
                ) if self.item_consistency_ramp_epochs > 0 and self.current_epoch >= self.item_consistency_warmup_epochs else float(self.current_epoch >= self.item_consistency_warmup_epochs and self.item_consistency_warmup_epochs == 0),
            }
            return denoised_seq.new_zeros(())

        valid_mask = labels > 0
        if not valid_mask.any():
            self.latest_item_consistency_stats = {
                'item_consistency_loss': 0.0,
                'lambda_item': float(self.lambda_item),
                'item_consistency_effective_weight': float(effective_weight),
                'item_consistency_curr_epoch': float(self.current_epoch),
                'item_alignment_mode': self.item_alignment_mode,
                'item_alignment_temperature': float(self.item_alignment_temperature),
                'item_alignment_topk': int(self.item_alignment_topk),
                'item_alignment_kd_temperature': float(self.item_alignment_kd_temperature),
                'item_alignment_teacher_source': self.item_alignment_teacher_source,
                'item_alignment_margin': float(self.item_alignment_margin),
                'item_alignment_num_negatives': int(self.item_alignment_num_negatives),
                'item_alignment_negative_source': self.item_alignment_negative_source,
                'pref_ratio_loss': 0.0,
                'kd_loss': 0.0,
                'teacher_topk_entropy': 0.0,
                'teacher_target_rank_in_topk': 0.0,
                'teacher_topk_logit_mean': 0.0,
                'student_topk_logit_mean': 0.0,
                'pos_score_mean': 0.0,
                'neg_score_mean': 0.0,
                'pos_minus_neg_gap_mean': 0.0,
                'x0_item_top1_acc': 0.0,
            }
            return denoised_seq.new_zeros(())

        if t.dim() == 1:
            if t.numel() == labels.numel():
                t = t.reshape_as(labels)
            else:
                t = t.unsqueeze(-1)
        if t.dim() == 2 and t.shape != labels.shape:
            if t.numel() == labels.numel():
                t = t.reshape_as(labels)
            elif t.size(1) == 1 and labels.size(1) != 1:
                t = t.expand(-1, labels.size(1))
        if t.shape != labels.shape:
            raise ValueError(f"unexpected timestep shape {tuple(t.shape)} for labels shape {tuple(labels.shape)}")
        t = t.long().clamp(min=0, max=self.diffu.num_timesteps - 1)

        flat_teacher_state = denoised_seq[valid_mask]
        x0_norm = F.normalize(denoised_seq, dim=-1)
        item_emb_norm = F.normalize(self.item_embedding.weight, dim=-1)
        flat_x0 = x0_norm[valid_mask]
        flat_labels = labels[valid_mask]
        flat_timesteps = t[valid_mask].float()

        if self.item_consistency_mode == 'snr':
            alpha_bar = torch.as_tensor(self.diffu.alphas_cumprod, device=denoised_seq.device, dtype=denoised_seq.dtype)
            item_weights = alpha_bar[t].pow(self.item_consistency_snr_power)
        else:
            item_weights = denoised_seq.new_ones(labels.shape, dtype=denoised_seq.dtype)
        flat_weights = item_weights[valid_mask]
        weighted_loss_sum = denoised_seq.new_zeros(())
        total_weight = flat_weights.sum().clamp_min(1e-6)
        pos_logits_chunks = []
        neg_logits_chunks = []
        gap_chunks = []
        teacher_entropy_chunks = []
        teacher_rank_chunks = []
        teacher_topk_logit_chunks = []
        student_topk_logit_chunks = []
        top1_acc_sum = denoised_seq.new_zeros(())
        total_examples = max(flat_labels.numel(), 1)

        chunk_size = max(int(self.item_consistency_chunk_size), 1)
        for start in range(0, flat_labels.size(0), chunk_size):
            end = min(start + chunk_size, flat_labels.size(0))
            chunk_x0 = flat_x0[start:end]
            chunk_teacher_state = flat_teacher_state[start:end]
            chunk_labels = flat_labels[start:end]
            chunk_weights = flat_weights[start:end]
            if self.item_alignment_mode == 'topk_kd':
                teacher_logits = torch.matmul(chunk_teacher_state.detach(), self.item_embedding.weight.detach().t())
                candidate_ids, teacher_topk_logits, teacher_entropy, teacher_target_rank = self._build_teacher_topk_candidates(
                    teacher_logits=teacher_logits,
                    target_labels=chunk_labels,
                    topk=self.item_alignment_topk,
                )
                candidate_emb = item_emb_norm[candidate_ids]
                student_topk_logits = (chunk_x0.unsqueeze(1) * candidate_emb).sum(dim=-1) / self.item_alignment_temperature
                teacher_probs = F.softmax(teacher_topk_logits / self.item_alignment_kd_temperature, dim=-1)
                student_log_probs = F.log_softmax(student_topk_logits / self.item_alignment_kd_temperature, dim=-1)
                chunk_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
                chunk_loss = chunk_loss * (self.item_alignment_kd_temperature ** 2)
                target_positions = candidate_ids.eq(chunk_labels.unsqueeze(-1)).float().argmax(dim=-1)
                chunk_pos_logits = student_topk_logits.gather(-1, target_positions.unsqueeze(-1)).squeeze(-1)
                chunk_neg_logits_mean = (student_topk_logits.sum(dim=-1) - chunk_pos_logits) / max(student_topk_logits.shape[-1] - 1, 1)
                top1_acc_sum = top1_acc_sum + (candidate_ids.gather(-1, student_topk_logits.argmax(dim=-1, keepdim=True)).squeeze(-1) == chunk_labels).float().sum()
                teacher_entropy_chunks.append(teacher_entropy.detach())
                teacher_rank_chunks.append(teacher_target_rank.detach().float())
                teacher_topk_logit_chunks.append(teacher_topk_logits.mean(dim=-1).detach())
                student_topk_logit_chunks.append(student_topk_logits.mean(dim=-1).detach())
            elif self.item_alignment_mode == 'pref_ratio':
                neg_ids = self._sample_pref_ratio_negatives(chunk_labels, self.item_alignment_num_negatives)
                pos_scores = (chunk_x0 * item_emb_norm[chunk_labels]).sum(dim=-1, keepdim=True)
                if neg_ids.numel() > 0:
                    neg_emb = item_emb_norm[neg_ids]
                    neg_scores = (chunk_x0.unsqueeze(1) * neg_emb).sum(dim=-1)
                    sampled_logits = torch.cat([pos_scores, neg_scores], dim=-1) / self.item_alignment_temperature
                else:
                    neg_scores = chunk_x0.new_zeros((chunk_x0.size(0), 0))
                    sampled_logits = pos_scores / self.item_alignment_temperature
                chunk_loss = -F.log_softmax(sampled_logits, dim=-1)[:, 0]
                chunk_pos_logits = sampled_logits[:, 0]
                chunk_neg_logits_mean = neg_scores.mean(dim=-1) / self.item_alignment_temperature if neg_scores.numel() > 0 else chunk_pos_logits.new_zeros(chunk_pos_logits.shape)
                top1_acc_sum = top1_acc_sum + (sampled_logits.argmax(dim=-1) == 0).float().sum()
            else:
                chunk_logits = torch.matmul(chunk_x0, item_emb_norm.t())
                chunk_logits = chunk_logits / self.item_alignment_temperature
                if self.item_alignment_mode == 'cosine_margin_ce':
                    margin_mask = F.one_hot(chunk_labels, num_classes=chunk_logits.size(-1)).to(chunk_logits.dtype)
                    chunk_logits = chunk_logits - margin_mask * self.item_alignment_margin
                chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, reduction='none')
                chunk_pos_logits = chunk_logits.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)
                chunk_neg_logits_mean = (chunk_logits.sum(dim=-1) - chunk_pos_logits) / max(chunk_logits.shape[-1] - 1, 1)
                top1_acc_sum = top1_acc_sum + (chunk_logits.argmax(dim=-1) == chunk_labels).float().sum()

            weighted_loss_sum = weighted_loss_sum + (chunk_loss * chunk_weights).sum()
            pos_logits_chunks.append(chunk_pos_logits.detach())
            neg_logits_chunks.append(chunk_neg_logits_mean.detach())
            gap_chunks.append((chunk_pos_logits - chunk_neg_logits_mean).detach())

        item_consistency_loss = weighted_loss_sum / total_weight
        pos_logits = torch.cat(pos_logits_chunks, dim=0) if pos_logits_chunks else denoised_seq.new_zeros(1)
        neg_logits_mean = torch.cat(neg_logits_chunks, dim=0) if neg_logits_chunks else denoised_seq.new_zeros(1)
        pos_minus_neg_gap = torch.cat(gap_chunks, dim=0) if gap_chunks else denoised_seq.new_zeros(1)
        teacher_topk_entropy = torch.cat(teacher_entropy_chunks, dim=0) if teacher_entropy_chunks else denoised_seq.new_zeros(1)
        teacher_target_rank = torch.cat(teacher_rank_chunks, dim=0) if teacher_rank_chunks else denoised_seq.new_zeros(1)
        teacher_topk_logit_mean = torch.cat(teacher_topk_logit_chunks, dim=0) if teacher_topk_logit_chunks else denoised_seq.new_zeros(1)
        student_topk_logit_mean = torch.cat(student_topk_logit_chunks, dim=0) if student_topk_logit_chunks else denoised_seq.new_zeros(1)
        top1_acc = top1_acc_sum / total_examples
        self.latest_item_consistency_stats = {
            'item_consistency_loss': float(item_consistency_loss.detach().item()),
            'lambda_item': float(self.lambda_item),
            'item_consistency_effective_weight': float(effective_weight),
            'item_consistency_curr_epoch': float(self.current_epoch),
            'item_consistency_warmup_epochs': float(self.item_consistency_warmup_epochs),
            'item_consistency_ramp_epochs': float(self.item_consistency_ramp_epochs),
            'item_consistency_chunk_size': int(chunk_size),
            'item_alignment_mode': self.item_alignment_mode,
            'item_alignment_temperature': float(self.item_alignment_temperature),
            'item_alignment_topk': int(self.item_alignment_topk),
            'item_alignment_kd_temperature': float(self.item_alignment_kd_temperature),
            'item_alignment_teacher_source': self.item_alignment_teacher_source,
            'item_alignment_margin': float(self.item_alignment_margin),
            'item_alignment_num_negatives': int(self.item_alignment_num_negatives),
            'item_alignment_negative_source': self.item_alignment_negative_source,
            'current_timestep_mean': float(flat_timesteps.mean().detach().item()),
            'current_timestep_std': float(flat_timesteps.std(unbiased=False).detach().item()),
            'item_consistency_weight_mean': float(flat_weights.mean().detach().item()),
            'item_consistency_weight_std': float(flat_weights.std(unbiased=False).detach().item()),
            'x0_item_pos_logit_mean': float(pos_logits.mean().detach().item()),
            'x0_item_neg_logit_mean': float(neg_logits_mean.mean().detach().item()),
            'pref_ratio_loss': float(item_consistency_loss.detach().item()) if self.item_alignment_mode == 'pref_ratio' else 0.0,
            'kd_loss': float(item_consistency_loss.detach().item()) if self.item_alignment_mode == 'topk_kd' else 0.0,
            'teacher_topk_entropy': float(teacher_topk_entropy.mean().detach().item()),
            'teacher_target_rank_in_topk': float(teacher_target_rank.mean().detach().item()),
            'teacher_topk_logit_mean': float(teacher_topk_logit_mean.mean().detach().item()),
            'student_topk_logit_mean': float(student_topk_logit_mean.mean().detach().item()),
            'pos_score_mean': float(pos_logits.mean().detach().item()),
            'neg_score_mean': float(neg_logits_mean.mean().detach().item()),
            'pos_minus_neg_gap_mean': float(pos_minus_neg_gap.mean().detach().item()),
            'x0_item_top1_acc': float(top1_acc.detach().item()),
        }
        return item_consistency_loss

    def get_latest_aux_stats(self):
        merged_stats = dict(self.latest_item_consistency_stats)
        merged_stats.update(self.latest_png_stats)
        if hasattr(self.diffu, 'get_latest_stats'):
            merged_stats.update(self.diffu.get_latest_stats())
        return merged_stats
    
    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embedding(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))

    def prepare_inputs(self, sequence, tag):
        item_embeddings = self.item_embedding(sequence)
        tag_embeddings = self.item_embedding(tag)
        if self.geodesic:
            tag_embeddings = F.normalize(tag_embeddings,p=2, dim=-1)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.hist_norm(item_embeddings)
        mask_seq = (sequence>0).float()
        mask_tag = (tag>0).float().view(tag.shape[0],-1)
        return item_embeddings, tag_embeddings, mask_seq, mask_tag

    def forward(self, sequence, tag, train_flag=True):
        item_embeddings, tag_embeddings, mask_seq, mask_tag = self.prepare_inputs(sequence, tag)
        self._reset_png_stats()

        # out_seq = item_embeddings
        # last_item = item_embeddings[:, -1, :]
        # dif_loss =torch.ones(1)
        if train_flag:
            # pass

            diffu_outputs = self.diffu(
                item_embeddings,
                tag_embeddings,
                mask_seq,
                mask_tag,
                item_embedding_weight=self.item_embedding.weight,
                labels=tag,
            )
            out_seq, dif_loss = diffu_outputs[:2]
            timesteps = diffu_outputs[2] if len(diffu_outputs) > 2 else None
            png_payload = diffu_outputs[3] if len(diffu_outputs) > 3 else None
            last_item = out_seq[:, -1, :]
            item_consistency_loss = self.compute_item_consistency_loss(out_seq, tag, timesteps) if timesteps is not None else out_seq.new_zeros(())
            gal_loss = self.compute_positive_negative_guidance_loss(out_seq, png_payload, tag_embeddings, tag)

            # item_rep_dis = self.regularization_rep(rep_item, mask_seq)
            # seq_rep_dis = self.regularization_seq_item_rep(last_item, rep_item, mask_seq)

        else:
            # noise_x_t = th.randn_like(tag_emb)
            # print("noise_x_t",noise_x_t.shape)
            out_seq = self.diffu.denoise_sample(
                item_embeddings,
                tag_embeddings,
                mask_seq,
                mask_tag,
                item_embedding_weight=self.item_embedding.weight,
                labels=tag,
            )
            # out_seq = self.diffu.subseq_guidence(item_embeddings, tag_embeddings, mask_seq, mask_tag)
            last_item = out_seq[:, -1, :]
            dif_loss = None
            item_consistency_loss = None
            gal_loss = None
            self.latest_item_consistency_stats = {}
        # item_rep = self.model_main(item_embeddings, last_item, mask_seq)
        # seq_rep = item_rep[:, -1, :]
        # scores = torch.matmul(seq_rep, self.item_embeddings.weight.t())
            # scores = None
        return out_seq, last_item, dif_loss, item_consistency_loss, gal_loss

    def denoise_sample_only(self, sequence, tag):
        item_embeddings, tag_embeddings, mask_seq, mask_tag = self.prepare_inputs(sequence, tag)
        self._reset_png_stats()
        out_seq = self.diffu.denoise_sample(
            item_embeddings,
            tag_embeddings,
            mask_seq,
            mask_tag,
            item_embedding_weight=self.item_embedding.weight,
            labels=tag,
        )
        return out_seq, out_seq[:, -1, :]


def create_model_diffu(args):
    if args.model == 'diffurec':
        return DiffuRec(args)
    elif args.model == 'adrec':
        return AdRec(args)
    elif args.model == 'dreamrec':
        return DreamRec(args)
    else:
        print('args.model is wrong')
        return None
