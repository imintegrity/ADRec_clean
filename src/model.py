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
        }.get(args.dif_decoder, 'none')
        self.lambda_item = getattr(args, 'lambda_item', 0.0)
        self.item_consistency_temperature = getattr(args, 'item_consistency_temperature', 0.07)
        self.item_consistency_snr_power = getattr(args, 'item_consistency_snr_power', 1.0)
        self.latest_item_consistency_stats = {}
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

    def compute_item_consistency_loss(self, denoised_seq, labels, t):
        if self.item_consistency_mode == 'none' or self.lambda_item <= 0:
            self.latest_item_consistency_stats = {
                'item_consistency_loss': 0.0,
                'lambda_item': float(self.lambda_item),
            }
            return denoised_seq.new_zeros(())

        valid_mask = labels > 0
        if not valid_mask.any():
            self.latest_item_consistency_stats = {
                'item_consistency_loss': 0.0,
                'lambda_item': float(self.lambda_item),
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

        x0_norm = F.normalize(denoised_seq, dim=-1)
        item_emb_norm = F.normalize(self.item_embedding.weight, dim=-1)
        item_logits = torch.matmul(x0_norm, item_emb_norm.t()) / self.item_consistency_temperature

        flat_logits = item_logits[valid_mask]
        flat_labels = labels[valid_mask]
        ce_per_token = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        flat_timesteps = t[valid_mask].float()

        if self.item_consistency_mode == 'snr':
            alpha_bar = torch.as_tensor(self.diffu.alphas_cumprod, device=denoised_seq.device, dtype=denoised_seq.dtype)
            item_weights = alpha_bar[t].pow(self.item_consistency_snr_power)
        else:
            item_weights = denoised_seq.new_ones(labels.shape, dtype=denoised_seq.dtype)
        flat_weights = item_weights[valid_mask]
        item_consistency_loss = (ce_per_token * flat_weights).sum() / flat_weights.sum().clamp_min(1e-6)

        pos_logits = flat_logits.gather(-1, flat_labels.unsqueeze(-1)).squeeze(-1)
        neg_logits_mean = (flat_logits.sum(dim=-1) - pos_logits) / max(flat_logits.shape[-1] - 1, 1)
        top1_acc = (flat_logits.argmax(dim=-1) == flat_labels).float().mean()
        self.latest_item_consistency_stats = {
            'item_consistency_loss': float(item_consistency_loss.detach().item()),
            'lambda_item': float(self.lambda_item),
            'current_timestep_mean': float(flat_timesteps.mean().detach().item()),
            'current_timestep_std': float(flat_timesteps.std(unbiased=False).detach().item()),
            'item_consistency_weight_mean': float(flat_weights.mean().detach().item()),
            'item_consistency_weight_std': float(flat_weights.std(unbiased=False).detach().item()),
            'x0_item_pos_logit_mean': float(pos_logits.mean().detach().item()),
            'x0_item_neg_logit_mean': float(neg_logits_mean.mean().detach().item()),
            'x0_item_top1_acc': float(top1_acc.detach().item()),
        }
        return item_consistency_loss

    def get_latest_aux_stats(self):
        return dict(self.latest_item_consistency_stats)
    
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

        # out_seq = item_embeddings
        # last_item = item_embeddings[:, -1, :]
        # dif_loss =torch.ones(1)
        if train_flag:
            # pass

            diffu_outputs = self.diffu(item_embeddings, tag_embeddings, mask_seq, mask_tag)
            out_seq, dif_loss = diffu_outputs[:2]
            timesteps = diffu_outputs[2] if len(diffu_outputs) > 2 else None
            last_item = out_seq[:, -1, :]
            item_consistency_loss = self.compute_item_consistency_loss(out_seq, tag, timesteps) if timesteps is not None else out_seq.new_zeros(())

            # item_rep_dis = self.regularization_rep(rep_item, mask_seq)
            # seq_rep_dis = self.regularization_seq_item_rep(last_item, rep_item, mask_seq)

        else:
            # noise_x_t = th.randn_like(tag_emb)
            # print("noise_x_t",noise_x_t.shape)
            out_seq = self.diffu.denoise_sample(item_embeddings, tag_embeddings, mask_seq, mask_tag)
            # out_seq = self.diffu.subseq_guidence(item_embeddings, tag_embeddings, mask_seq, mask_tag)
            last_item = out_seq[:, -1, :]
            dif_loss = None
            item_consistency_loss = None
            self.latest_item_consistency_stats = {}
        # item_rep = self.model_main(item_embeddings, last_item, mask_seq)
        # seq_rep = item_rep[:, -1, :]
        # scores = torch.matmul(seq_rep, self.item_embeddings.weight.t())
            # scores = None
        return out_seq, last_item, dif_loss, item_consistency_loss

    def denoise_sample_only(self, sequence, tag):
        item_embeddings, tag_embeddings, mask_seq, mask_tag = self.prepare_inputs(sequence, tag)
        out_seq = self.diffu.denoise_sample(item_embeddings, tag_embeddings, mask_seq, mask_tag)
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
