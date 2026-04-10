
import os
import json
import time
from metrics import *
from utils import *
from model import Att_Diffuse_model
from pcgrad import PCGrad
from torch import optim
from sasrec import SASRec


# from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup
def extract(data):
    seq= data[0]
    diff_loss = data[1] if len(data) == 2 else torch.zeros(1,device=seq.device)
    return seq, seq[:,-1], diff_loss

def item_num_create(args):
    length = {"ml-100k":1008,
              'yelp': 64669,
              'sports':12301,
              'baby':4731,
              'toys':7309,
              'beauty':6086
              }
    args.item_num = length[args.dataset]
    return args
def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        if args.model == 'adrec':
            opt= optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        else:
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        opt= optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError
    return opt

def choose_model(args):
    device = args.device
    if args.model in ['diffurec','adrec','dreamrec']:
        if args.model == 'adrec':
            if args.disable_adrec_pretrained:
                args.pretrained = False
                args.freeze_emb = False
                args.embedding_warmup_epochs = 0
            else:
                args.pretrained = True
                args.freeze_emb = True
        if args.model == 'diffurec':
            args.split_onebyone=True
            args.parallel_ag = False
            args.is_causal = False
        model = Att_Diffuse_model(args)
    elif args.model == 'sasrec' or args.model == 'pretrain':
        model = SASRec(args)
    else:
        model=None
    return model.to(device)
# ("bert4rec" "core" "eulerformer" "fearec" "gru4rec" "trimlp")
def load_data(args):

    path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    return tra_data_loader, val_data_loader, test_data_loader


def is_cuda_device(device):
    return isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available()


def sync_cuda_if_needed(device):
    if is_cuda_device(device):
        torch.cuda.synchronize(device)


def count_parameters(module):
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def build_efficiency_report(model_joint, args):
    diffu_net = getattr(getattr(model_joint, 'diffu', None), 'net', None)
    decoder = getattr(diffu_net, 'decoder', None)
    total_params, trainable_params = count_parameters(model_joint)
    if diffu_net is None:
        diffusion_net_params, diffusion_net_trainable_params = 0, 0
    else:
        diffusion_net_params, diffusion_net_trainable_params = count_parameters(diffu_net)
    if decoder is None:
        decoder_params, decoder_trainable_params = 0, 0
    else:
        decoder_params, decoder_trainable_params = count_parameters(decoder)
    return {
        'dif_decoder': args.dif_decoder,
        'tcond_gate_alpha_max': getattr(args, 'tcond_gate_alpha_max', None),
        'local_attn_window': getattr(args, 'local_attn_window', None),
        'local_attn_heads': getattr(args, 'local_attn_heads', None),
        'local_attn_dim': getattr(args, 'local_attn_dim', None),
        'route_num_stages': getattr(args, 'route_num_stages', None),
        'route_shared_ratio': getattr(args, 'route_shared_ratio', None),
        'route_extra_ratio_low': getattr(args, 'route_extra_ratio_low', None),
        'route_extra_ratio_high': getattr(args, 'route_extra_ratio_high', None),
        'ffn_adapter_num_stages': getattr(args, 'ffn_adapter_num_stages', None),
        'ffn_adapter_ctx_num_stages': getattr(args, 'ffn_adapter_ctx_num_stages', None),
        'ffn_adapter_bottleneck_ratio': getattr(args, 'ffn_adapter_bottleneck_ratio', None),
        'lambda_item': getattr(args, 'lambda_item', None),
        'item_alignment_mode': getattr(args, 'item_alignment_mode', None),
        'item_alignment_temperature': getattr(args, 'item_alignment_temperature', None),
        'item_alignment_topk': getattr(args, 'item_alignment_topk', None),
        'item_alignment_kd_temperature': getattr(args, 'item_alignment_kd_temperature', None),
        'item_alignment_teacher_source': getattr(args, 'item_alignment_teacher_source', None),
        'item_alignment_margin': getattr(args, 'item_alignment_margin', None),
        'item_alignment_num_negatives': getattr(args, 'item_alignment_num_negatives', None),
        'item_alignment_negative_source': getattr(args, 'item_alignment_negative_source', None),
        'item_consistency_temperature': getattr(args, 'item_consistency_temperature', None),
        'item_consistency_snr_power': getattr(args, 'item_consistency_snr_power', None),
        'item_consistency_chunk_size': getattr(args, 'item_consistency_chunk_size', None),
        'item_consistency_warmup_epochs': getattr(args, 'item_consistency_warmup_epochs', None),
        'item_consistency_ramp_epochs': getattr(args, 'item_consistency_ramp_epochs', None),
        'item_consistency_max_weight': getattr(args, 'item_consistency_max_weight', None),
        'stationary_anchor_scale': getattr(args, 'stationary_anchor_scale', None),
        'stationary_anchor_max_scale': getattr(args, 'stationary_anchor_max_scale', None),
        'stationary_shift_norm_cap': getattr(args, 'stationary_shift_norm_cap', None),
        'prediction_target_mode': getattr(args, 'prediction_target_mode', None),
        'pref_teacher_topk': getattr(args, 'pref_teacher_topk', None),
        'pref_teacher_temperature': getattr(args, 'pref_teacher_temperature', None),
        'pref_mix_topk_alpha': getattr(args, 'pref_mix_topk_alpha', None),
        'pref_mix_stationary_beta': getattr(args, 'pref_mix_stationary_beta', None),
        'generative_process_mode': getattr(args, 'generative_process_mode', None),
        'flow_source_mode': getattr(args, 'flow_source_mode', None),
        'flow_source_hist_blend_rho': getattr(args, 'flow_source_hist_blend_rho', None),
        'flow_num_steps': getattr(args, 'flow_num_steps', None),
        'flow_time_schedule': getattr(args, 'flow_time_schedule', None),
        'flow_loss_weight': getattr(args, 'flow_loss_weight', None),
        'trajectory_consistency_mode': getattr(args, 'trajectory_consistency_mode', None),
        'td_delta_step': getattr(args, 'td_delta_step', None),
        'td_loss_weight': getattr(args, 'td_loss_weight', None),
        'td_weighting_mode': getattr(args, 'td_weighting_mode', None),
        'self_condition_mode': getattr(args, 'self_condition_mode', None),
        'self_condition_train_prob': getattr(args, 'self_condition_train_prob', None),
        'self_condition_dropout_prob': getattr(args, 'self_condition_dropout_prob', None),
        'self_condition_noise_std': getattr(args, 'self_condition_noise_std', None),
        'self_condition_fusion_mode': getattr(args, 'self_condition_fusion_mode', None),
        'phase_target_mode': getattr(args, 'phase_target_mode', None),
        'phase_t_split': getattr(args, 'phase_t_split', None),
        'phase_high_beta': getattr(args, 'phase_high_beta', None),
        'use_positive_negative_guidance': getattr(args, 'use_positive_negative_guidance', None),
        'negative_condition_source': getattr(args, 'negative_condition_source', None),
        'negative_condition_topk': getattr(args, 'negative_condition_topk', None),
        'png_guidance_scale': getattr(args, 'png_guidance_scale', None),
        'gal_margin': getattr(args, 'gal_margin', None),
        'gal_weight': getattr(args, 'gal_weight', None),
        'decoder_mode': getattr(decoder, 'decoder_mode', None),
        'decoder_active_components': getattr(decoder, 'active_components', None),
        'tcond_placement': getattr(getattr(diffu_net, 'decoder', None), 'placement', None),
        'tcond_active_branches': getattr(getattr(diffu_net, 'decoder', None), 'active_branches', None),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'diffusion_net_params': diffusion_net_params,
        'diffusion_net_trainable_params': diffusion_net_trainable_params,
        'decoder_params': decoder_params,
        'decoder_trainable_params': decoder_trainable_params,
        'epoch_efficiency': [],
    }


def get_tcond_stats(model_joint):
    merged_stats = {}
    decoder = getattr(getattr(getattr(model_joint, 'diffu', None), 'net', None), 'decoder', None)
    if decoder is not None and hasattr(decoder, 'get_latest_stats'):
        merged_stats.update(decoder.get_latest_stats())
    if hasattr(model_joint, 'get_latest_aux_stats'):
        merged_stats.update(model_joint.get_latest_aux_stats())
    return merged_stats


def is_numeric_stat(value):
    return isinstance(value, (int, float))


def loss_requires_grad(loss_tensor):
    return isinstance(loss_tensor, torch.Tensor) and loss_tensor.requires_grad


def save_efficiency_report(report, args, train_time):
    saved_dir = os.path.join('saved', args.model, args.dataset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    report_path = os.path.join(saved_dir, str(train_time) + args.description + '_efficiency.json')
    with open(report_path, 'w', encoding='utf-8') as file_obj:
        json.dump(report, file_obj, indent=2)
    return report_path

def model_train(model_joint,tra_data_loader, val_data_loader, test_data_loader, args, logger,train_time):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    # model_joint = torch.compile(model_joint, )
    torch.set_float32_matmul_precision('high')
    optimizer = PCGrad(optimizers(model_joint, args),args)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.optim, T_max=500)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    best_model = None
    efficiency_report = build_efficiency_report(model_joint, args)
    logger.info(f"Model efficiency summary: {efficiency_report}")
    train_wall_clock_start = time.perf_counter()
    for epoch_temp in range(epochs):
        model_joint.train()
        if hasattr(model_joint, 'set_curriculum_epoch'):
            model_joint.set_curriculum_epoch(epoch_temp)
        if (
            args.model == 'adrec'
            and args.embedding_warmup_epochs > 0
            and epoch_temp == args.embedding_warmup_epochs
        ):
            print(f'warm up finishied in epoch {epoch_temp}')
            logger.info(f'warm up finishied in epoch {epoch_temp}')
            model_joint.item_embedding.weight.requires_grad = True
        ce_losses = []
        dif_losses = []
        item_losses = []
        gal_losses = []
        epoch_tcond_stats = {}
        epoch_tcond_meta = {}
        flag_update = 0
        epoch_samples = 0
        if is_cuda_device(device):
            torch.cuda.reset_peak_memory_stats(device)
        sync_cuda_if_needed(device)
        epoch_start_time = time.perf_counter()
        pbr_train = tqdm(enumerate(tra_data_loader),desc='Epoch: {}'.format(epoch_temp),leave=False, total=len(tra_data_loader))
        # print('len',len(tra_data_loader))
        for index_temp, train_batch in pbr_train:
            train_batch = [x.to(device) for x in train_batch]
            epoch_samples += train_batch[0].shape[0]
            optimizer.zero_grad()
            outputs = model_joint(train_batch[0], train_batch[1], train_flag=True)
            out_seq, last_item = outputs[:2]
            dif_loss = outputs[2] if len(outputs) > 2 and outputs[2] is not None else torch.zeros(1, device=args.device)
            item_consistency_loss = outputs[3] if len(outputs) > 3 and outputs[3] is not None else torch.zeros(1, device=args.device)
            gal_loss = outputs[4] if len(outputs) > 4 and outputs[4] is not None else torch.zeros(1, device=args.device)
            ce_loss = model_joint.calculate_loss(out_seq, train_batch[1])  ## use this not above
            if args.model=='adrec' and args.loss=='mse':
                current_item_weight = getattr(model_joint, 'get_current_item_consistency_weight', lambda: getattr(args, 'lambda_item', 0.0))()
                current_gal_weight = getattr(model_joint, 'get_gal_weight', lambda: getattr(args, 'gal_weight', 0.0))()
                candidate_losses = [ce_loss, args.loss_scale * dif_loss]
                if current_item_weight > 0:
                    candidate_losses.append(current_item_weight * item_consistency_loss)
                if current_gal_weight > 0:
                    candidate_losses.append(current_gal_weight * gal_loss)
                losses = [loss for loss in candidate_losses if loss_requires_grad(loss)]
                if not losses:
                    losses = [ce_loss]
            elif args.model=='dreamrec':
                losses = [loss for loss in [dif_loss] if loss_requires_grad(loss)]
                if not losses:
                    losses = [ce_loss]
            else:
                losses=[ce_loss]
            optimizer.pc_backward(losses)
            ce_losses.append(ce_loss.item())
            dif_losses.append(dif_loss.item())
            item_losses.append(item_consistency_loss.item())
            gal_losses.append(gal_loss.item())
            current_tcond_stats = get_tcond_stats(model_joint)
            for key, value in current_tcond_stats.items():
                if is_numeric_stat(value):
                    epoch_tcond_stats.setdefault(key, []).append(value)
                elif isinstance(value, str):
                    epoch_tcond_meta[key] = value
            optimizer.step()
            pbr_train.set_postfix_str(f'loss={ce_losses[-1]:.3f}')
            # if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
            #     print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all[-1]))
            #     logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all[-1]))
        sync_cuda_if_needed(device)
        epoch_time = time.perf_counter() - epoch_start_time
        avg_step_time = epoch_time / max(len(tra_data_loader), 1)
        samples_per_sec = epoch_samples / max(epoch_time, 1e-12)
        peak_memory_mb = 0.0
        if is_cuda_device(device):
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        avg_item_loss = sum(item_losses) / len(item_losses) if item_losses else 0.0
        avg_gal_loss = sum(gal_losses) / len(gal_losses) if gal_losses else 0.0
        print(f"loss in epoch {epoch_temp}: ce_loss {sum(ce_losses)/len(ce_losses):.3f}, dif_loss {sum(dif_losses)/len(dif_losses):.3f}, item_loss {avg_item_loss:.3f}, gal_loss {avg_gal_loss:.3f}")
        logger.info(f"loss in epoch {epoch_temp}: ce_loss {sum(ce_losses)/len(ce_losses):.3f}, dif_loss {sum(dif_losses)/len(dif_losses):.3f}, item_loss {avg_item_loss:.3f}, gal_loss {avg_gal_loss:.3f}")
        logger.info(
            "epoch %d efficiency: epoch_time_sec=%.4f avg_step_time_sec=%.6f samples_per_sec=%.4f peak_gpu_mem_mb=%.2f",
            epoch_temp,
            epoch_time,
            avg_step_time,
            samples_per_sec,
            peak_memory_mb,
        )
        epoch_efficiency_entry = {
            'epoch': epoch_temp,
            'epoch_time_sec': epoch_time,
            'avg_step_time_sec': avg_step_time,
            'samples_per_sec': samples_per_sec,
            'peak_gpu_mem_mb': peak_memory_mb,
            'avg_gal_loss': avg_gal_loss,
        }
        if epoch_tcond_stats:
            epoch_tcond_summary = {
                key: float(np.mean(values)) for key, values in epoch_tcond_stats.items()
            }
            epoch_tcond_summary.update(epoch_tcond_meta)
            logger.info(f"epoch {epoch_temp} tcond gate stats: {epoch_tcond_summary}")
            epoch_efficiency_entry.update(epoch_tcond_summary)
        efficiency_report['epoch_efficiency'].append(epoch_efficiency_entry)
        lr_scheduler.step()
        # if epoch_temp == 10:
        #     args.eval_interval=3



        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            # logger.info(f"loss in epoch {epoch_temp}: ce_loss: {sum(ce_losses) / len(ce_losses):.3f}, dif_loss: {sum(dif_losses) / len(dif_losses):.3f}")
            # print(f"loss in epoch {epoch_temp}: ce_loss: {sum(ce_losses) / len(ce_losses):.3f}, dif_loss: {sum(dif_losses) / len(dif_losses):.3f}")
            # print('start predicting: ', datetime.datetime.now())
            # logger.info('start predicting: {}'.format(datetime.datetime.now()))
            metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
            val_inference_latencies = []
            model_joint.eval()
            with torch.no_grad():
                for val_batch in tqdm(val_data_loader,leave=False,desc='Denoising..., Epoch: {}'.format(epoch_temp)):
                    val_batch = [x.to(device) for x in val_batch]
                    sync_cuda_if_needed(device)
                    val_batch_start = time.perf_counter()
                    out_seq, last_item, *_= model_joint(val_batch[0], val_batch[1], train_flag=False)
                    sync_cuda_if_needed(device)
                    val_inference_latencies.append(time.perf_counter() - val_batch_start)
                    scores_rec_diffu = model_joint.calculate_score(last_item)    ### inner_production
                    # scores_rec_diffu = model_joint.routing_rep_pre(rep_diffu)   ### routing_rep_pre
                    # print(scores_rec_diffu.shape,val_batch[1][:,-1].shape)
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1][:,-1:], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
            if val_inference_latencies:
                efficiency_report['val_inference_avg_batch_latency_sec'] = float(np.mean(val_inference_latencies))
                efficiency_report['val_inference_p95_batch_latency_sec'] = float(np.percentile(val_inference_latencies, 95))

            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    best_epoch_temp = epoch_temp

            if flag_update == 0:
                bad_count += 1
                print('patience to end: ', args.patience - bad_count)
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model_joint)
            if bad_count >= args.patience:
                break
    # if args.model == 'adrec' and args.lambda_schedule:
    #     model_joint.diffu.net.lambda_uncertainty = schedule[best_epoch_temp]
    saved_dir = os.path.join('saved',args.model, args.dataset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if args.model == 'pretrain':
        output_path = os.path.join(saved_dir,'pretrain.pth')
    else:
        output_path = os.path.join(saved_dir, str(train_time) + args.description + '.pth')
    # torch.save(best_model._orig_mod.state_dict(), str(output_path))
    torch.save(best_model.state_dict(), str(output_path))
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    # if args.eval_interval > epochs:
    #     best_model = copy.deepcopy(model_joint)

    print('start testing: ', datetime.datetime.now())
    logger.info('start testing: {}'.format(datetime.datetime.now()))
    top_100_item = []
    model_joint.eval()
    inference_latencies = []
    scoring_latencies = []
    denoise_latencies = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in tqdm(test_data_loader,leave=False):
            test_batch = [x.to(device) for x in test_batch]
            sync_cuda_if_needed(device)
            full_batch_start_time = time.perf_counter()
            sync_cuda_if_needed(device)
            denoise_start_time = time.perf_counter()
            out_seq, last_item = best_model.denoise_sample_only(test_batch[0], test_batch[1])
            sync_cuda_if_needed(device)
            denoise_latencies.append(time.perf_counter() - denoise_start_time)
            sync_cuda_if_needed(device)
            score_start_time = time.perf_counter()
            scores_rec_diffu = best_model.calculate_score(last_item)   ### Inner Production
            sync_cuda_if_needed(device)
            scoring_latencies.append(time.perf_counter() - score_start_time)
            inference_latencies.append(time.perf_counter() - full_batch_start_time)
            # scores_rec_diffu = best_model.routing_rep_pre(rep_diffu)   ### routing

            _, indices = torch.topk(scores_rec_diffu, k=20)
            top_100_item.append(indices)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1][:,-1:], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    best_test_gap = float(best_metrics_dict.get('Best_NDCG@20', 0.0) - test_metrics_dict_mean.get('NDCG@20', 0.0))
    print(f'best/test gap: {best_test_gap:.4f}')
    logger.info(f'best/test gap: {best_test_gap:.4f}')
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    print(args)

    if inference_latencies:
        efficiency_report['test_inference_avg_batch_latency_sec'] = float(np.mean(inference_latencies))
        efficiency_report['test_inference_p95_batch_latency_sec'] = float(np.percentile(inference_latencies, 95))
    if denoise_latencies:
        efficiency_report['test_denoise_avg_batch_latency_sec'] = float(np.mean(denoise_latencies))
        efficiency_report['test_denoise_p95_batch_latency_sec'] = float(np.percentile(denoise_latencies, 95))
    if scoring_latencies:
        efficiency_report['test_scoring_avg_batch_latency_sec'] = float(np.mean(scoring_latencies))
        efficiency_report['test_scoring_p95_batch_latency_sec'] = float(np.percentile(scoring_latencies, 95))
    efficiency_report['best_test_gap'] = best_test_gap
    efficiency_report['train_wall_clock_sec'] = float(time.perf_counter() - train_wall_clock_start)
    if efficiency_report['epoch_efficiency']:
        efficiency_report['avg_epoch_time_sec'] = float(np.mean([entry['epoch_time_sec'] for entry in efficiency_report['epoch_efficiency']]))
    report_path = save_efficiency_report(efficiency_report, args, train_time)
    logger.info(f"Efficiency report saved to {report_path}")




    return best_model, test_metrics_dict_mean

