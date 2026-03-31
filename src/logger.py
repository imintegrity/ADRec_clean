import os
import time
import argparse
import logging
import yaml

import argparse


def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')





def reset_log(log_path):
    import logging
    fileh = logging.FileHandler(log_path, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.DEBUG)

def make_logger(train_time):
    # 加载默认配置
    config = load_config('config.yaml')

    # 解析命令行参数
    args = cmdline_args()
    args = merge_config_with_args(config, args)

    # 计算日志文件夹路径
    log_dir =os.path.join(args.log_file,args.model,args.dataset)
    # log_dir = os.path.abspath(args.log_file + args.model + args.dataset)

    # 检查并创建文件夹
    if not os.path.exists(args.log_file):
        print(f"Creating base log directory: {args.log_file}")
        os.makedirs(args.log_file)
    if not os.path.exists(log_dir):
        print(f"Creating dataset-specific log directory: {log_dir}")
        os.makedirs(log_dir)

    # 打印路径调试信息
    # print(f"Log directory: {log_dir}")

    # 设置日志文件的完整路径
    log_file_path = os.path.join(log_dir, str(train_time) + str(args.description)+ '.log')
    print(f"Log file path: {log_file_path}")
    reset_log(log_file_path)
    # 设置日志配置
    # logging.basicConfig(level=logging.INFO,
    #                     filename=log_file_path,
    #                     datefmt='%Y/%m/%d %H:%M:%S',
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
    #                     filemode='w')

    logger = logging.getLogger(__name__)

    # 测试日志输出
    # logger.info("This is a test log message.")
    return logger, args

def cmdline_args():
    # 创建 argparse 解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name: toys, amazon_beauty, steam, ml-1m')
    parser.add_argument('--log_file', help='log dir path')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--max_len', type=int, help='The max length of sequence')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1'], help='Device: cpu, cuda:0, cuda:1')
    parser.add_argument('--num_gpu', type=int, help='Number of GPU')
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument("--hidden_size", type=int, help="hidden size of model")
    parser.add_argument('--dif_blocks', type=int, help='Number of denoiser blocks')
    parser.add_argument('--dropout', type=float, help='Dropout of representation')
    parser.add_argument('--emb_dropout', type=float, help='Dropout of item embedding')
    parser.add_argument("--hidden_act", type=str, help="Activation function: gelu or relu")
    # parser.add_argument('--num_blocks', type=int, help='Number of denoised decoder blocks')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--decay_step', type=int, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, help='Gamma for StepLR')
    parser.add_argument('--metric_ks', nargs='+', type=int, help='ks for Metric@k')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], help='Optimizer choice: SGD or Adam')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--loss_lambda', type=float, help='loss weight for diffusion')
    parser.add_argument('--weight_decay', type=float, help='L2 regularization')
    parser.add_argument('--momentum', type=float, help='SGD momentum')
    parser.add_argument('--schedule_sampler_name', type=str, help='Diffusion for t generation')
    parser.add_argument('--diffusion_steps', type=int, help='Diffusion step')
    parser.add_argument('--lambda_uncertainty', type=float, help='uncertainty weight')
    parser.add_argument('--lambda_schedule', type=str2bool, help='use lambda schedule')
    parser.add_argument('--lambda_beta_a', type=float, help='uncertainty weight')
    parser.add_argument('--lambda_beta_b', type=float, help='uncertainty weight')
    parser.add_argument('--noise_schedule', type=str, help='Noise schedule')
    parser.add_argument('--beta_a', type=float)
    parser.add_argument('--beta_b', type=float)
    parser.add_argument('--rescale_timesteps', help='rescale timesteps')
    parser.add_argument('--eval_interval', type=int, help='the number of epoch to eval')
    parser.add_argument('--patience', type=int, help='the number of epoch to wait before early stop')
    parser.add_argument('--long_head', type=str2bool, help='Long and short sequence, head and long-tail items')
    parser.add_argument('--diversity_measure', type=str2bool, help='Measure the diversity of recommendation results')
    parser.add_argument('--epoch_time_avg', type=str2bool, help='Calculate the average time of one epoch training')
    parser.add_argument('--dif_decoder', type=str, choices=['att', 'mlp', 'mamba', 'mamba_tcond', 'mamba_tcond_ssm', 'mamba_tcond_ffn', 'mamba_tcond_input', 'spc_mamba_nogate', 'spc_mamba', 'mamba_adaln_only', 'mamba_tcond_input_adaln', 'mamba_tcond_input_adaln_localattn_last', 'mamba_tcond_input_adaln_localattn_all', 'mamba_tcond_input_adaln_stage_route_m', 'mamba_tcond_input_adaln_stage_route_f', 'mamba_tcond_input_adaln_stage_route_both', 'mamba_tcond_input_adaln_ffn_stage_adapter', 'mamba_tcond_input_adaln_ffn_tsm_adapter', 'mamba_tcond_input_adaln_ffn_tsm_adapter_noglobal', 'mamba_tcond_input_adaln_item_consistency_all', 'mamba_tcond_input_adaln_item_consistency_snr', 'mamba_tcond_input_adaln_stationary_latent_all', 'mamba_tcond_input_adaln_stationary_latent_snr'], help='Choose denoised decoder')
    parser.add_argument('--mamba_d_state', type=int, help='Mamba state size')
    parser.add_argument('--mamba_d_conv', type=int, help='Mamba local conv width')
    parser.add_argument('--mamba_expand', type=int, help='Mamba expand ratio')
    parser.add_argument('--tcond_gate_alpha_max', type=float, help='Upper bound for timestep-conditioned residual gates')
    parser.add_argument('--local_attn_window', type=int, help='Window size for lightweight local attention')
    parser.add_argument('--local_attn_heads', type=int, help='Number of heads for lightweight local attention')
    parser.add_argument('--local_attn_dim', type=int, help='Projection dim for lightweight local attention; <=0 uses hidden size')
    parser.add_argument('--route_num_stages', type=int, help='Number of diffusion noise stages for deterministic routing')
    parser.add_argument('--route_shared_ratio', type=float, help='Always-on shared channel ratio for stage routing')
    parser.add_argument('--route_extra_ratio_low', type=float, help='Extra routed channel ratio for low-noise stage')
    parser.add_argument('--route_extra_ratio_high', type=float, help='Extra routed channel ratio for high-noise stage')
    parser.add_argument('--ffn_adapter_num_stages', type=int, help='Number of fine-grained FFN adapter stages')
    parser.add_argument('--ffn_adapter_ctx_num_stages', type=int, help='Number of coarse FFN context adapter stages')
    parser.add_argument('--ffn_adapter_bottleneck_ratio', type=float, help='Bottleneck ratio for FFN-side timestep adapters')
    parser.add_argument('--lambda_item', type=float, help='Weight for x0 item consistency supervision')
    parser.add_argument('--item_alignment_mode', type=str, choices=['ce', 'pref_ratio', 'cosine_margin_ce'], help='Item-space alignment loss mode for x0_hat')
    parser.add_argument('--item_alignment_temperature', type=float, help='Temperature for normalized x0_hat-to-item cosine logits')
    parser.add_argument('--item_alignment_margin', type=float, help='Additive cosine margin subtracted from the positive target logit')
    parser.add_argument('--item_alignment_num_negatives', type=int, help='Number of sampled negatives for preference-ratio item alignment')
    parser.add_argument('--item_alignment_negative_source', type=str, choices=['inbatch', 'random', 'inbatch_then_random'], help='Negative source for preference-ratio item alignment')
    parser.add_argument('--item_consistency_temperature', type=float, help='Temperature for x0-to-item logits')
    parser.add_argument('--item_consistency_snr_power', type=float, help='Power for SNR-aware item consistency weighting')
    parser.add_argument('--item_consistency_chunk_size', type=int, help='Chunk size for x0-to-item logits to reduce GPU memory')
    parser.add_argument('--item_consistency_warmup_epochs', type=int, help='Epochs to keep item consistency disabled before ramp-up')
    parser.add_argument('--item_consistency_ramp_epochs', type=int, help='Epochs to linearly ramp item consistency weight')
    parser.add_argument('--item_consistency_max_weight', type=float, help='Maximum curriculum weight for item consistency; defaults to lambda_item')
    parser.add_argument('--stationary_anchor_scale', type=float, help='History-anchor mixing ratio for stationary latent target')
    parser.add_argument('--stationary_anchor_max_scale', type=float, help='Maximum curriculum scale for stationary latent anchor mixing')
    parser.add_argument('--stationary_shift_norm_cap', type=float, help='Absolute norm cap for stationary anchor shift before mixing')
    parser.add_argument('--split_onebyone', type=str2bool, help='Split sequence one by one')
    parser.add_argument('--parallel_ag', type=str2bool, help='Train in a per token auto-aggressive manner')
    parser.add_argument('--is_causal', type=str2bool, help='Use causal attention')
    parser.add_argument('--dif_objective', type=str, choices=['pred_noise', 'pred_x0', 'pred_v'],
                        help='Choose diffusion loss objective')
    parser.add_argument('--pretrained', type=str2bool, help='use pretrained embedding weight')
    parser.add_argument('--freeze_emb', type=str2bool, help='freezing embedding weight')
    parser.add_argument('--disable_adrec_pretrained', type=str2bool, help='Disable ADRec pretrained embedding initialization')
    parser.add_argument('--embedding_warmup_epochs', type=int, help='Number of epochs to keep ADRec item embeddings frozen')
    parser.add_argument('--model', type=str)
    parser.add_argument('--loss', type=str, choices=['ce', 'mse'])
    parser.add_argument('--loss_scale', type=float)
    parser.add_argument('--cfg_scale', type=float)
    parser.add_argument('--description', type=str)
    parser.add_argument('--pcgrad', type=str2bool)
    parser.add_argument('--geodesic', type=str2bool)
    # 解析命令行参数
    args = parser.parse_args()

    return args
def load_config(config_file):
    # 模拟加载配置
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def merge_config_with_args(config, args):
    # 将 YAML 配置字典转为 Namespace 对象
    config_namespace = argparse.Namespace(**config)

    # 使用命令行参数覆盖配置字典中的值
    for key, value in vars(args).items():
        if value is not None:
            setattr(config_namespace, key, value)

    return config_namespace
