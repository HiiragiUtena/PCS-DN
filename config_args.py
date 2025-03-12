import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='erc RoBERTa text huggingface training')
    # 硬件参数
    parser.add_argument( "--GPU_TRAIN_MODE", help = "GPU使用策略，'all'：全部GPU，'specific'：手工选择GPU编号，'auto':自动选择最空闲GPU", 
                        default = 'auto') # # TRAIN_MODE = 'all' ,'specific', 'auto'
    parser.add_argument( "--GPU_INDEX", help = "'specific'：手工选择GPU编号，模式下才有用。0 or 1", 
                        default = '1') # # TRAIN_MODE = 'all' ,'specific', 'auto'
    # parser.add_argument('--UseLLM', type=bool, default=False, help='True or False')
    # 训练参数
    parser.add_argument('--DATASET', type=str, default="IEMOCAP")
    # IEMOCAP   MELD    EmoryNLP    DailyDialog 
    parser.add_argument('--CUDA', type=bool, default=True)
    parser.add_argument('--model_checkpoint', type=str, default="roberta-large")
    parser.add_argument('--speaker_mode', type=str, default="upper")
    parser.add_argument('--num_past_utterances', type=int, default=1000)
    parser.add_argument('--num_future_utterances', type=int, default=1000)
    parser.add_argument('--BATCH_SIZE', type=int, default=4)
    parser.add_argument('--LR', type=float, default=1e-5)
    parser.add_argument('--HP_ONLY_UPTO', type=int, default=10)
    parser.add_argument('--NUM_TRAIN_EPOCHS', type=int, default=10)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.01)
    parser.add_argument('--WARMUP_RATIO', type=float, default=0.2)
    parser.add_argument('--HP_N_TRIALS', type=int, default=5)
    parser.add_argument('--OUTPUT-DIR', type=str, default="./output")
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of original model.")

    parser.add_argument('--dim_z_disentangled', type=int, default=256, metavar='dim_z_disentangled', help='dim_z_disentangled')
    parser.add_argument('--gamma1', type=float, default=1.0, help='weight of pred loss')
    parser.add_argument('--gamma2', type=float, default=1.0, help='weight of gamma2')
    parser.add_argument('--gamma3', type=float, default=1.0, help='weight of gamma3')
    parser.add_argument('--gamma_mi', type=float, default=0.05, help='weight of MI loss')
    parser.add_argument('--gamma_dvae_kl', type=float, default=0.001, help='weight of DVAE KL loss')
    parser.add_argument('--gamma_re', type=float, default=0.5, help='weight of Recon loss')
    parser.add_argument('--gamma_pib_kl', type=float, default=1, help='weight of PIB KL loss')
    parser.add_argument('--gamma_hsic_red', type=float, default=1.0, help='weight of hsic_syn loss')
    parser.add_argument('--gamma_hsic_un', type=float, default=0.1, help='weight of hsic_un loss')
    parser.add_argument('--alpha', default=0.0,
                        type=float, help='The loss coefficient.')
    return parser.parse_args()