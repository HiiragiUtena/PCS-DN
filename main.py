"""Full training script"""
import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
import json
from model import AdapterSCCLClassifier
from utils import ErcTextDataset, get_num_classes, compute_metrics, set_seed, get_label_VAD, convert_label_to_VAD, compute_predicts
import math
import argparse
import yaml
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
import torch.cuda.amp.grad_scaler as grad_scaler
import torch.cuda.amp.autocast_mode as autocast_mode
# 自动选择GPU
import torch.distributed as dist
import GPUtil
import os
# 额外
import datetime
import random
import logging
from tqdm import tqdm
from config_args import parse_args

"""Parameters"""
args = parse_args() # 配置文件在：config_args.py中

def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_devices(strategy, index_device='0'):
    if strategy == 'all':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dist.init_process_group(backend="nccl")
        (f"Using GPU: {device}")
        print("Using all available GPUs")
    elif strategy == 'specific':
        selected_gpu = int(index_device)
        device = torch.device(f'cuda:{selected_gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using GPU: {selected_gpu}")
    elif strategy == 'auto':
        deviceID = GPUtil.getAvailable(order='memory', limit=1)[0]
        device = torch.device(f'cuda:{deviceID}' if torch.cuda.is_available() else 'cpu')
        print(f"Automatically selected GPU: {deviceID}")
    else:
        raise ValueError("Invalid strategy. Choose 'all', 'specific', or 'auto'.")   
    return device


def log_message(message, log_file):
    """Write a log message to a file and print it to the console."""
    print(message)  # Print to console
    with open(log_file, 'a') as f:  # Append to the log file
        f.write(message + '\n')


def train_SCCL_vad(DATASET, epoch, model, dict_opt, scheduler, loss_function, mode, data, batch_size, cuda, label_VAD, alpha, scaler):
    # The training function for AdapterSCCLClassifier.
    random.shuffle(data)
    #crossentropy_loss = nn.CrossEntropyLoss()
    if mode == 'train':
        model.train()
    else:
        model.eval()

    predicts = []
    ground_truth = []
    losses = []
    num_classes = get_num_classes(DATASET)
    label_VAD = torch.stack(label_VAD, dim=0)
    vads = []
    #label_VAD = torch.rand(num_classes, 3)
    progress_bar = tqdm(range(0, len(data), batch_size), desc=f"{mode.capitalize()} Epoch {epoch}")
    for i in progress_bar:
        if mode == 'train':
            # optimizer.zero_grad()
            dict_opt['other'].zero_grad()
            dict_opt['DVAE_encoder'].zero_grad()
            dict_opt['DVAE_decoder'].zero_grad()

        bs_data = data[i: min(i+batch_size, len(data))]
        input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True, padding_value=1)
        masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True, padding_value=0)
        input_current = pad_sequence([torch.LongTensor(item['input_ids_current']) for item in bs_data], batch_first=True, padding_value=1)
        masks_current = pad_sequence([torch.LongTensor(item['attention_mask_current']) for item in bs_data], batch_first=True, padding_value=0)
        input_context = pad_sequence([torch.LongTensor(item['input_ids_context']) for item in bs_data], batch_first=True, padding_value=1)
        masks_context = pad_sequence([torch.LongTensor(item['attention_mask_context']) for item in bs_data], batch_first=True, padding_value=0)
        o_labels = [item['label'] for item in bs_data]
        labels = torch.LongTensor(o_labels)
        label_mask = torch.zeros(num_classes, dtype=torch.float)
        label_mask[o_labels] = 1.
        one_hot_labels = nn.functional.one_hot(labels, num_classes=num_classes)
        one_hot_vad_labels = one_hot_labels.T.float()
        
        if cuda:
            input_data = input_data.to(args['device'])
            masks = masks.to(args['device'])
            input_current = input_current.to(args['device'])
            masks_current = masks_current.to(args['device'])
            input_context = input_context.to(args['device'])
            masks_context = masks_context.to(args['device'])
            labels = labels.to(args['device'])
            one_hot_vad_labels = one_hot_vad_labels.to(args['device'])
            label_mask = label_mask.to(args['device'])
            label_VAD = label_VAD.to(args['device'])
        
        with autocast_mode.autocast():
            outputs, loss_ib, logits,\
            mi_losses_sum, dvae_kl_loss, recon_loss,\
            con_loss_un, con_loss_red,\
            dict_mi, latent_params_dis,\
            h_attn_weights = model(input_data, masks, input_current, masks_current, input_context, masks_context, label_VAD, one_hot_vad_labels, label_mask)
            # emd_loss = torch.mean(loss_function(logits, labels))
            # vad_loss = loss_function(logits, vad_labels)
            ce_loss = loss_function(outputs, labels)
            loss = ce_loss + alpha * loss_ib
            loss = loss + args['gamma_mi'] * mi_losses_sum
            loss = loss + args['gamma_dvae_kl'] * dvae_kl_loss
            loss = loss + args['gamma_re'] * recon_loss
            loss = loss + args['gamma_hsic_un'] * con_loss_un + args['gamma_hsic_red'] * con_loss_red
        
        if mode == 'train':
            '''loss.backward()
            optimizer.step()
            scheduler.step()'''
            scaler.scale(loss).backward()
            scaler.step(dict_opt['other'])
            scheduler.step()
            scaler.update()
            dict_opt['DVAE_encoder'].step()
            dict_opt['DVAE_decoder'].step()
            '''vCLUB '''
            for (latent_pair_name, s_mi_loss) in dict_mi.items():
                s_mi_loss.train()
                latent_name_1, latent_name_2 = latent_pair_name.split('-')
                params1 = latent_params_dis[latent_name_1]
                params2 = latent_params_dis[latent_name_2]
                this_e_mi_loss = s_mi_loss.learning_loss(params1.z.detach(), params2.z.detach())
                s_mi_loss.optimizer_step(this_e_mi_loss)
                s_mi_loss.eval()

        if mode == 'test':
            if logits != None:
                vads.append(logits.detach().cpu())
            else:
                vads = None
        
        ground_truth += labels.cpu().numpy().tolist()
        predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        #predicts += predicter(logits, "cat").cpu().numpy().tolist()
        losses.append(loss.item())

    # Calculate confusion matrix
    cm = confusion_matrix(ground_truth, predicts)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average='weighted') * 100, 2)
    
    if DATASET == 'DailyDialog':
        micro_f1 = round(f1_score(ground_truth, predicts, average='micro', labels=list(range(1, 7))) * 100, 2)
    else:
        micro_f1 = round(f1_score(ground_truth, predicts, average='micro') * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average='macro') * 100, 2)
    
    if mode == 'train':
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'dev':
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'test':
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
        print(f1_score(ground_truth, predicts, average=None))
    
    return weighted_f1, macro_f1, micro_f1, avg_loss, avg_accuracy, cm


def main(CUDA: bool, LR: float, SEED: int, DATASET: str, BATCH_SIZE: int, model_checkpoint: str,
         speaker_mode: str, num_past_utterances: int, num_future_utterances: int,
         NUM_TRAIN_EPOCHS: int, WEIGHT_DECAY: float, WARMUP_RATIO: float, **kwargs):

    #ROOT_DIR = './multimodal-datasets/'
    ROOT_DIR = './data'
    NUM_CLASS = get_num_classes(DATASET)
    lr = float(LR)
    label_VAD = get_label_VAD(DATASET)
    model_checkpoint = 'roberta-large'

    # 创建日志文件夹
    log_dir = "./logs/" + DATASET + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{DATASET}_{timestamp}.log")

    log_message(f"Experiment log for dataset: {DATASET}", log_file)
    log_message(f"Learning rate: {LR}, Batch size: {BATCH_SIZE}, Seed: {SEED}", log_file)

    '''Load data'''
    ds_train = ErcTextDataset(DATASET=DATASET, SPLIT='train', speaker_mode=speaker_mode,
                              num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                              model_checkpoint=model_checkpoint,
                              ROOT_DIR=ROOT_DIR, SEED=SEED)

    ds_val = ErcTextDataset(DATASET=DATASET, SPLIT='val', speaker_mode=speaker_mode,
                            num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                            model_checkpoint=model_checkpoint,
                            ROOT_DIR=ROOT_DIR, SEED=SEED)

    ds_test = ErcTextDataset(DATASET=DATASET, SPLIT='test', speaker_mode=speaker_mode,
                             num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                             model_checkpoint=model_checkpoint,
                             ROOT_DIR=ROOT_DIR, SEED=SEED)
    tr_data = ds_train.inputs_
    dev_data = ds_val.inputs_
    test_data = ds_test.inputs_

    model = AdapterSCCLClassifier(kwargs, NUM_CLASS)
    
    '''为不同模块设置不同的优化器''' 
    # DVAE
    dvae_encoder_params_names = [] # DVAE 的Encoder部分的参数名列表
    for name, param in model.named_parameters():
        if 'DVAE_context' in name:
            if 'representation2params' in name:
                dvae_encoder_params_names.append(name)
    dvae_decoder_params_names = [] # DVAE的Decoder部分的参数名列表
    for name, param in model.named_parameters():
        if 'decoder' in name:
            dvae_decoder_params_names.append(name)
    # 获取DVAE的参数
    dvae_params_names = []
    dvae_params_names = dvae_encoder_params_names + dvae_decoder_params_names
    dvae_decoder_params = [param for name, param in model.named_parameters() if name in dvae_decoder_params_names]
    dvae_encoder_params = [param for name, param in model.named_parameters() if name in dvae_encoder_params_names]
    # 获取其他部分的参数
    other_params = [param for name, param in model.named_parameters() if name not in dvae_params_names]
    # 获取其他部分的模块名称
    other_params_names = set()
    for name, param in model.named_parameters():
        if name not in dvae_params_names:
            module_name = name  # 获取模块名
            other_params_names.add(module_name)
    '''为不同模块单独分配优化器'''
    dict_opt = dict()
    dict_opt['DVAE_encoder'] = AdamW(dvae_encoder_params, lr=1e-4)
    dict_opt['DVAE_decoder'] = Adam(dvae_decoder_params, lr=1e-4)
    dict_opt['other'] = AdamW(other_params, lr=lr, weight_decay=WEIGHT_DECAY)

    predicter = None

    '''Use linear scheduler.'''
    total_steps = float(10*len(ds_train.inputs_))/BATCH_SIZE
    scheduler = get_linear_schedule_with_warmup(dict_opt['other'], int(total_steps*WARMUP_RATIO), math.ceil(total_steps))
    loss_function = nn.CrossEntropyLoss()

    '''Due to the limitation of computational resources, we use mixed floating point precision.'''
    scaler = grad_scaler.GradScaler()


    if CUDA:
        model.to(device)

    best_f1 = 0.0
    best_epoch = -1
    best_model_state = None
    bset_model_cm = None
    
    for n in range(NUM_TRAIN_EPOCHS):
        train_SCCL_vad(DATASET, n, model, dict_opt, scheduler, loss_function, "train", tr_data, BATCH_SIZE, CUDA, label_VAD,
                  args['alpha'], scaler)
        val_w_f1, val_macro_f1, val_micro_f1, val_avg_loss, val_avg_acc, _ = train_SCCL_vad(DATASET, n, model, dict_opt, scheduler, loss_function, "dev", dev_data, 1, CUDA,
                  label_VAD, args['alpha'], scaler)
        test_w_f1, test_macro_f1, test_micro_f1, test_avg_loss, test_avg_acc, test_cm = train_SCCL_vad(DATASET, n, model, dict_opt, scheduler, loss_function, "test", test_data, 1, CUDA,
                  label_VAD, args['alpha'], scaler)
        
        log_message(f"Epoch {n}: Test Loss: {test_avg_loss}, Test weight F1: {test_w_f1}, Test micro F1: {test_micro_f1}", log_file)
        log_message("-------------------------------", log_file)
        if DATASET == 'DailyDialog':
            if test_micro_f1 > best_f1:
                best_f1 = test_micro_f1
                best_epoch = n
                bset_model_cm = test_cm
                best_model_state = model.state_dict()
        else:
            if test_w_f1 > best_f1:
                best_f1 = test_w_f1
                best_epoch = n
                bset_model_cm = test_cm
                best_model_state = model.state_dict()
    
    log_message(f"Best model with weight-F1: {best_f1} at Epoch {best_epoch}", log_file)
    log_message(f"Best model with Confusion Matrix:\n {bset_model_cm}", log_file)
    
    save_dir = "./best_model/"f"{DATASET}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, f"{DATASET}_best_model.pt")
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)


if __name__ == "__main__":
    
    seed = args.seed
    strategy = args.GPU_TRAIN_MODE
    index_device = args.GPU_INDEX
    today = datetime.datetime.now()
    device = set_devices(strategy, index_device=index_device)
    set_seed(seed)

    print(args)
    args = vars(args)

    args['n_gpu'] = torch.cuda.device_count()
    args['device'] = device

    logging.info(f"arguments given to {__file__}: {args}")
    main(**args)