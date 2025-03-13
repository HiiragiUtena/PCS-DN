import torch
from torch import nnf
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from pytorch_transformers.my_modeling_roberta import RobertaModelwithAdapter
from loss import EMDLoss, SupConLoss

from sub_models.information_encoders import DisentangledVAE_TE

from config_args import parse_args




class PSCDN(nn.Module):
    """
    Supervised cluster-level contrastive learning which computes cluster-level VAD for each emotion and
    contrast with the emotion prototypes.
    """
    def __init__(self, args, num_class):
        super(PSCDN, self).__init__()
        # 加载本地模型
        self.bert = RobertaModel.from_pretrained("roberta-large")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.config = RobertaConfig.from_pretrained("roberta-large")
        self.device = args['device']
        # self.dim_z_disentangled = int(hidden_size//4)
        self.dim_z_disentangled = args['dim_z_disentangled'] 
        self.num_classes_true = num_class


        # 设置 tokenizer 的 max_length
        self.tokenizer.model_max_length = 512  # 或其他你希望的最大长度
        hidden_size = self.config.hidden_size
        self.dense_x = nn.Linear(hidden_size, hidden_size)
        self.dense_current = nn.Linear(hidden_size, hidden_size)
        self.dense_context = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dense_class = nn.Linear(self.dim_z_disentangled, self.n_classes)
        self.dense_x_vad = nn.Linear(hidden_size, 3)
        self.dense_current_vad = nn.Linear(hidden_size, 3)
        self.dense_context_vad = nn.Linear(hidden_size, 3)

        # 语义解耦网络
        dis_decoder_type = 'Normal'
        input_type_for_dis = 'fusion'
        self.DVAE_context = DisentangledVAE_TE(input_dim=hidden_size,
                                latent_dim=self.dim_z_disentangled, 
                                hidden_size=hidden_size, 
                                device=self.device,
                                decoder_type=dis_decoder_type,
                                input_type=input_type_for_dis)
        self.gate_syn = GatedUnit(hidden_size=self.dim_z_disentangled)
        self.gate_red = GatedUnit(hidden_size=self.dim_z_disentangled)
        self.gate_un1 = GatedUnit(hidden_size=self.dim_z_disentangled)
        self.gate_un2 = GatedUnit(hidden_size=self.dim_z_disentangled)
        self.matt = TransformerEncoderLayerWithAttn(
            d_model=self.dim_z_disentangled,
            nhead=4,
            dim_feedforward=self.dim_z_disentangled,
            dropout=0.1,
            batch_first=True
        )

    def SCCL_loss(self, x, label_VAD, one_hot_vad_labels, label_mask, temperature=1):
        """
        :param x: The predicted VAD scores. Dim:[B, 3]
        :param label_VAD: The VAD prototypes of each emotion. Dim:[label_num, 3]
        :param one_hot_vad_labels: One-hot matrix showing which instances has the emotion. Dim: [label_num, B]
        :param label_mask: One-hot vector that masks out the emotions that do not exist in current batch. Dim:[label_num]
        """
        '''Mask out unrelated instances for each emotion, and compute the cluster-level representation
         with the predicted VADs.'''
        masked_logits = one_hot_vad_labels.unsqueeze(2) * x.repeat(one_hot_vad_labels.shape[0], 1, 1) #[label_num, B, 3]
        logits = torch.mean(masked_logits, dim=1) #[label_num, 3]

        '''Compute logits for all clusters.'''
        logits = torch.div(torch.matmul(logits, label_VAD.T), temperature) #[label_num, label_num]

        '''Extract the logits to be maximised.'''
        up_logits = torch.diag(logits) #[label_num]

        '''Compute contrastive loss.'''
        all_logits = torch.log(torch.sum(torch.exp(logits), dim=1))
        loss = (up_logits-all_logits)*label_mask
        return -loss.mean()

    def mse_loss(self, x, vad_label):
        return self.MSE_loss(x.squeeze(1), vad_label)

    # input_data, masks, input_current, masks_current, input_context, masks_context
    def forward(self, x, mask, x_current, mask_current, x_context, mask_context, label_VAD, one_hot_vad_labels, label_mask):
        x = self.bert(x, attention_mask=mask)[0]
        x = x[:, 0, :].squeeze(1)
        x = self.dropout(x)
        x = self.dense_x(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        x_current = self.bert(x_current, attention_mask=mask_current)[0]
        x_current = x_current[:, 0, :].squeeze(1)
        x_current = self.dropout(x_current)
        x_current = self.dense_current(x_current)
        x_current = torch.tanh(x_current)
        x_current = self.dropout(x_current)
        
        x_context = self.bert(x_context, attention_mask=mask_context)[0]
        x_context = x_context[:, 0, :].squeeze(1)
        x_context = self.dropout(x_context)
        x_context = self.dense_current(x_context)
        x_context = torch.tanh(x_context)
        x_context = self.dropout(x_context)

        '''IB'''
        vad = torch.sigmoid(self.dense_x_vad(x))
        vad_current = torch.sigmoid(self.dense_current_vad(x))
        vad_context = torch.sigmoid(self.dense_context_vad(x))
        loss_ib_x = self.SCCL_loss(vad, label_VAD, one_hot_vad_labels, label_mask)
        loss_ib_current = self.SCCL_loss(vad_current, label_VAD, one_hot_vad_labels, label_mask)
        loss_ib_context = self.SCCL_loss(vad_context, label_VAD, one_hot_vad_labels, label_mask)
        loss_ib = loss_ib_x + loss_ib_current + loss_ib_context

        '''解纠缠网络'''
        latent_params_dis, dict_mi_losses, dict_mi, dvae_kl_loss,\
            recon_loss, contrastive_loss_un, contrastive_loss_red = self.DVAE_context(fused_x1_x2=x, x1=x_context, x2=x_current, u_mask=None, emb_spk=None)
        
        # 整合解耦的互信息损失
        mi_losses_sum = 0.0
        '''vCLUB '''        # dict_mi_losses, dict_mi = None, None
        for key_mi, mi_loss_latent_variable in dict_mi_losses.items():
            mi_losses_sum = mi_losses_sum + abs(mi_loss_latent_variable)
       
        # latent_params_dis['Syn'].z, latent_params_dis['Red'].z, latent_params_dis['Un1'].z, latent_params_dis['Un2'].z
        z_syn = self.gate_syn(latent_params_dis['Syn'].z)      # (B, d)
        z_red = self.gate_red(latent_params_dis['Red'].z)      # (B, d)
        z_un1 = self.gate_un1(latent_params_dis['Un1'].z)      # (B, d)
        z_un2 = self.gate_un2(latent_params_dis['Un2'].z)      # (B, d)
        # z_fusion = torch.cat([latent_params_dis['Syn'].z, latent_params_dis['Red'].z, latent_params_dis['Un1'].z, latent_params_dis['Un2'].z], dim=-1)
        # z_fusion = z_fusion.unsqueeze(0)
        z_fusion = torch.stack([z_syn, z_red, z_un1, z_un2], dim=1) # 改为 stack -> (B, 4, d)
        h_fusion, h_attn_weights = self.matt(z_fusion)
        # h_fusion: (B,4,d)
        # h_attn_weights: (B, n_heads, 4, 4)   # 多头注意力
        h_fusion = h_fusion.mean(dim=1)

        '''分类器'''
        prob = self.dense_class(h_fusion)
        prob = prob[:, :(self.num_classes_true)]
        
        return prob, loss_ib, vad,\
               mi_losses_sum, dvae_kl_loss, recon_loss,\
               contrastive_loss_un, contrastive_loss_red,\
               dict_mi, latent_params_dis,\
               h_attn_weights

