import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
import numpy as np

from src.models.bert.modeling_bert import BertModel
# from src.models.gpt2.modeling_gpt2 import GPT2Model


from src.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForTokenClassification


from src.models.qwen2.configuration_qwen2 import Qwen2Config
from src.models.bert.configuration_bert import BertConfig


from transformers import (
    AutoTokenizer,
    PreTrainedModel
    
)

from dataclasses import dataclass

@dataclass
class NerConfig:
    num_ner_labels: int
    
    
    
class Expert(nn.Module):  
    def __init__(self, hidden_size, dropout=0.1):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(hidden_size, 4*hidden_size),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(4*hidden_size, hidden_size)  
        )  
    
    def forward(self, x):  
        return self.net(x)  


class MoEModel(nn.Module):
    def __init__(self, num_experts, hidden_size, top_k=2):
        super(MoEModel, self).__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(hidden_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        self.balance_loss = 0.0
        
        
        
    def forward(self, x):
        '''
        x.shape = (batch_size, seq_len, hidden_size)
        '''
        # 路由计算
        logits = self.gate(x.detach())  # (batch_size, seq_len, num_experts)

        probs = F.softmax(logits, dim=-1)
        
        # tok_k_weights.shape = (batch_size, seq_len, top_k), 其中一个token对应的topk-weights 为 [0.2, 0.8]
        # tok_k_indices.shape = (batch_size, seq_len, top_k)， 其中一个token对应的topk-weights 为 [0, 1]
        top_k_weights, topk_indices = probs.topk(probs, dim=-1)
        
        
        # 专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))   # (batch_size, seq_len, hidden_size)

        # 确保每个token都能对应一个形状为 (num_experts, hidden_size) 的矩阵
        expert_outputs = torch.stack(expert_outputs, dim=2)  # (batch_size, seq_len, num_experts, hidden_size)
        

class BertMoEQwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config  # 暂时借用一下qwen2的config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    def __init__():
        super().__init__()
        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
    

class BertMoEQwen2EncoderDecoder(BertMoEQwen2PreTrainedModel):
    '''
    结合了Bert和Qwen2的命名实体识别模型， Bert作为Encoder， Qwen2作为Decoder, MoE插在Bert和Qwen2中间，Bert出来的每个Token都会选择一个Expert进行计算。计算得到后的隐向量再通过qwen2进行预测
    '''
    def __init__(self, bert_config: BertConfig, qwen_config:Qwen2Config, ner_config:NerConfig):
        super().__init__()  

        self.num_ner_labels = ner_config.num_ner_labels
        self.encoder = BertModel.from_pretrained(config = bert_config)
        self.decoder = Qwen2ForTokenClassification.from_pretrained(config = qwen_config)
        self.classifier = nn.Linear(qwen_config.hidden_size, self.num_ner_labels)
        
        self.moe = MoEModel(num_experts = 16, hidden_size = qwen_config.hidden_size)

        
        # 初始化权重
        self.post_init()

        
        
    def forward(self,input_ids, attention_mask, labels=None):
        pass
    
    
    
        
        
        
        
        


'''
下面的代码已弃用
'''

# class BertGpt2EncoderDecoder(nn.Module):  
#     def __init__(self, bert_config, gpt2_config, ner_config:NerConfig):  
#         super().__init__()  
#         self.num_ner_labels = ner_config.num_ner_labels
        
#         self.encoder = BertModel(bert_config)  
#         self.decoder = GPT2Model(gpt2_config)  
#         self.classifier = nn.Linear(gpt2_config.n_embd, self.num_ner_labels)  
        
#         # 连接适配层  
#         self.adapter = nn.Linear(bert_config.hidden_size, gpt2_config.n_embd)  
        
#     def forward(self, encoder_input, decoder_input, labels=None):  
#         encoder_outputs = self.encoder(**encoder_input)  
#         hidden_states = self.adapter(encoder_outputs.last_hidden_state)  
        
#         decoder_outputs = self.decoder(  
#             input_ids=decoder_input['input_ids'],  
#             attention_mask=decoder_input['attention_mask'],  
#             encoder_hidden_states=hidden_states  
#         )  
#         logits = self.classifier(decoder_outputs.last_hidden_state)  
#         # 计算序列标注损失...  
#         return logits  