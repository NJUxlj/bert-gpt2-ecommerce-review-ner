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
        top_k_weights, top_k_indices = probs.topk(probs, dim=-1)
        
        
        # 专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))   # (batch_size, seq_len, hidden_size)

        # 确保每个token都能对应一个形状为 (num_experts, hidden_size) 的矩阵
        expert_outputs = torch.stack(expert_outputs, dim=2)  # (batch_size, seq_len, num_experts, hidden_size)


        # 专家输出的稀疏组合  
        batch_size, seq_len, _ = x.size()   # 输入形状：x.shape=(B,S,H)
        # 将三维输入展平为二维索引：(batch_size*seq_len, 1)
        # 分解计算步骤（假设 B=2, S=3, K=2, N=4）
        # 1. 生成基础偏移量矩阵
        base_offset =  torch.arange(batch_size*seq_len, device=x.device). \
                            unsqueeze(-1) \
                            *self.num_experts \
                            .view(batch_size, seq_len, 1)
        # top_k_indices形状：(B,S,K) → 展平后：(B*S*K,)
        # 最终flat_indices形状：(B,S,K)
        
        # 2. 广播相加（关键作用：为每个token生成独立索引空间）
            # 原来的 expert top_k_indices 的基础上， 加上 token 索引
            # 本质上来说， 就是让每个token的token索引与expert索引挂钩
        flat_indices = top_k_indices + base_offset  # 广播规则：(B,S,K) + (B,S,1) → (B,S,K)
        
        

        selected_outputs = expert_outputs.view(-1, self.num_experts, self.hidden_size)[   # 展平为：(B*S, N, H)
            flat_indices.view(-1, self.top_k)  # 索引形状：(B*S, K)
        ].view(batch_size, seq_len, self.top_k, self.hidden_size)  # 恢复形状 → (B,S,K,H)
        
        # 加权求和（爱因斯坦求和约定）：
        output = torch.einsum('bstk,bst->bsth', selected_outputs, top_k_weights).sum(dim=-2)  
        # selected_outputs形状：(B,S,K,H), top_k_weights形状：(B,S,K)
        # einsum结果形状：(B,S,H) → sum后保持形状不变
        
        # 平衡损失计算（专家负载均衡控制，防止专家坍塌）  
        # 计算专家使用频率（沿batch和sequence维度平均）
        expert_usage = prots.mean(dim=[0,1])  # 形状：(num_experts,)  # 输入prots形状：(B,S,N) → 输出形状：(N,)
        # 计算熵形式的平衡损失（鼓励均匀分布）
        self.balance_loss = (expert_usage * torch.log(expert_usage + 1e-12)).sum()  # 标量值

        '''
        该损失函数的作用：

        1. 当某些专家长期不被选择时（usage→0），log(usage)→-∞，但usage*log(usage)→0
        2. 当所有专家使用率相等时（usage=1/num_experts），熵值最大
        3. 通过最小化该损失，可以防止"专家坍塌"（少数专家主导整个模型）
        '''
        
        return output  
        

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