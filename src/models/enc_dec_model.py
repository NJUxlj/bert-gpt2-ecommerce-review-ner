import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical 

import os
import random
import numpy as np  


from typing import Dict, List, Optional, Tuple, Union, Literal
from src.models.bert.modeling_bert import BertModel
# from src.models.gpt2.modeling_gpt2 import GPT2Model


from src.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForTokenClassification


from src.models.qwen2.configuration_qwen2 import Qwen2Config
from src.models.bert.configuration_bert import BertConfig


from src.models.crf import CRF


from src.configs.config import BERT_MODEL_PATH, QWEN2_MODEL_PATH


from transformers import (
    AutoTokenizer,
    PreTrainedModel
    
)

from dataclasses import dataclass

from collections import defaultdict

class NerConfig:
    
    
    def __init__(
        self, 
        ner_data_type:Literal["chinese_ner_sft", "simple_ner"] = "chinese_ner_sft",
        label2id:Dict = defaultdict(int),
        num_ner_labels: int = 9
        ):
        
        
        self.ner_data_type = ner_data_type
    
        self.label2id = {
            "O": 0,
            "B-HCCX": 1,
            "B-MISC": 2,
            "B-HPPX": 3,
            "B-XH": 4,
            "I-HCCX": 5,
            "I-MISC": 6,
            "I-HPPX": 7,
            "I-XH": 8
        } if self.ner_data_type == "chinese_ner_sft" else {
            "B-LOCATION": 0,
            "B-ORGANIZATION": 1,
            "B-PERSON": 2,
            "B-TIME": 3,
            "I-LOCATION": 4,
            "I-ORGANIZATION": 5,
            "I-PERSON": 6,
            "I-TIME": 7,
            "O": 8
        }
        
        self.num_ner_labels = len(self.label2id.keys())
    

    
    
    
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
        top_k_weights, top_k_indices = probs.topk(k=self.top_k, dim=-1)
        
        
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
        base_offset =  torch.arange(batch_size*seq_len, device=x.device) \
                            .view(batch_size, seq_len, 1) \
                            *self.num_experts \
                            
        '''
        base_offset = torch.arange(batch_size*seq_len)  # 形状 (B*S,) → [0,1,2,3,4,5]
                         .unsqueeze(-1)                 # 形状 (B*S,1) → [[0],[1],[2],[3],[4],[5]]
                         * self.num_experts             # 形状 (B*S,1) → [[0],[4],[8],[12],[16],[20]]
                         .view(batch_size, seq_len, 1) # 重塑为 (B,S,1) → [[[0],[4],[8]], [[12],[16],[20]]]
        '''
        
        # 2. 广播相加（关键作用：为每个token生成独立索引空间）
            # 原来的 expert top_k_indices 的基础上， 加上 token 索引
            # 本质上来说， 就是让每个token的token索引与expert索引挂钩
            # 这个计算的核心作用是：为每个token创建独立的索引空间，确保不同token选择的专家索引不会冲突
        flat_indices = top_k_indices + base_offset  # 广播规则：(B,S,K) + (B,S,1) → (B,S,K)
        assert flat_indices.size() == (batch_size, seq_len, self.top_k), f"flat_indices size error, got {flat_indices.size()}, expected {(batch_size, seq_len, self.top_k)}"
        
        flat_indices = flat_indices.view(-1)  # 展平为 (B*S*K)  
        
        assert expert_outputs.size() == (batch_size, seq_len, self.num_experts, self.hidden_size), f"expert_outputs size error, got {expert_outputs.size()}, expected {(batch_size, seq_len, self.num_experts, self.hidden_size)}"  
        assert flat_indices.max() < (batch_size * seq_len * self.num_experts), "Index out of bound"  
        
             
        # 核心作用：从所有专家输出中筛选每个token对应的top-k专家结果
        selected_outputs = expert_outputs.view(-1, self.hidden_size)[   # 展平为：(B*S*N, H)
            flat_indices  
        ].view(batch_size, seq_len, self.top_k, -1)  # 恢复形状 → (B,S,K,H)
        
        
        
        print(f"expert_outputs shape: {expert_outputs.shape}")  
        print(f"flat_indices shape: {flat_indices.shape}")  
        print(f"selected_outputs shape: {selected_outputs.shape}")  
        
        
        # 专家输出的 加权求和（爱因斯坦求和约定, 实际上就是对位相乘）：
        # 输入张量形状：
            # selected_outputs: (B,S,K,H) 每个token对应的k个专家输出
            # top_k_weights: (B,S,K) 每个专家对应的路由权重
        '''
        根据打印的维度信息：

        selected_outputs形状为[16, 512, 2, 896] → (batch, seq_len, top_k, hidden)
        top_k_weights形状为[16, 512, 2] → (batch, seq_len, top_k)
        正确的爱因斯坦求和标记：

        bskh：对应selected_outputs的四个维度 (batch, seq_len, top_k, hidden)
        bsk：对应top_k_weights的三个维度 (batch, seq_len, top_k)
        bsh：输出维度 (batch, seq_len, hidden)
        '''
        output = torch.einsum('bskh,bsk->bsh', selected_outputs, top_k_weights)

        print("einsum_output.shape = ", output.shape)
        # 计算步骤分解：
            # 1. 维度扩展：将top_k_weights从(B,S,K)扩展为(B,S,K,1)
            # 2. 逐元素相乘：selected_outputs * top_k_weights → (B,S,K,H)
            # 3. 沿K维度求和：sum(dim=2) → (B,S,H)   # 表示每个token经过专家融合后的最终隐藏状态。
        
        # 平衡损失计算（专家负载均衡控制，防止专家坍塌）  
        # 计算专家使用频率（沿batch和sequence维度平均，得到每个专家的全局使用概率）
            # 示例：若有4个专家，可能得到 [0.3, 0.2, 0.4, 0.1]
        expert_usage = probs.mean(dim=[0,1])  # 形状：(num_experts,)  # 输入probs形状：(B,S,N) → 输出形状：(N,)
        print(f"expert_usage shape: {expert_usage.shape}")

        # 计算熵形式的平衡损失（鼓励均匀分布）
            # 当所有专家使用率相等时熵最大（平衡状态），损失值（负熵）最小
        self.balance_loss = - (expert_usage * torch.log(expert_usage + 1e-12)).sum()  # 标量值
        print(f"self.balance_loss: {self.balance_loss}")
        '''~
        该损失函数的作用：

        1. 当某些专家长期不被选择时（usage→0），log(usage)→-∞，但usage*log(usage)→0
        2. 当所有专家使用率相等时（usage=1/num_experts），熵值最大
        3. 通过最小化该损失，可以防止"专家坍塌"（少数专家主导整个模型）
        '''
        
        return output  
        

class BertMoEQwen2PreTrainedModel(nn.Module):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True  
    
    
    def __init__(self):
        super().__init__()
        self.config = Qwen2Config()
        
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
        
    

class BertMoEQwen2CRF(BertMoEQwen2PreTrainedModel):
    '''
    结合了Bert和Qwen2的命名实体识别模型， Bert作为Encoder， Qwen2作为Decoder, MoE插在Bert和Qwen2中间，Bert出来的每个Token都会选择一个Expert进行计算。计算得到后的隐向量再通过qwen2进行预测
    '''
    def __init__(self, bert_config: BertConfig, qwen_config:Qwen2Config, ner_config:NerConfig):
        super().__init__()  
        
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        print("BertMoEQwen2EncoderDecoder tokenizer 加载完毕 ~~~~")
        self.id2label = {i: tag for i, tag in enumerate(ner_config.label2id)}
        
        
        self.num_ner_labels = ner_config.num_ner_labels
        self.encoder = BertModel.from_pretrained(BERT_MODEL_PATH, config = bert_config)
        self.decoder = Qwen2Model.from_pretrained(QWEN2_MODEL_PATH, config = qwen_config)
        
        # 维度适配（假设BERT=768，Qwen2=1024） 
        self.dim_adapter = nn.Linear(bert_config.hidden_size, qwen_config.hidden_size)
        
        self.moe = MoEModel(num_experts = 16, hidden_size = qwen_config.hidden_size)

        self.classifier = nn.Linear(qwen_config.hidden_size, self.num_ner_labels)
        
        self.crf = CRF(self.num_ner_labels, batch_first = True)
        
        # 初始化权重 [除了 self.encoder, self.decoder]
        self._init_weights(self.dim_adapter)
        self._init_weights(self.moe)
        self._init_weights(self.classifier)
        

        
        
    def forward(
            self,
            input_ids, 
            attention_mask, 
            labels=None
        ):
        # BERT编码  
        encoder_outputs = self.encoder(  
            input_ids=input_ids,  
            attention_mask=attention_mask  
        ).last_hidden_state    # shape = (batch_size, seq_len, bert_hidden_size)
         
        adapted_hidden = self.dim_adapter(encoder_outputs) # shape = (batch_size, seq_len, qwen2_hidden_size)
        print("adapted_hidden.shape = ", adapted_hidden.shape)
        # MoE处理
        moe_output = self.moe(adapted_hidden)  # shape = (batch_size, seq_len, qwen2_hidden_size)
        
        
        # Qwen2
        decoder_outputs = self.decoder.forward(
            inputs_embeds = moe_output,
            attention_mask = attention_mask
        ).last_hidden_state  # shape = (batch_size, seq_len, qen2_hidden_size)
        
        
        logits = self.classifier.forward(decoder_outputs)  # shape = (batch_size, seq_len, num_ner_labels)
        
        outputs = (logits,)
        if labels is not None:
            
            # 使用CRF计算损失
            loss = -self.crf.forward(logits, labels, mask=attention_mask.byte())
            
            # loss_fct = nn.CrossEntropyLoss()
            # active_loss = attention_mask.view(-1) == 1   # attention_mask.view(-1) = (batch_size * seq_len)
            # active_logits = logits.view(-1, self.num_ner_labels)[active_loss]  # shape = (num_active_tokens, num_ner_labels)
            # active_labels = labels.view(-1)[active_loss]   # shape = (num_active_tokens,)
            # loss = loss_fct(active_logits, active_labels)  
            
            # 添加路由平衡正则项  
            loss += 0.01 * self.moe.balance_loss  
            outputs = (loss,) + outputs  
        
        return outputs
    
    def predict(self, input_ids):
        '''
        用于推理的成员函数， 为input_ids中的每个token预测一个实体标签。
        根据预测出的实体标签，将每个实体字符串提取出来，形成 [{实体字符串:{"start_pos":..., "end_pos":..., "entity_type":...}}, {...}] 的字典列表
        '''
        
        self.eval()
        with torch.no_grad():
            # 生成attention mask（假设无padding）
            attention_mask = torch.ones_like(input_ids)
            
            # 前向传播
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs[0]  # (batch_size, seq_len, num_ner_labels)
            
            # 获取预测标签
            # preds = logits.argmax(-1).squeeze().cpu().numpy()  # (batch_size, seq_len)

            # 使用CRF解码获取预测标签
            preds = self.crf.decode(logits, mask=attention_mask.byte())
            preds = preds[0]  # 取batch中的第一个样本
    
    
        # 提取实体
        entities = []
        current_entity = None
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        for i, (token, tag_id) in enumerate(zip(tokens, preds)):
            # 此处需要根据实际标签定义转换id到标签（如BIO/BILOU格式）
            tag = self.id2label[tag_id]  # 需要用户提供id2label映射
            
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "start_pos": i,
                    "end_pos": i,
                    "entity_type": tag.split("-")[1]
                }
            elif tag.startswith("I-"):
                if current_entity and current_entity["entity_type"] == tag.split("-")[1]:
                    current_entity["text"] += token
                    current_entity["end_pos"] = i
                else:
                    current_entity = None  # 非连续实体则丢弃
            else: # O标签
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                    
        if current_entity:  # 添加最后一个实体
            entities.append(current_entity)

        # 格式转换
        result = [{e["text"]: {
            "start_pos": e["start_pos"],
            "end_pos": e["end_pos"],
            "entity_type": e["entity_type"]
        }} for e in entities]
        
        return result
        
        



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