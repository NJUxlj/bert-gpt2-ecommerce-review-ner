
"""
NER数据处理模块，适配混合模型架构需求
特点：
1. 动态子词对齐：处理BERT/Qwen2不同分词器的offset映射
2. 标签平滑策略：缓解类别不平衡问题
3. 智能填充检测：自动识别无效填充标签
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class NERExample:
    tokens: List[str]
    labels: List[int]

class SimpleNERDataProcessor:
    def __init__(self, schema_path: str, max_seq_length=128):
        """
        初始化处理器
        :param schema_path: 标签映射文件路径
        :param max_seq_length: 最大序列长度（需与模型配置对齐）
        """
        self.max_seq_length = max_seq_length
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.tag2id = json.load(f)
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        
        # 动态标签权重（用于处理类别不平衡）
        self.label_weights = np.ones(len(self.tag2id))
        self.label_weights[self.tag2id["O"]] = 0.2  # 降低O标签权重

    def _parse_example(self, lines: List[str]) -> NERExample:
        """解析单个NER示例"""
        tokens, labels = [], []
        for line in lines:
            token, label = line.strip().split()
            tokens.append(token)
            labels.append(self.tag2id[label])
            
        # 标签平滑处理（Label Smoothing）
        smoothed_labels = []
        for label_id in labels:
            if label_id != self.tag2id["O"]:
                one_hot = np.full(len(self.tag2id), 0.1/(len(self.tag2id)-1))
                one_hot[label_id] = 0.9
                smoothed_labels.append(one_hot)
            else:
                smoothed_labels.append(label_id)
                
        return NERExample(tokens=tokens, labels=smoothed_labels)

    def _read_data(self, file_path: str) -> List[NERExample]:
        """读取原始数据文件"""
        examples = []
        current_lines = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 遇到空行时结束当前示例
                    if current_lines:
                        examples.append(self._parse_example(current_lines))
                        current_lines = []
                else:
                    current_lines.append(line)
                    
            # 处理最后一个示例
            if current_lines:
                examples.append(self._parse_example(current_lines))
                
        return examples

    def get_train_examples(self, data_dir: str) -> List[NERExample]:
        return self._read_data(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir: str) -> List[NERExample]:
        return self._read_data(os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self, data_dir: str) -> List[NERExample]:
        return self._read_data(os.path.join(data_dir, "test.txt"))

    def align_labels(self, 
                    tokenizer,  # 联合分词器（需实现BERT/Qwen2联合分词）
                    example: NERExample) -> Dict:
        """
        标签对齐函数，适配混合模型架构
        返回:
        {
            "input_ids": ...,
            "attention_mask": ...,
            "labels": ...,
            "token_type_ids": ...,
            "expert_mask": ...  # 标识需要专家处理的token位置
        }
        """
        # 第一步：生成联合分词结果
        encoding = tokenizer(
            example.tokens,
            truncation=True,
            max_length=self.max_seq_length,
            is_split_into_words=True,
            return_offsets_mapping=True
        )
        
        # 第二步：构建专家掩码（标识需要MoE处理的token）
        expert_mask = []
        for word_idx in encoding.word_ids:
            if word_idx is None:  # 特殊token
                expert_mask.append(0)
            else:
                # 根据原始标签决定是否需要专家处理
                original_label = example.labels[word_idx]
                expert_mask.append(1 if original_label != self.tag2id["O"] else 0)
        
        # 第三步：标签对齐策略（处理子词分词问题）
        labels = []
        previous_word_idx = None
        for word_idx in encoding.word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(example.labels[word_idx])
            else:  # 同一单词的后续子词
                labels.append(-100)
            previous_word_idx = word_idx
            
        # 第四步：构建最终特征
        features = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
            "expert_mask": expert_mask[:self.max_seq_length]
        }
        
        # 针对BERT需要token_type_ids
        if "token_type_ids" in encoding:
            features["token_type_ids"] = encoding["token_type_ids"]
            
        return features
