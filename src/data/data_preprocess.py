
"""  
适配 JSON 实体标注格式的数据处理器，支持字符级位置标注转换  
功能：  
1. 自动验证实体边界是否合法  
2. 支持BIO/BILOU标注格式切换  
3. 提供多粒度校验（字符级/分词级）  
"""  

from typing import Dict, List, Tuple, Optional
import os
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from datasets import load_dataset, DatasetDict, Dataset

import logging
logger = logging.getLogger(__name__) 
from ..configs.config import CHINESE_NER_DATA_PATH



@dataclass  
class EntityAnnotation:  
    text: str  
    label: str  
    start_idx: int  
    end_idx: int  # 包含最后一个字符的位置 [start, end]  

def validate_entity(text: str, entity: EntityAnnotation) -> bool:  
    """严格验证实体标注的正确性"""  
    if entity.start_idx >= entity.end_idx:  
        logger.warning(f"Invalid span {entity}: start >= end")  
        return False  
    
    if entity.end_idx > len(text):  
        logger.warning(f"Entity {entity} exceeds text length ({len(text)})")  
        return False  
    
    actual_text = text[entity.start_idx:entity.end_idx]  
    if actual_text != entity.text:  
        logger.warning(f"Text mismatch: '{actual_text}' vs '{entity.text}' in span {entity}")  
        return False  
    
    return True  

def convert_entities_to_bio(text: str, entities: List[EntityAnnotation], scheme: str = "BIO") -> List[str]:  
    """  
    将实体标注 (如，HCCX) 转换为字符级别的标签序列  (如，B-HCCX, I-HCCX, O)
    scheme: BIO (默认) 或 BILOU  
    """  
    labels = ["O"] * len(text)  
    
    # 按起始位置排序  
    '''
    主排序条件（x.start_idx）：按实体的起始位置升序排列，确保先处理文本中靠前的实体

    次排序条件（-x.end_idx）：当起始位置相同时，按结束位置降序排列（即较长的实体优先处理）。这是为了处理嵌套实体场景：
    '''
    sorted_entities = sorted(entities, key=lambda x: (x.start_idx, -x.end_idx))  
    
    # 排除冲突标注  
    last_end = -1   # 记录上一个实体的尾指针
    for entity in sorted_entities:  
        if not validate_entity(text, entity):  
            continue  
        
        # 检测重叠冲突  
        if entity.start_idx < last_end:  
            logger.warning(f"Overlapping entity: {entity} (last end: {last_end})")  
            continue  
        
        # 应用标签  
        tag_prefix = "B" if scheme == "BILOU" else "B"  
        labels[entity.start_idx] = f"{tag_prefix}-{entity.label}"  
        for i in range(entity.start_idx + 1, entity.end_idx):  
            if scheme == "BILOU" and i == entity.end_idx -1:  
                labels[i] = f"L-{entity.label}"  
            else:  
                labels[i] = f"I-{entity.label}"  
        
        last_end = entity.end_idx  
        
    return labels  

class NERDataProcessor:  
    def __init__(self,  
                 tokenizer: PreTrainedTokenizerBase,  
                 label_scheme: str = "BIO",  
                 max_length: int = 256,  
                 label_vocab: Optional[Dict[str, int]] = None):  
        self.tokenizer = tokenizer  
        self.label_scheme = label_scheme  
        self.max_length = max_length  
        
        # 动态构建label词汇表  
        self.label_map = label_vocab or {
                                            "O": 0,
                                            "B-HCCX": 1,
                                            "B-MISC": 2,
                                            "B-HPPX": 3,
                                            "B-XH": 4,
                                            "I-HCCX": 5,
                                            "I-MISC": 6,
                                            "I-HPPX": 7,
                                            "I-XH": 8
                                        } 
        self._verify_label_scheme()  

    def _verify_label_scheme(self):  
        """确保标签格式与方案兼容"""  
        valid_prefix = {"B", "I"} if self.label_scheme == "BIO" else {"B", "I", "L", "U"}  
        
        for label in self.label_map:  
            if label == "O":  
                continue  
            parts = label.split("-")  
            if len(parts) != 2 or parts[0] not in valid_prefix:  
                raise ValueError(f"Invalid label '{label}' for scheme {self.label_scheme}")  
                
    def load_and_process(self, data_path: str) -> DatasetDict:  
        dataset = {"train": [], "valid": [], "test": []}  
        label_counter = defaultdict(int)  
        
        # 自动生成数据划分（示例比例可按需调整）  
        paths = list(Path(data_path).glob("*.jsonl")) 
        paths.extend(list(Path(data_path).glob("*.json")))
        if not paths:  
            raise FileNotFoundError(f"No JSON files found in {data_path}")  
        
        # 处理每个文件  
        for file_path in paths:  
            with open(file_path, "r", encoding="utf-8") as f:  
                for line in f:  
                    data = json.loads(line)  
                    
                    # 转换实体标注  
                    entities = [  
                        EntityAnnotation(  
                            text=e["entity_text"],  
                            label=e["entity_label"],  
                            start_idx=e["start_idx"],  
                            end_idx=e["end_idx"]  # 需要确认原始数据的end_idx是否包含最后一个字符  
                        ) for e in data["entities"]  
                    ]  
                    
                    # 生成字符级标签  
                    char_labels = convert_entities_to_bio(data["text"], entities, self.label_scheme)  
                    tokens = list(data["text"])  # 字符级分词  
                    
                    # Tokenize并构建标签对齐  
                    processed = self._tokenize_and_align_labels(tokens, char_labels)  
                    if not processed:  
                        continue  
                        
                    # 收集标签统计  
                    for label in char_labels:  
                        label_counter[label] += 1  
                        
                    # TODO: 划分训练集/验证集/测试集  
                    dataset["train"].append(processed)  
        
        # 自动构建label映射（如果未提供）  
        if not self.label_map:  
            labels = sorted(label_counter.keys(), key=lambda x: (x.split("-")[-1], x))  
            self.label_map = {label: idx for idx, label in enumerate(labels)}  
            self._verify_label_scheme()  
        
        # 转换为HF Dataset格式  
        return DatasetDict({  
            split: Dataset.from_list(data)   
            for split, data in dataset.items()  
        })  
    
    def _tokenize_and_align_labels(self, tokens: List[str], char_labels: List[str]) -> Optional[Dict]:  
        """核心对齐逻辑"""  
        tokenized = self.tokenizer(  
            tokens,  
            is_split_into_words=True,  
            truncation=True,  
            max_length=self.max_length,  
            return_offsets_mapping=True,  
            return_token_type_ids=False  
        )  
        
        word_ids = tokenized.word_ids()  
        aligned_labels = []  
        last_word_idx = -1  
        
        for word_idx in word_ids:  
            if word_idx is None:  
                aligned_labels.append(-100)  
            elif word_idx != last_word_idx:  
                # 当前token是词的起始位置  
                aligned_labels.append(self.label_map[char_labels[word_idx]])  
                last_word_idx = word_idx  
            else:  
                # 处理子词（将B-转换为I-）  
                original_label = char_labels[word_idx]  
                if original_label.startswith("B-"):  
                    converted = original_label.replace("B-", "I-")  
                    aligned_labels.append(self.label_map.get(converted, -100))  
                else:  
                    aligned_labels.append(self.label_map.get(original_label, -100))  
        
        # 验证对齐有效性  
        valid_labels = [lb for lb in aligned_labels if lb != -100]  
        if not valid_labels:  
            return None  # 跳过无有效标签的样本  
        
        return {  
            "input_ids": tokenized["input_ids"],  
            "attention_mask": tokenized["attention_mask"],  
            "labels": aligned_labels  
        }  

# 使用示例  
if __name__ == "__main__":  
    from transformers import AutoTokenizer  
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")  
    processor = NERDataProcessor(  
        tokenizer=tokenizer,  
        label_scheme="BIO"  
    )  
    
    dataset = processor.load_and_process(CHINESE_NER_DATA_PATH)  
    dataset.save_to_disk("./data/processed_ner_data")  

