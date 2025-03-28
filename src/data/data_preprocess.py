
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

from datasets import load_dataset, DatasetDict, Dataset, load_from_disk

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
        # tag_prefix = "B" if scheme == "BILOU" else "B"  
        # labels[entity.start_idx] = f"{tag_prefix}-{entity.label}"  
        # for i in range(entity.start_idx + 1, entity.end_idx):  
        #     if scheme == "BILOU" and i == entity.end_idx -1:  
        #         labels[i] = f"L-{entity.label}"  
        #     else:  
        #         labels[i] = f"I-{entity.label}"  
        
        if scheme == "BILOU":  
            entity_length = entity.end_idx - entity.start_idx  
            if entity_length == 1:  
                labels[entity.start_idx] = f"U-{entity.label}"  
            else:  
                labels[entity.start_idx] = f"B-{entity.label}"  
                for i in range(entity.start_idx + 1, entity.end_idx):  
                    if i == entity.end_idx - 1:  
                        labels[i] = f"L-{entity.label}"  
                    else:  
                        labels[i] = f"I-{entity.label}"  
        else:  # BIO 模式  
            labels[entity.start_idx] = f"B-{entity.label}"  
            for i in range(entity.start_idx + 1, entity.end_idx):  
                labels[i] = f"I-{entity.label}"  
        
        last_end = entity.end_idx  

    return labels  

class NERDataProcessor:  
    def __init__(self,  
                 tokenizer: PreTrainedTokenizerBase,  
                 label_scheme: str = "BIO",  
                 max_length: int = 512,  
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
        self.id2tag = {v: k for k, v in self.label_map.items()}
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
                
    def load_and_process(self, data_path: str, train_size = 500, valid_size = 500, test_size = 500) -> DatasetDict:  
        
        train_size:int = train_size
        valid_size:int = valid_size
        test_size:int = test_size
        
        all_data = []
        dataset = {"train": [], "validation": [], "test": []}  
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
                    # print("processed.type = ", type(processed)) 
                    # print("processed.keys() = ", processed.keys())
                    if not processed:  
                        continue  
                        
                    # 收集标签统计  
                    for label in char_labels:  
                        label_counter[label] += 1  
                        
                    # TODO: 划分训练集/验证集/测试集  
                    all_data.append(processed)  
                    
                    
        # print("all_data.type", type(all_data))
        # print(all_data[:10])
        dataset["train"] = all_data[:train_size]  
        dataset["validation"] = all_data[train_size:train_size+valid_size]  
        dataset["test"] = all_data[train_size+valid_size:train_size+valid_size+test_size]  
        
        # 自动构建label映射（如果未提供）  
        if not self.label_map:  
            labels = sorted(label_counter.keys(), key=lambda x: (x.split("-")[-1], x))  
            self.label_map = {label: idx for idx, label in enumerate(labels)}  
            self._verify_label_scheme()  
            
        from matplotlib import pyplot as plt  
        plt.bar(label_counter.keys(), label_counter.values())  
        plt.xticks(rotation=45)  
        plt.show()  
        
        
        # 将列表转换为列式字典  
        train_column_data = {  
            "input_ids": [x["input_ids"] for x in dataset["train"]],  
            "attention_mask": [x["attention_mask"] for x in dataset["train"]],  
            "labels": [x["labels"] for x in dataset["train"]]  
        }  
        
        validation_column_data = {
            "input_ids": [x["input_ids"] for x in dataset["validation"]],
            "attention_mask": [x["attention_mask"] for x in dataset["validation"]],
            "labels": [x["labels"] for x in dataset["validation"]]
        }
        
        test_column_data = {
            "input_ids": [x["input_ids"] for x in dataset["test"]],
            "attention_mask": [x["attention_mask"] for x in dataset["test"]],
            "labels": [x["labels"] for x in dataset["test"]]
        }

    
        
        # 转换为HF Dataset格式  
        # return DatasetDict({  
        #     split: Dataset.from_dict(data)    # data 的类型是 List[Dict]
        #     for split, data in dataset.items()  
        # }) 
        
        return DatasetDict({  
            "train": Dataset.from_dict(train_column_data),  
            "validation": Dataset.from_dict(validation_column_data),  
            "test": Dataset.from_dict(test_column_data)  
        })   
    
    def _tokenize_and_align_labels(self, tokens: List[str], char_labels: List[str]) -> Optional[Dict]:  
        """
        核心对齐逻辑
        
        1. 你必须确保你使用的bert是 bert-base-chinese, 因为这样tokenizer会直接按照字/词来进行分词
        
        2. 然后 一个 字/词 才能对应一个 NER label
        
        input_ids 和 word_ids 的区别：
        
        Token:  101 → Word index: None  # [CLS]  
        Token: 2769 → Word index: 0     # 我  
        Token: 4263 → Word index: 1     # 喜  
        Token: 3341 → Word index: 2     # 欢  
        Token: 1355 → Word index: 3     # 自  
        Token: 1767 → Word index: 4     # 然  
        Token: 2158 → Word index: 5     # 语  
        Token:  702 → Word index: 6     # 言  
        Token:  511 → Word index: 7     # 处  
        Token:  102 → Word index: None  # [SEP]  
        Token:    0 → Word index: None  # [PAD]  
        Token:    0 → Word index: None  # [PAD]  
        
        """  
        tokenized = self.tokenizer(  
            tokens,  
            is_split_into_words=True,  
            truncation=True,  
            max_length=self.max_length,
            padding = "max_length",
            return_offsets_mapping=True,  # 必须要返回偏移量映射，因为我们使用的是 bert-base-chinese, 他会自动在中文词之前加上两个##  
            return_token_type_ids=False  
        )  
        
        aligned_labels = []  
        word_ids = tokenized.word_ids()
        
        # 直接通过 token 与原始字符的一一对应关系处理标签  
        for i, word_idx in enumerate(tokenized.word_ids()):   
            # 跳过特殊标记 [CLS]/[SEP] 等（具体根据分词器情况调整）  
            if tokenized.input_ids[i] in [self.tokenizer.cls_token_id,   
                                    self.tokenizer.sep_token_id,  
                                    self.tokenizer.pad_token_id]:  
                aligned_labels.append(-100)  
                continue  
                
            # 处理截断后的token（重要！） 
            if word_idx is None or i >= len(word_ids):  
                aligned_labels.append(-100)  
                continue  

            # 中文BERT的特殊处理：通过offset mapping验证  
            start, end = tokenized.offset_mapping[i]  # offset_mapping的作用：告诉我们token对应原始字符的起始和结束位置
            if start==end==0: # 特殊token
                aligned_labels.append(-100)
            else:
                aligned_labels.append(self.label_map.get(char_labels[word_idx],-100))
                
            # 直接映射字符级标签（无子词转换）  
            label = char_labels[word_idx]  
            aligned_labels.append(self.label_map.get(label, -100))  # 防止未知标签  

        # 强制标签长度对齐（防止max_length截断导致长度不一致）  
        aligned_labels = aligned_labels[:self.max_length] + [-100]*(self.max_length - len(aligned_labels)) 
        
        
        # 有效性检查  
        if all(lb == -100 for lb in aligned_labels):  
            return None  
            
        return {  
            "input_ids": tokenized["input_ids"],  
            "attention_mask": tokenized["attention_mask"],  
            "labels": aligned_labels  
        }  
        
        
    def load_hf_data(self, data_path, split = 'train'):
        '''
        load huggingface format dataset
        '''
        data_files = {  
            "train": f"{data_path}/train/data-00000-of-00001.arrow",  
            "validation": f"{data_path}/validation/data-00000-of-00001.arrow",  
            "test": f"{data_path}/test/data-00000-of-00001.arrow"  
        } 
        
        ds = load_dataset(data_path, data_files = data_files, split = split)
        
        print("数据集结构示例:", ds)  
        print("特征字段:", ds.features)  
        print("首样本:", ds[0]) 
                
        return ds
    
    
    def load_json_data(self, data_path, split = "train"):
        # 假设数据文件结构为：  
        # data_path/  
        #   train.json  
        #   validation.json  
        #   test.json  
        
        data_files = {  
            "train": f"{data_path}/train.json",  
            "validation": f"{data_path}/validation.json",  
            "test": f"{data_path}/test.json"  
        }  
        
        # 明确指定加载本地文件  
        ds = load_dataset(  
            "json",   
            data_files=data_files,  
            split=split  
        )  
        return ds  
        

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

