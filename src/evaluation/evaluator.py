
"""
NER模型评估器，支持细粒度分析：
1. 精确边界匹配（Exact Boundary Match）
2. 部分匹配（Partial Boundary Match）
3. 类型敏感/不敏感评估
4. 分层指标（micro/macro平均）
"""

import torch
import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import defaultdict


from typing import Union, List, Dict, Tuple, Optional, Callable, Set


from src.data.data_preprocess import NERDataProcessor

from src.data.simple_data_preprocess import SimpleNERDataProcessor

class NEREvaluator:
    def __init__(self, model, processor:Union[NERDataProcessor, SimpleNERDataProcessor], tokenizer, device="cuda"):
        self.model = model.to(device)
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        
        # 初始化指标容器
        self.metrics = {
            'strict': defaultdict(int),
            'partial': defaultdict(int),
            'type_only': defaultdict(int)
        }
    
    def _convert_ids_to_labels(self, ids):
        """将标签ID序列转换为BIO标签"""
        return [self.processor.id2tag.get(i, "O") for i in ids]

    def _postprocess(self, predictions, labels)->Tuple[List[List[str]], List[List[str]]]:
        """对齐预测结果与真实标签
        
        prediction.shape = List[(seq_len)]
        
        labels.shape = List[(seq_len)]
        
        """
        true_predictions = []
        true_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            # 过滤padding和特殊token
            filtered_pred = []
            filtered_label = []
            
            for p, l in zip(pred_seq, label_seq):
                if l != -100:  # 忽略填充位置
                    filtered_pred.append(p)
                    filtered_label.append(l)
                    
            # 转换标签格式
            pred_tags = self._convert_ids_to_labels(filtered_pred)
            true_tags = self._convert_ids_to_labels(filtered_label)
            
            true_predictions.append(pred_tags)
            true_labels.append(true_tags)
            
        return true_predictions, true_labels

    def _calculate_entity_metrics(self, true_sequence, pred_sequence):
        """计算细粒度实体指标"""
        # 提取实体
        true_entities = self._get_entities(true_sequence)
        pred_entities = self._get_entities(pred_sequence)
        
        # 精确匹配（严格模式）
        strict_matches = set(pred_entities) & set(true_entities)
        self.metrics['strict']['TP'] += len(strict_matches)
        self.metrics['strict']['FP'] += len(pred_entities - strict_matches)
        self.metrics['strict']['FN'] += len(true_entities - strict_matches)
        
        
        # 精确匹配，非严格模式
        
        
        # 部分匹配
        partial_matches = []
        for pred in pred_entities:
            for true in true_entities:
                if self._is_partial_match(pred, true):
                    partial_matches.append(pred)
                    break
        
        self.metrics['partial']['TP'] += len(partial_matches)
        self.metrics['partial']['FP'] += len(pred_entities) - len(partial_matches)
        
        # 仅类型匹配
        type_only = []
        for pred in pred_entities:
            for true in true_entities:
                if pred[0] == true[0]:
                    type_only.append(pred)
                    break
        
        self.metrics['type_only']['TP'] += len(type_only)
        self.metrics['type_only']['FP'] += len(pred_entities) - len(type_only)

    @staticmethod
    def _get_entities(seq)->Set[Tuple]:
        """从BIO序列中提取实体"""
        entities = []
        current_entity = None
        
        for i, tag in enumerate(seq):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'type': tag[2:], 'start': i, 'end': i}
            elif tag.startswith('I-'):
                if current_entity and tag[2:] == current_entity['type']:
                    current_entity['end'] = i
                else:
                    if current_entity:
                        entities.append(current_entity) # 当前I标签的类型与当前实体不同，则赶紧保存当前实体，并且开始记录一个新的实体
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        return set([
            (e['type'], e['start'], e['end']) 
            for e in entities
        ])

    @staticmethod
    def _is_partial_match(pred, true):
        """判断部分匹配条件"""
        pred_type, pred_start, pred_end = pred
        true_type, true_start, true_end = true
        
        overlap_start = max(pred_start, true_start)
        overlap_end = min(pred_end, true_end)
        return (overlap_start < overlap_end) and (pred_type == true_type)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        print("eval dataset structure:", eval_dataset)  
        # print("eval dataset features  =", dataset.features)
        
        with torch.no_grad():
            for example in tqdm(eval_dataset, desc="Evaluating"):
                
                # print("example = ", example)
                
                assert isinstance(example, dict) and "labels" in example, "example must be a dict and must contain 'labels' key"
                
                if isinstance(self.processor, SimpleNERDataProcessor):
                    features = self.processor.align_labels(
                        self.tokenizer, example
                    )
                else:
                    # features = self.processor._tokenize_and_align_labels(
                    #     self.tokenizer, example["labels"]
                    # )
                    features = example
                
                
                # 转为张量
                inputs = {
                    'input_ids': torch.tensor([features['input_ids']], device=self.device),
                    'attention_mask': torch.tensor([features['attention_mask']], device=self.device)
                }
                if 'token_type_ids' in features:
                    inputs['token_type_ids'] = torch.tensor(
                        [features['token_type_ids']], device=self.device
                    )
                
                # 前向传播
                outputs = self.model(**inputs)
                logits = outputs[0][0].to(torch.float16).detach().cpu().numpy() # shape = (seq_len, num_ner_labels)

                # print("logits.shape = ", logits.shape)
                
                # 获取预测结果
                preds = np.argmax(logits, axis=1) # shape = (seq_len, )
                # print("preds.shape = ", preds.shape)
                labels = features['labels'] # shape = (seq_len, )
                
                all_preds.append(preds)
                all_labels.append(labels)
                
        print("最后一组预测值和标签： ")
        # print("preds = ", all_preds[-1])
        # print("labels = ", all_labels[-1])
        
        # 处理结果
        pred_sequences, label_sequences = self._postprocess(all_preds, all_labels)
        
        
        print("pred_sequences[-1] = ", pred_sequences[-1])
        print("label_sequences[-1] = ", label_sequences[-1])
        
        # 计算指标
        results = {}
        
        # from sklearn.metrics import f1_score
        # flat_preds = [p for seq in pred_sequences for p in seq]
        # flat_labels = [l for seq in label_sequences for l in seq]
        # f1 = f1_score(flat_labels, flat_preds, average='macro')  # 使用macro平均
        # results[f'{metric_key_prefix}_f1_score'] = f1
        
        flat_preds = [p for seq in pred_sequences for p in seq]
        flat_labels = [l for seq in label_sequences for l in seq]
        
        # 手动计算F1-score
        unique_labels = set(flat_labels + flat_preds)
        f1_scores = []
        
        for label in unique_labels:
            if label == 'O':  # 通常不计算'O'标签的F1
                continue
                
            tp = sum((p == label) & (l == label) for p, l in zip(flat_preds, flat_labels))
            fp = sum((p == label) & (l != label) for p, l in zip(flat_preds, flat_labels))
            fn = sum((p != label) & (l == label) for p, l in zip(flat_preds, flat_labels))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        # 计算macro平均F1
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        results[f'{metric_key_prefix}_f1_score'] = macro_f1
        
        
        # 1. seqeval标准报告
        results['classification_report'] = classification_report(
            label_sequences,
            pred_sequences,
            scheme=IOB2,
            mode='strict'
        )
        
        # 2. Token级别准确率
        flat_preds = [p for seq in pred_sequences for p in seq]
        flat_labels = [l for seq in label_sequences for l in seq]
        results[f'{metric_key_prefix}_token_accuracy'] = accuracy_score(flat_labels, flat_preds)
        
        # 3. 细粒度实体评估
        for pred_seq, true_seq in zip(pred_sequences, label_sequences):
            self._calculate_entity_metrics(true_seq, pred_seq)
            
        # 计算综合指标
        for mode in ['strict', 'partial', 'type_only']:
            tp = self.metrics[mode]['TP']
            fp = self.metrics[mode]['FP']
            fn = self.metrics[mode]['FN']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.update({
                f'{metric_key_prefix}_{mode}_precision': precision,
                f'{metric_key_prefix}_{mode}_recall': recall,
                f'{metric_key_prefix}_{mode}_f1': f1
            })
            
        print("eval results = \n", results)
        
        return results

# 使用示例
if __name__ == "__main__":
    pass
