
"""
基于混合模型的LoRA微调训练器，集成DeepSpeed优化
特点：
1. 分层LoRA应用：对BERT和Qwen2的不同层应用差异化的LoRA配置
2. MoE参数冻结：保持专家网络参数固定以节省显存
3. 动态梯度累积：根据batch大小自动调整累积步数
"""

import os
import json
import torch
import deepspeed
import numpy as np
import swanlab
from transformers import TrainerCallback 
from peft import LoraConfig, get_peft_model
from datasets import Dataset


from typing import (
    List, 
    Dict,
    Tuple,
    Optional,
    Literal,
    Union,
    Any,
)

from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    
    AutoTokenizer,
)


from src.models.enc_dec_model import BertMoEQwen2EncoderDecoder

from src.models.bert.configuration_bert import BertConfig
from src.models.qwen2.configuration_qwen2 import Qwen2Config
from src.models.enc_dec_model import NerConfig

from src.data.simple_data_preprocess import SimpleNERDataProcessor
from src.data.data_preprocess import NERDataProcessor

from src.evaluation.evaluator import NEREvaluator

from src.configs.config import (
    SFT_MODEL_PATH,
    SCHEMA_PATH,
    HYBRID_MODEL_PATH,
    BERT_MODEL_PATH,
    QWEN2_MODEL_PATH,
)

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)



class SwanLabCallback(TrainerCallback):  
    def __init__(self, swan_config:Dict):  
        self.swan_run = swanlab.init(  
            project=swan_config.get("project", "NER-Experiment"),
            config=swan_config,  
            experiment_name=swan_config.get("name", "ner_experiment")  
        )  
        
    def on_log(self, args, state, control, logs=None, **kwargs):  
        """实时记录训练指标到SwanLab"""  
        if logs:  
            metrics = {  
                f"train/{k}": v for k,v in logs.items()   
                if not k.startswith("_")  
            }  
            self.swan_run.log(metrics)  
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  
        """记录评估指标"""  
        if metrics:  
            eval_metrics = {  
                f"eval/{k}": v for k,v in metrics.items()  
                if not k.startswith("_")  
            }  
            self.swan_run.log(eval_metrics) 
            
            

class EnhancedTrainer(Trainer):  
    def __init__(self, evaluator, swan_config, **kwargs):  
        super().__init__(**kwargs)  
        self.evaluator: NEREvaluator = evaluator  
        self.swan_callback = SwanLabCallback(swan_config)  
        self.add_callback(self.swan_callback)  
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):  
        """使用自定义评估器进行验证"""  
        # 执行父类评估获取基础指标  
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)  
        
        # 执行扩展评估  
        extended_metrics = self.evaluator.evaluate(eval_dataset)  
        
        # 合并指标  
        eval_results.update({  
            "strict_f1": extended_metrics["strict_f1"],  
            "partial_f1": extended_metrics["partial_f1"],  
            "token_acc": extended_metrics["token_accuracy"]  
        })  
        
        return eval_results 




           
        

class HybridModelTrainer:
    def __init__(
        self, 
        config_path: str =None, 
        ner_data_type:Literal["chinese_ner_sft", "simple_ner"] = "chinese_ner_sft",
        swan_config = None,
        bert_config: BertConfig = None,
        qwen2_config: Qwen2Config = None,
        ner_config: NerConfig = None
        ):
        
        self.bert_config = bert_config or BertConfig.from_pretrained(BERT_MODEL_PATH)
        self.qwen2_config = qwen2_config or Qwen2Config.from_pretrained(QWEN2_MODEL_PATH)
        self.ner_config = ner_config

        if config_path == None:
            self.config = {
                            "model_path": "path/to/pretrained_model",
                            "bert_model_name": BERT_MODEL_PATH,
                            "max_seq_length": 512,
                            "batch_size": 16,
                            "learning_rate": 3e-4,
                            "max_steps": 10000,
                            "output_dir": "./output",
                            "lora_rank": 8,
                            "ner_data_type": "chinese_ner_sft",
                            'num_train_epochs': 2,
                        }
        else:
            with open(config_path) as f:
                self.config = json.load(f)
        
        # 初始化模型和分词器
        self.model, self.tokenizer = self._initialize_model()
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding='longest',
            max_length=512,
            label_pad_token_id=-100
        )
    
    
        self.swan_config = {  
            "project": "NER-Experiment",  
            "name": f"Bert-MoE-Qwen2-LoRA",  
            "config": {  
                "lora_rank": self.config['lora_rank'],  
                "batch_size": self.config['batch_size'],  
                "learning_rate": self.config['learning_rate']  
            }  
        }  if not swan_config else swan_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        
        if ner_data_type == "chinese_ner_sft":
            self.processor = NERDataProcessor(tokenizer = self.tokenizer)
        else:
            self.processor = SimpleNERDataProcessor(SCHEMA_PATH)
        
        
    def _initialize_model(self):
        """加载混合模型并应用LoRA"""
        # 加载基础模型
        model = BertMoEQwen2EncoderDecoder(
            bert_config=self.bert_config,
            qwen_config=self.qwen2_config,
            ner_config=self.ner_config
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config['bert_model_name'])
        
        # 冻结MoE参数
        for param in model.moe.parameters():
            param.requires_grad = False
            
        # 分层LoRA配置
        lora_config = LoraConfig(
            r=self.config['lora_rank'],
            lora_alpha=32,
            target_modules=[
                # "encoder.layer.*.attention",  # BERT的注意力层
                # "decoder.layers.*.self_attn",  # Qwen2的注意力层
                # "decoder.layers.*.cross_attn"  # 跨注意力层（如果有）
                "q_proj","k_proj", "v_proj"
                # "query", "key", "value"
            ],
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["classifier"],  # 分类头保持可训练
            layers_to_transform=list(range(8,12))  # 仅微调深层
        )
        
        return get_peft_model(model, lora_config), tokenizer
    
    def _ds_config(self):
        """生成动态DeepSpeed配置"""
        return {
            "train_micro_batch_size_per_gpu": self.config['batch_size'],
            "gradient_accumulation_steps": "auto",
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "weight_decay": "auto",
                    "torch_adam": True
                }
            },
            "scheduler": {
                "type": "WarmupDecay",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config['learning_rate'],
                    "warmup_num_steps": 500,
                    "total_num_steps": self.config['max_steps']
                }
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "contiguous_gradients": True,
                "overlap_comm": True,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights": True
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
                "number_checkpoints": 4
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16
            }
        }
    
    def train(self, train_dataset, eval_dataset):
        print("train_dataset[0].keys() = ", train_dataset[0].keys())
        print("train_dataset[0] = ", train_dataset[0])
        print("train_dataset[1] = ", train_dataset[0])
        # 初始化评估器  
        evaluator = NEREvaluator(  
            model=self.model,  
            processor=self.processor,  
            tokenizer=self.tokenizer  
        )  
        # 初始化训练参数
        training_args = TrainingArguments(
            num_train_epochs=self.config['num_train_epochs'],
            output_dir=self.config['output_dir'],
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            logging_steps=100,
            max_steps=self.config['max_steps'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config.get('gradient_accumulation', 2),
            learning_rate=self.config['learning_rate'],
            warmup_ratio=0.1,
            weight_decay=0.01,
            deepspeed=self._ds_config(),
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="strict_f1",
            greater_is_better=True,
            remove_unused_columns=False,
            fp16=True,
            warmup_steps=500,
        )
        
        # 创建增强版Trainer
        trainer = EnhancedTrainer(
            evaluator=evaluator,
            swan_config=self.swan_config,  
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._compute_basic_metrics,  
            callbacks=[  
                EarlyStoppingCallback(early_stopping_patience=3),  
            ]  
        )
        
        # 开始训练
        trainer.train()
        trainer.save_model(os.path.join(self.config['output_dir'], "hybrid_model"))
        
        return trainer
    
    def _compute_moe_metrics(self, eval_pred):
        """支持MoE平衡损失的评估指标计算"""
        predictions, labels = eval_pred  # predictions.shape = (batch_size, seq_len, num_labels)
        predictions = np.argmax(predictions, axis=2)
        
        # 移除填充标签， 过滤填充标签(-100)，同时转换id到标签名称
        true_predictions = [
            [self.config['id2tag'][p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.config['id2tag'][l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        
        # 展平嵌套结构（需要添加此步骤）
        flat_predictions = [p for sublist in true_predictions for p in sublist]
        flat_labels = [l for sublist in true_labels for l in sublist]

        return {
            "precision": precision_score(true_labels, true_predictions, average='micro'),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }
        
        
    def _compute_basic_metrics(self, eval_pred):  
        """供Trainer使用的基础指标计算"""  
        predictions, labels = eval_pred  
        predictions = np.argmax(predictions, axis=2)  
        
        # 计算准确率  
        active_mask = labels != -100  
        active_labels = labels[active_mask]  
        active_preds = predictions[active_mask]  
        acc = accuracy_score(active_labels, active_preds)  
        
        return {"accuracy": acc}  

# 使用示例
if __name__ == "__main__":

    
    # 启动SwanLab监控  
    swan_config = {  
        "project": "NER-Experiment",  
        "name": "Bert-MoE-Qwen2-LoRA",  
        "config": {  
            "lora_rank": 8,  
            "batch_size": 16,  
            "learning_rate": 3e-4  
        }  
    }  
    
    
    trainer = HybridModelTrainer(config_path =None, swan_config = swan_config)
    processor = SimpleNERDataProcessor("schema.json")
    
    # 加载数据集
    train_dataset = processor.get_train_examples("data/train.txt")
    eval_dataset = processor.get_dev_examples("data/dev.txt")
    
    # 转换数据集格式
    train_dataset = Dataset.from_dict({"features": [processor.align_labels(trainer.tokenizer, e) for e in train_dataset]})
    eval_dataset = Dataset.from_dict({"features": [processor.align_labels(trainer.tokenizer, e) for e in eval_dataset]})
    
    # 开始训练
    trainer.train(train_dataset, eval_dataset)
