import sys
from pathlib import Path
import argparse
# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.data.data_preprocess import NERDataProcessor

from src.models.bert.configuration_bert import BertConfig
from src.models.qwen2.configuration_qwen2 import Qwen2Config
from src.models.enc_dec_model import NerConfig


from src.configs.config import (
    CHINESE_NER_DATA_PATH, 
    PROCESSED_CHINESE_NER_DATA_PATH,
    BERT_MODEL_PATH, 
    QWEN2_MODEL_PATH,
    HYBRID_MODEL_PATH, 
    SCHEMA_PATH, 
    NER_DATA_PATH
)

from transformers import AutoTokenizer,Trainer
from src.models.enc_dec_model import BertMoEQwen2Encoder
from src.data.simple_data_preprocess import SimpleNERDataProcessor
from src.evaluation.evaluator import NEREvaluator


from src.finetune.ner_trainer import (
    HybridModelTrainer
)


from datasets import load_dataset, Dataset, DatasetDict


def get_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--ner_data_type', type=str, default='chinese_ner_sft', help='chinese_ner_sft or single_ner')
    
    
    return parser.parse_args()



def test_ner_dataset():
    from transformers import AutoTokenizer  
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)  
    processor = NERDataProcessor(  
        tokenizer=tokenizer,  
        label_scheme="BIO"  
    )  
    
    dataset = processor.load_and_process(CHINESE_NER_DATA_PATH)  
    
    print("dataset.keys() = ", dataset.keys())
    print("dataset['train'].features = ", dataset['train'].features)
    print("dataset['train][0] = ", dataset['train'][0])
    
    save_path = Path("./src/data/processed_ner_data")
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_path)  
    
    
    
    



def test_simple_ner_dataset():
    # 初始化处理器  
    processor = SimpleNERDataProcessor(SCHEMA_PATH)  

    # 获取训练数据  
    train_examples = processor.get_train_examples(NER_DATA_PATH)  

    # 使用联合分词器（示例）  
    from transformers import AutoTokenizer  
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)  

    # 转换为模型输入  
    sample = train_examples[0]  
    features = processor.align_labels(tokenizer, sample)  
    
    print("features = ", features)





def test_trainer(ner_data_type = "chinese_ner_sft"):
    swan_config = {  
        "project": "NER-Experiment",  
        "name": "Bert-MoE-Qwen2-LoRA",  
        "config": {  
            "lora_rank": 32,  
            "batch_size": 16,  
            "learning_rate": 3e-4  
        }  
    }  
    
    bert_config = BertConfig.from_pretrained(BERT_MODEL_PATH)
    qwen2_config = Qwen2Config.from_pretrained(QWEN2_MODEL_PATH)
    ner_config = NerConfig()
    
    trainer = HybridModelTrainer(
        config_path =None, 
        swan_config = swan_config,
        bert_config=bert_config,
        qwen2_config=qwen2_config,
        ner_config=ner_config,
    )
    
    tokenizer = trainer.tokenizer
    
    if ner_data_type == 'single_ner':
        processor = SimpleNERDataProcessor("schema.json")
    
        # 加载数据集
        train_dataset = processor.get_train_examples("data/train.txt")
        eval_dataset = processor.get_dev_examples("data/dev.txt")
        
        # 转换数据集格式
        train_dataset = Dataset.from_dict({"features": [processor.align_labels(trainer.tokenizer, e) for e in train_dataset]})
        eval_dataset = Dataset.from_dict({"features": [processor.align_labels(trainer.tokenizer, e) for e in eval_dataset]})
    else:
        print("load Chinese NER data")
        processor:NERDataProcessor = trainer.processor
        train_dataset = processor.load_hf_data(PROCESSED_CHINESE_NER_DATA_PATH, "train")
        eval_dataset = processor.load_hf_data(PROCESSED_CHINESE_NER_DATA_PATH, "validation")
        
    
    

    # 开始训练
    trainer.train(train_dataset, eval_dataset)
    
    trainer:HybridModelTrainer
    




def test_evaluator():

    
    # 初始化组件
    processor = SimpleNERDataProcessor(SCHEMA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = BertMoEQwen2EncoderDecoder(BertConfig(), Qwen2Config(), NerConfig())
    
    # 加载测试数据
    test_examples = processor.get_test_examples("data/test.txt")
    
    # 实例化评估器
    evaluator = NEREvaluator(model, processor, tokenizer)
    
    # 运行评估
    results = evaluator.evaluate(test_examples)
    
    print("===== 分类报告 =====")
    print(results['classification_report'])
    
    print("\n===== 综合指标 =====")
    print(f"Token准确率: {results['token_accuracy']:.4f}")
    print(f"严格F1: {results['strict_f1']:.4f}")
    print(f"部分匹配F1: {results['partial_f1']:.4f}")
    print(f"仅类型F1: {results['type_only_f1']:.4f}")




def main():
    args = get_args()
    
    

if __name__ == "__main__":
    # test_ner_dataset()
    
    test_trainer()