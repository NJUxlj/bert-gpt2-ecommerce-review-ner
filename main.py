import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.data.data_preprocess import NERDataProcessor


from src.configs.config import CHINESE_NER_DATA_PATH, BERT_MODEL_PATH, HYBRID_MODEL_PATH, SCHEMA_PATH, NER_DATA_PATH

from transformers import AutoTokenizer
from src.models.enc_dec_model import BertMoEQwen2EncoderDecoder
from src.data.simple_data_preprocess import SimpleNERDataProcessor
from src.evaluation.evaluator import NEREvaluator


def test_ner_dataset():
    from transformers import AutoTokenizer  
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)  
    processor = NERDataProcessor(  
        tokenizer=tokenizer,  
        label_scheme="BIO"  
    )  
    
    dataset = processor.load_and_process(CHINESE_NER_DATA_PATH)  

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





def test_trainer():
    pass




def test_evaluator():

    
    # 初始化组件
    processor = SimpleNERDataProcessor("schema.json")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = BertMoEQwen2EncoderDecoder.from_pretrained("path/to/model")
    
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






if __name__ == "__main__":
    # test_ner_dataset()
    
    test_simple_ner_dataset()