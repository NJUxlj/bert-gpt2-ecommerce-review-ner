import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.data.data_preprocess import NERDataProcessor


from src.configs.config import CHINESE_NER_DATA_PATH, BERT_MODEL_PATH



def test_ner_dataset():
    from transformers import AutoTokenizer  
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)  
    processor = NERDataProcessor(  
        tokenizer=tokenizer,  
        label_scheme="BIO"  
    )  
    
    dataset = processor.load_and_process(CHINESE_NER_DATA_PATH)  
    dataset.save_to_disk(".src/data/processed_ner_data")  
    
    
    
    



def test_simple_ner_dataset():
    pass







if __name__ == "__main__":
    test_ner_dataset()