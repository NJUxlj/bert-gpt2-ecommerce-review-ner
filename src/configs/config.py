

CHINESE_NER_DATA_PATH = f"./src/data/chinese_ner_sft/data"

PROCESSED_CHINESE_NER_DATA_PATH = "/root/autodl-tmp/bert-gpt2-ecommerce-review-ner/src/data/processed_ner_data"


NER_DATA_PATH = f"./src/data/ner_data"


SCHEMA_PATH = f"./src/data/ner_data/schema.json"


BERT_MODEL_PATH = "/root/autodl-tmp/models/bert-base-chinese"
QWEN2_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-0.5B"



SFT_MODEL_PATH = ""


HYBRID_MODEL_PATH = "./src/output/hybrid_model"


sft_config = {
    "model_path": "path/to/pretrained_model",   # hybrid_model
    "bert_model_name": BERT_MODEL_PATH,
    "max_seq_length": 512,
    "learning_rate": 0.001,
    "batch_size": 10,
    "epochs": 3,
    "max_steps": 10000,
    "num_train_epochs": 2,
    "gradient_accumulation_steps": 2,
    "output_dir": "output",
    "lora_rank":32,
    "ner_data_type": "chinese_ner_sft"


}