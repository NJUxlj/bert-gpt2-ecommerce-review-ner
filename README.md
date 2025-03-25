# bert-gpt2-ecommerce-review-ner
基于Bert+GPT2改造和LoRA微调的Encoder-Decoder电商评论NER模型


## Project Description

- gpt2模型所在的 `modeling_gpt2.py` 中的 Multi-Head-Attention 改造为 Group-Query-Attention (GQA).
- 在原有的 `modeling_gpt2.py` 的基础上， 在最后又增添了 GPT2ForSequenceClassification, 使其可以用于电商评论任务。
- 将Bert (Encoder) 和 GPT2 (Decoder) 拼接起来，来完整最终的分类任务。
- 使用LoRA对Encoder-Decoder模型在NER任务上进行微调。



## Project Structure
```
bert-gpt2-ecommerce-review-ner
├── README.md
├── src
│   ├── data
         |—— data_preprocess.py
         |—— ner_data
                 |—— dev
                 |—— test
                 |—— train
                 |—— schema.json
│   |── utils
│   ├── configs
│       |——— config.py
│   ├── evaluation
│        ├── evaluator.py 
│   ├── models
|       ├── bert
              |── configuration_bert.py
              |── modeling_bert.py
              |── tokenization_bert.py
|       ├── gpt2
              |── configuration_gpt2.py
              |── modeling_gpt2.py
              |── tokenization_gpt2.py
        |── enc_dec_model.py    # 将bert作为Encoder，gpt2作为Decoder， 组成一个完整的Encoder-Decoder模型
 
│   ├── finetune
│       |—— ner_trainer.py 
|—— main.py # 主训练文件， 用来调用所有在src中封装好的API




## Dataset

- `train` 中的数据示例

```Plain Text
他 O
说 O
: O
中 B-ORGANIZATION
国 I-ORGANIZATION
政 I-ORGANIZATION
府 I-ORGANIZATION
对 O
目 B-TIME
前 I-TIME
南 B-LOCATION
亚 I-LOCATION
出 O
现 O
的 O
核 O
军 O
备 O
竞 O
赛 O
的 O
局 O
势 O


```

- `dev` 和 `test` 中的数据示例
```Plain Text
, O
通 O
过 O
演 O
讲 O
赛 O
、 O
法 O
律 O
知 O
识 O
竞 O
赛 O
, O
举 O
行 O
法 O
律 O
小 O
品 O
晚 O
会 O
等 O
形 O
式 O

```


- schema.json 示例

```Plain Text

{
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

```



- chinese_ner_sft 中的数据示例

```Plain Text
{"text": "雄争霸点卡/七雄争霸元宝/七雄争霸100元1000元宝直充,自动充值", "entities": [{"start_idx": 3, "end_idx": 5, "entity_text": "点卡", "entity_label": "HCCX", "entity_names": ["产品", "商品", "产品名称", "商品名称"]}, {"start_idx": 6, "end_idx": 10, "entity_text": "七雄争霸", "entity_label": "MISC", "entity_names": ["其它实体"]}, {"start_idx": 10, "end_idx": 12, "entity_text": "元宝", "entity_label": "HCCX", "entity_names": ["产品", "商品", "产品名称", "商品名称"]}, {"start_idx": 13, "end_idx": 17, "entity_text": "七雄争霸", "entity_label": "MISC", "entity_names": ["其它实体"]}, {"start_idx": 17, "end_idx": 21, "entity_text": "100元", "entity_label": "MISC", "entity_names": ["其它实体"]}, {"start_idx": 21, "end_idx": 25, "entity_text": "1000", "entity_label": "MISC", "entity_names": ["其它实体"]}, {"start_idx": 25, "end_idx": 27, "entity_text": "元宝", "entity_label": "HCCX", "entity_names": ["产品", "商品", "产品名称", "商品名称"]}], "data_source": "ECommerce"}

```

- chinese_ner_sft 中的实体类型：
```
"ECommerce": {
            # 点卡
            "HCCX": ["产品", "商品", "产品名称", "商品名称"],
            # 地名, 数值, 金额, 重量, 数量.
            "MISC": ["其它实体"],
            "HPPX": ["品牌"],
            "XH": ["型号", "产品型号"],
        },

```

- chinese_ner_sft 中的schema.json
```

{
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


```

## Citation






