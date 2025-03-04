#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json

from project_settings import project_path
from chinese_ner_sft import DatasetLabels
import decode




def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--output_file",
        default = (project_path / "data/e_commerce.jsonl"),
        type=str
    )

    args = parser.parse_args()
    return args



label_map = DatasetLabels.label_map["ECommerce"]





def main():
    '''
     调用了一个 bio_decode 方法，将 tokens 和 ner_tags 转换为实体列表。
     
    tokens 是当前句子的单词列表。
        ner_tags 是与 tokens 一一对应的NER标签列表。
        bio_decode 方法的作用是解析 BIO 格式的标签（如 B-PER, I-PER, O），提取实体信息（如实体的起始位置、结束位置、实体文本和标签）。
    
    BIO格式解释：
        B-<label> 表示实体的开始（Begin）。
        I-<label> 表示实体的中间部分（Inside）。
        O 表示非实体（Outside）。
    
    '''
    args = get_args()

    # download
    # https://github.com/allanj/ner_incomplete_annotation/tree/master

    filename_list = [
        "ner_incomplete_annotation-master/data/ecommerce/train.txt",
        "ner_incomplete_annotation-master/data/ecommerce/dev.txt",
        "ner_incomplete_annotation-master/data/ecommerce/test.txt",
    ]
    tokens = list()
    ner_tags = list()
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for filename in filename_list:
            with open(filename, "r", encoding="utf-8", errors="ignore") as fin:
                for row in fin:
                    row = str(row).strip()

                    if len(row) == 0: # 如果当前行是空行（len(row) == 0），说明一个句子已经结束（NER数据通常用空行分隔句子）。
                        entities = decode.bio_decode(tokens, ner_tags)

                        text = "".join(tokens)

                        entities_ = list()
                        for entity in entities:
                            entity_text = entity["text"] # 实体的文本内容。
                            entity_label = entity["entity_label"] # 实体的标签（如 PER, LOC, ORG 等）。
                            if entity_label not in label_map.keys():
                                print(text)
                                print(entity_text)
                                print(entity_label)

                                exit(0)

                            entities_.append({
                                "start_idx": entity["start_idx"],
                                "end_idx": entity["end_idx"],
                                "entity_text": entity["text"],
                                "entity_label": entity_label, 
                                "entity_names": label_map.get(entity_label),
                            })

                        row = {
                            "text": text,
                            "entities": entities_,
                            "data_source": "ECommerce"
                        }
                        row = json.dumps(row, ensure_ascii=False)
                        fout.write("{}\n".format(row))
                        fout.flush() #  确保数据立即写入文件，而不是留在缓冲区中。

                        tokens = list() # 处理完一个句子后，重置 tokens 和 ner_tags，以便开始处理下一个句子
                        ner_tags = list()
                    else:
                        '''
                        使用 split(" ") 将行按照空格分割为两个部分：token（单词）和 tag（NER标签）。
                        如果分割后的长度不是2（例如行格式不正确），跳过该行。
                        将 token 添加到 tokens 列表，将 tag 添加到 ner_tags 列表。
                        '''     
                        splits = row.split(" ")
                        if len(splits) != 2:
                            continue
                        token, tag = splits

                        tokens.append(token)
                        ner_tags.append(tag)

    return












if __name__ == "__main__":
    main()