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
        default=(project_path / "data/e_commerce.jsonl"),
        type=str
    )

    args = parser.parse_args()
    return args


label_map = DatasetLabels.label_map["ECommerce"]


def main():
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

                    if len(row) == 0:
                        entities = decode.bio_decode(tokens, ner_tags)

                        text = "".join(tokens)

                        entities_ = list()
                        for entity in entities:
                            entity_text = entity["text"]
                            entity_label = entity["entity_label"]
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
                        fout.flush()

                        tokens = list()
                        ner_tags = list()
                    else:
                        splits = row.split(" ")
                        if len(splits) != 2:
                            continue
                        token, tag = splits

                        tokens.append(token)
                        ner_tags.append(tag)

    return


if __name__ == '__main__':
    main()
