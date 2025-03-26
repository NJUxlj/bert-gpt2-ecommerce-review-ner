#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json

from datasets import load_dataset


dataset = load_dataset(
    path="chinese_ner_sft.py",
    # path="qgyd2021/chinese_ner_sft",
    # name="Bank_prompt",
    # name="CCFBDCI_prompt",
    # name="ECommerce_prompt",
    name="FinanceSina_prompt",
    # name="CLUENER_prompt",
    # name="CMeEE_prompt",
    # name="MSRA_prompt",
    # name="WeiBo_prompt",
    # name="YouKu_prompt",
    split="train",
    streaming=True
)

for sample in dataset:
    print(sample["prompt"])
    print(sample["response"])
    # sample = json.dumps(sample, indent=4, ensure_ascii=False)
    # print(sample)
    print("-" * 100)


if __name__ == '__main__':
    pass
