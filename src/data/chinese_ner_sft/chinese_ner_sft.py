#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from glob import glob
import json
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple

import datasets


_entity_urls = {
    "Bank": "data/bank.jsonl",
    "CCFBDCI": "data/ccfbdci.jsonl",
    "ccks2019_task1": "data/ccks2019_task1.jsonl",
    "CLUENER": "data/cluener.jsonl",
    "CMeEE": "data/cmeee.jsonl",
    "DLNER": "data/dlner.jsonl",
    "ECommerce": "data/e_commerce.jsonl",
    "FinanceSina": "data/finance_sina.jsonl",
    "MSRA": "data/msra.jsonl",
    "NLPCC2018_task4": "data/nlpcc_2018_task4.jsonl",
    "WeiBo": "data/weibo.jsonl",
    "YouKu": "data/youku.jsonl",

}

_template_urls = {
    "Bank_template": "data/bank_template.txt",
    "CCFBDCI_template": "data/ccfbdci_template.txt",
    "ccks2019_task1_template": "data/ccks2019_task1_template.txt",
    "CLUENER_template": "data/cluener_template.txt",
    "CMeEE_template": "data/cmeee_template.txt",
    "DLNER_template": "data/dlner_template.txt",
    "ECommerce_template": "data/e_commerce_template.txt",
    "FinanceSina_template": "data/finance_sina_template.txt",
    "MSRA_template": "data/msra_template.txt",
    "NLPCC2018_task4_template": "data/nlpcc_2018_task4_template.txt",
    "WeiBo_template": "data/weibo_template.txt",
    "YouKu_template": "data/youku_template.txt",

}

_prompt_urls = {
    "Bank_prompt": None,
    "CCFBDCI_prompt": None,
    "ccks2019_task1_prompt": None,
    "CLUENER_prompt": None,
    "CMeEE_prompt": None,
    "DLNER_prompt": None,
    "ECommerce_prompt": None,
    "FinanceSina_prompt": None,
    "MSRA_prompt": None,
    "NLPCC2018_task4_prompt": None,
    "WeiBo_prompt": None,
    "YouKu_prompt": None,

}

_CITATION = """
@dataset{chinese_ner_sft,
  author       = {Xing Tian},
  title        = {chinese_ner_sft},
  month        = sep,
  year         = 2023,
  publisher    = {Xing Tian},
  version      = {1.0},
}
"""


class DatasetLabels(object):
    label_map = {
        "Bank": {
            "BANK": ["银行", "银行名称"],
            "COMMENTS_N": ["金融名词"],
            "COMMENTS_ADJ": ["形容词"],
            "PRODUCT": ["产品", "产品名称", "金融名词", "金融产品", "银行产品"],
        },
        "ccks2019_task1": {
            "疾病和诊断": ["疾病和诊断", "疾病或诊断"],
            "手术": ["手术"],
            "解剖部位": ["解剖部位", "身体部位", "身体位置", "人体部位"],
            "药物": ["药物", "药品", "医药", "药剂"],
            "影像检查": ["影像检查"],
            "实验室检验": ["实验室检验"]
        },
        "CLUENER": {
            "game": ["游戏"],
            "organization": ["组织", "团体", "机构", "组织机构"],
            "government": ["政府"],
            "movie": ["电影", "影片"],
            "book": ["书籍", "书名", "书本"],
            "company": ["公司", "企业", "厂商"],
            "address": ["地址", "地点"],
            "name": ["人名", "姓名"],
            "position": ["职位", "地位", "职务", "身分", "身份"],
            "scene": ["场景", "景象", "景色", "景点"]
        },
        "CMeEE": {
            "pro": [
                "医疗程序", "医疗过程",
                # "medical procedure",
            ],
            "dis": [
                "疾病", "病名", "病症",
                # "disease",
            ],
            "dru": [
                "药物", "药品", "医药", "药剂",
                # "drug", "medicine",
            ],
            "bod": [
                "身体部位", "身体位置", "人体部位",
                # "body",
            ],
            "sym": [
                "症状", "发病症状", "临床表现",
                # "symptom", "symptoms",
            ],
            "mic": [
                "致病因子",
                # "microbe"
            ],
            "equ": [
                "医疗设备", "医疗装置", "医疗器材", "医疗器械",
                # "equipment"
            ],
            "ite": [
                "诊断项目", "检查项目",
                # "item"
            ],
            "dep": [
                "科室",
                # "department"
            ]
        },
        "DLNER": {
            "Time": ["时间"],
            "Person": ["姓名或人称代词", "人名或人称代词"],
            "Thing": ["物品", "物品名词"],
            "Location": ["地点", "地址", "地点名词", "地址名词", "地点国家或城市", "地址国家或城市"],
            "Metric": ["数量", "量词", "数词"],
            "organization": ["组织", "团体", "机构", "组织机构"],
            "Abstract": ["抽象名词", "概念名词"],
            # "Physical": ["物理名词"],
            # "Term": ["量词"],
        },
        "ECommerce": {
            # 点卡
            "HCCX": ["产品", "商品", "产品名称", "商品名称"],
            # 地名, 数值, 金额, 重量, 数量.
            "MISC": ["其它实体"],
            "HPPX": ["品牌"],
            "XH": ["型号", "产品型号"],
        },
        "FinanceSina": {
            "PER": ["人名", "姓名"],
            "ORG": ["组织", "团体", "机构", "组织机构"],
            "GPE": ["地缘政治实体", "政治实体", "地理实体", "社会实体"],
            "LOC": ["地址", "地点", "地名"],
        },
        "MSRA": {
            "PER": ["人名", "姓名"],
            "LOC": ["地址", "地点", "地名"],
            "ORG": ["组织", "团体", "机构", "组织机构"],
        },
        "NLPCC2018_task4": {
            "singer": ["歌手", "演唱者"],
            "song": ["歌曲", "曲子", "音乐名", "曲名"],
            "theme": ["主题", "主旋律", "音乐主题"],
            "emotion": ["情感", "情绪"],
            "style": ["音乐风格", "风格", "曲风"],
            "destination": ["目的地", "地址", "地点"],
            "phone_num": ["电话号码", "手机号码"],
            "instrument": ["乐器", "乐器名称"],
            "contact_name": ["联系人姓名", "联系人"],
            "age": ["年龄", "时期", "时代", "年代"],
            "toplist": ["热门列表", "热歌榜", "热门榜单", "流行榜单"],
            "custom_destination": ["自定义目的地", "目的地", "地址", "地点"],
            "language": ["语言", "语种"],
            "scene": ["场景", "情景"],
            "origin": ["出发地", "出发地点", "始发地", "始发地点", "地址", "地点"],
        },
        "Resume": {
            "NAME": ["人名", "姓名"],
            "CONT": ["国籍"],
            "RACE": ["民族", "种族"],
            "TITLE": ["职称", "职位", "职衔"],
            "EDU": ["学历"],
            "PRO": ["专业"],
            "ORG": ["组织", "团体", "机构", "组织机构"],
            "LOC": ["地址", "地点", "地名"],

        },
        "CCFBDCI": {
            "PER": ["人名", "姓名"],
            "LOC": ["地址", "地点", "地名"],
            "ORG": ["组织", "团体", "机构", "组织机构"],
            "GPE": ["地缘政治实体", "政治实体", "地理实体", "社会实体"]
        },
        "WeiBo": {
            "GPE.NAM": ["地缘政治实体", "政治实体", "地理实体", "社会实体"],
            "LOC.NAM": ["地址", "地点", "地名"],
            "LOC.NOM": ["地址代词", "地点代词", "地名代词"],
            "ORG.NAM": ["组织", "团体", "机构", "组织机构"],
            "ORG.NOM": ["组织代词", "团体代词", "机构代词"],
            "PER.NAM": ["人名", "姓名"],
            "PER.NOM": ["人称代词"],
        },
        "YouKu": {
            "TELEVISION": ["电视", "电视节目", "影视作品", "影视节目"],
            # 地名, 数值, 金额, 重量, 数量.
            "MISC": ["其它实体"],
            "PER": ["人名", "姓名"],

        }
    }

    @classmethod
    def sample_candidate_entity_names(cls,
                                      dataset_name: str,
                                      text_entity_labels: List[str],
                                      must_entity_label_to_name: Dict[str, str]
                                      ) -> Dict[str, str]:
        full_entity_label_to_name = cls.label_map[dataset_name]
        sampled_entity_labels: List[str] = random.sample(
            full_entity_label_to_name.keys(),
            k=random.randint(1, len(full_entity_label_to_name.keys()))
        )
        sample_entity_label_to_name: Dict[str, str] = dict()
        for entity_label in sampled_entity_labels:
            sample_entity_label_to_name[entity_label] = random.sample(full_entity_label_to_name[entity_label], k=1)[0]

        sample_entity_label_to_name.update(must_entity_label_to_name)

        for entity_label in text_entity_labels:
            if entity_label in sample_entity_label_to_name.keys():
                if entity_label not in must_entity_label_to_name.keys():
                    sample_entity_label_to_name.pop(entity_label)
        return sample_entity_label_to_name


class Sample(object):
    def __init__(self, sample: dict):
        self.sample = sample
        self.text = sample["text"]
        self.entities = sample["entities"]

        label_to_name = defaultdict(set)
        for entity in self.entities:
            label_to_name[entity["entity_label"]].update(entity["entity_names"])
        self.entity_label_to_names = label_to_name

    def entity_labels(self) -> List[str]:
        return list(self.entity_label_to_names.keys())

    def sample_entity_label_to_name(self, sample_all: bool = False) -> Dict[str, str]:
        if len(self.entity_labels()) == 0:
            entity_labels = list()
        elif sample_all:
            entity_labels = self.entity_labels()
        else:
            entity_labels = random.sample(
                self.entity_label_to_names.keys(),
                k=random.randint(1, len(self.entity_labels()))
            )
        entity_label_to_name: Dict[str, str] = dict()
        for entity_label in entity_labels:
            entity_names = self.entity_label_to_names[entity_label]
            entity_label_to_name[entity_label] = random.sample(entity_names, k=1)[0]
        return entity_label_to_name

    def sample_entity(self, sample_all: bool = False) -> List[dict]:
        entity_label_to_name = self.sample_entity_label_to_name(sample_all)
        sample_entity = list()
        for entity in self.entities:
            if entity["entity_label"] not in entity_label_to_name.keys():
                continue
            sample_entity.append({
                "entity_text": entity["entity_text"],
                "entity_label": entity["entity_label"],
                "entity_name": entity_label_to_name[entity["entity_label"]],
            })
        return sample_entity


class PromptTemplate(object):
    def __init__(self,
                 template: str,
                 text: str,
                 candidate_entity_names: Dict[str, str],
                 entities: List[dict],
                 **kwargs
                 ):
        self.template = template
        self.text = text
        self.candidate_entity_names = candidate_entity_names
        self.entities = entities
        self.kwargs = kwargs

        pattern = r"\{([a-z_]*?)\}"
        input_variables = re.findall(pattern, self.template, flags=re.IGNORECASE)

        self.input_variables = input_variables

    def __str__(self):
        template_kwargs = dict()
        if "candidate_entity_names" in self.input_variables:
            candidate_entity_names_sep = self.kwargs["candidate_entity_names_sep"]
            template_kwargs["candidate_entity_names"] = candidate_entity_names_sep.join(self.candidate_entity_names.values())
        if "text" in self.input_variables:
            template_kwargs["text"] = self.text
        prompt = self.template.format(**template_kwargs)
        return prompt


class ResponseTemplate(object):
    def __init__(self,
                 template: str,
                 text: str,
                 candidate_entity_names: Dict[str, str],
                 entities: List[dict],
                 **kwargs):
        self.template = template
        self.text = text
        self.candidate_entity_names = candidate_entity_names
        self.entities = entities
        self.kwargs = kwargs

        pattern = r"\{([a-z_]*?)\}"
        input_variables = re.findall(pattern, self.template, flags=re.IGNORECASE)

        self.input_variables = input_variables

        self.entity_name_to_entities = self.get_entity_name_to_entities()

    def get_entity_name_to_entities(self):
        result = defaultdict(list)
        for entity in self.entities:
            result[entity["entity_name"]].append(entity)
        return result

    def get_entity_text_list(self, entities: List[dict] = None, drop_duplicates: bool = False) -> str:
        entity_text_sep = self.kwargs["entity_text_sep"]
        entities = entities or self.entities

        entity_text_list = [e["entity_text"] for e in entities]
        if drop_duplicates:
            entity_text_list = list(set(entity_text_list))

        entity_text_list = entity_text_sep.join(entity_text_list)
        return entity_text_list

    def get_no_entity_response(self):
        template = self.kwargs["no_entity_template"]
        pattern = r"\{([a-z_]*?)\}"
        input_variables = re.findall(pattern, template, flags=re.IGNORECASE)
        template_kwargs = dict()
        if "candidate_entity_names" in input_variables:
            candidate_entity_names_sep = self.kwargs["candidate_entity_names_sep"]
            template_kwargs["candidate_entity_names"] = candidate_entity_names_sep.join(self.candidate_entity_names.values())
        if "text" in input_variables:
            template_kwargs["text"] = self.text

        response = template.format(**template_kwargs)
        return response

    @staticmethod
    def format_entity_template(template: str, entity: dict):
        pattern = r"\{([a-z_]*?)\}"
        input_variables = re.findall(pattern, template, flags=re.IGNORECASE)

        template_kwargs = dict()
        if "entity_text" in input_variables:
            template_kwargs["entity_text"] = entity["entity_text"]
        if "entity_name" in input_variables:
            template_kwargs["entity_name"] = entity["entity_name"]

        entity_ = template.format(**template_kwargs)
        return entity_

    @staticmethod
    def format_entity_name_to_text_template(template: str, entity_name: str, entity_text_list: str):
        pattern = r"\{([a-z_]*?)\}"
        input_variables = re.findall(pattern, template, flags=re.IGNORECASE)

        template_kwargs = {
            "entity_name": entity_name,
            "entity_text_list": entity_text_list,
        }

        entity_ = template.format(**template_kwargs)
        return entity_

    def __str__(self):
        template_kwargs = dict()
        if "entity_text_list" in self.input_variables:
            entity_text_list = self.get_entity_text_list()
            template_kwargs["entity_text_list"] = entity_text_list
        if "entity_list" in self.input_variables:
            entity_list_sep = self.kwargs["entity_list_sep"]
            entity_template = self.kwargs["entity_template"]
            entity_list = [
                self.format_entity_template(entity_template, entity)
                for entity in self.entities
            ]
            entity_list = entity_list_sep.join(entity_list)
            template_kwargs["entity_list"] = entity_list
        if "entity_set" in self.input_variables:
            entity_list_sep = self.kwargs["entity_list_sep"]
            entity_template = self.kwargs["entity_template"]
            entity_list = [
                self.format_entity_template(entity_template, entity)
                for entity in self.entities
            ]
            entity_set = entity_list_sep.join(list(set(entity_list)))
            template_kwargs["entity_set"] = entity_set
        if "entity_name_to_text_list" in self.input_variables:
            entity_name_to_text_list_sep = self.kwargs["entity_name_to_text_list_sep"]
            entity_template = self.kwargs["entity_template"]
            entity_formatted_list = list()
            for entity_name, entities in self.entity_name_to_entities.items():
                entity_text_list = self.get_entity_text_list(entities)
                entity_formatted = self.format_entity_name_to_text_template(
                    entity_template, entity_name, entity_text_list)
                entity_formatted_list.append(entity_formatted)
            entity_name_to_text_list = entity_name_to_text_list_sep.join(entity_formatted_list)
            template_kwargs["entity_name_to_text_list"] = entity_name_to_text_list
        if "entity_name_to_text_set" in self.input_variables:
            entity_name_to_text_list_sep = self.kwargs["entity_name_to_text_set_sep"]
            entity_template = self.kwargs["entity_template"]
            entity_formatted_list = list()
            for entity_name, entities in self.entity_name_to_entities.items():
                entity_text_list = self.get_entity_text_list(entities, drop_duplicates=True)
                entity_formatted = self.format_entity_name_to_text_template(
                    entity_template, entity_name, entity_text_list)
                entity_formatted_list.append(entity_formatted)
            entity_name_to_text_list = entity_name_to_text_list_sep.join(entity_formatted_list)
            template_kwargs["entity_name_to_text_set"] = entity_name_to_text_list

        response = self.template.format(**template_kwargs)
        if len(str(response).strip()) == 0:
            response = self.get_no_entity_response()
        return response


class PromptResponseTemplate(object):
    def __init__(self, dataset_name: str, templates: List[dict], sample: dict):
        self.dataset_name = dataset_name
        self.templates = templates
        self.sample = sample

        self.template: dict = self.sample_template()

    def sample_template(self) -> dict:
        template = random.sample(self.templates, k=1)[0]
        return template

    def get_prompt_response(self) -> Tuple[str, str]:
        # kwargs
        kwargs = self.template["kwargs"]
        kwargs = json.loads(kwargs)

        # sample_all
        sample_all = kwargs.get("sample_all", False)

        sample = Sample(self.sample)
        sampled_entity: List[dict] = sample.sample_entity(sample_all=sample_all)

        sampled_entity_label_to_name = {
            e["entity_label"]: e["entity_name"]
            for e in sampled_entity
        }

        candidate_entity_names = DatasetLabels.sample_candidate_entity_names(
            dataset_name=self.dataset_name,
            text_entity_labels=sample.entity_labels(),
            must_entity_label_to_name=sampled_entity_label_to_name
        )

        # prompt_template
        prompt_template: str = self.template["prompt_template"]
        prompt_template = PromptTemplate(
            prompt_template,
            sample.text,
            candidate_entity_names,
            sampled_entity,
            **kwargs
        )
        prompt = str(prompt_template)

        # response_template
        response_template: str = self.template["response_template"]
        response_template = ResponseTemplate(
            response_template,
            sample.text,
            candidate_entity_names,
            sampled_entity,
            **kwargs
        )
        response = str(response_template)
        return prompt, response


class ChineseNerSFT(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    entity_configs = list()
    for name in _entity_urls.keys():
        config = datasets.BuilderConfig(name=name, version=VERSION, description=name)
        entity_configs.append(config)

    template_configs = list()
    for name in _template_urls.keys():
        config = datasets.BuilderConfig(name=name, version=VERSION, description=name)
        template_configs.append(config)

    prompt_configs = list()
    for name in _prompt_urls.keys():
        config = datasets.BuilderConfig(name=name, version=VERSION, description=name)
        prompt_configs.append(config)

    BUILDER_CONFIGS = [
        *entity_configs,
        *template_configs,
        *prompt_configs,
    ]

    def _info(self):
        if self.config.name in _entity_urls.keys():
            features = datasets.Features({
                "text": datasets.Value("string"),
                "entities": datasets.Sequence(
                    feature=datasets.Features({
                        "start_idx": datasets.Value("int64"),
                        "end_idx": datasets.Value("int64"),
                        "entity_text": datasets.Value("string"),
                        "entity_label": datasets.Value("string"),
                        "entity_names": datasets.Sequence(
                            datasets.Value("string")
                        )
                    })
                ),
                "data_source": datasets.Value("string"),
            })
        elif self.config.name in _template_urls.keys():
            features = datasets.Features({
                "prompt_template": datasets.Value("string"),
                "response_template": datasets.Value("string"),
                "kwargs": datasets.Value("string"),
            })
        elif self.config.name in _prompt_urls.keys():
            features = datasets.Features({
                "prompt": datasets.Value("string"),
                "response": datasets.Value("string"),
            })
        else:
            raise NotImplementedError

        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
            homepage="",
            license="",
            citation=_CITATION,
        )

    def _split_entity_generators(self, dl_manager):
        url = _entity_urls.get(self.config.name)
        dl_path = dl_manager.download_and_extract(url)
        archive_path = dl_path

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": archive_path, "split": "train"},
            ),
        ]

    def _split_template_generators(self, dl_manager):
        url = _template_urls.get(self.config.name)
        dl_path = dl_manager.download_and_extract(url)
        archive_path = dl_path

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": archive_path, "split": "train"},
            ),
        ]

    def _split_prompt_generators(self, dl_manager):
        dataset_name = self.config.name[:-7]
        entity_url = _entity_urls.get(dataset_name)
        entity_dl_path = dl_manager.download(entity_url)

        template_name = "{}_template".format(dataset_name)
        template_url = _template_urls.get(template_name)
        template_dl_path = dl_manager.download(template_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"entity_dl_path": entity_dl_path, "template_dl_path": template_dl_path, "split": "train"},
            ),
        ]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name in _entity_urls.keys():
            return self._split_entity_generators(dl_manager)
        elif self.config.name in _template_urls.keys():
            return self._split_template_generators(dl_manager)
        elif self.config.name in _prompt_urls.keys():
            return self._split_prompt_generators(dl_manager)
        else:
            raise NotImplementedError

    def _generate_entity(self, archive_path, split):
        """Yields examples."""
        archive_path = Path(archive_path)

        idx = 0
        with open(archive_path, "r", encoding="utf-8") as f:
            for row in f:
                row = json.loads(row)

                yield idx, {
                    "text": row["text"],
                    "entities": row["entities"],
                    "data_source": row["data_source"],
                }
                idx += 1

    def _generate_template(self, archive_path, split):
        archive_path = Path(archive_path)

        idx = 0
        prompt_template = ""
        response_template = ""
        kwargs = ""

        with open(archive_path, "r", encoding="utf-8") as f:
            flag = None
            for row in f:
                row = str(row).strip()

                if row == "--- [define prompt template end] ---":
                    if len(prompt_template) != 0:
                        t = {
                            "prompt_template": prompt_template.strip(),
                            "response_template": response_template.strip(),
                            "kwargs": kwargs.strip(),
                        }
                        yield idx, t
                        idx += 1
                        prompt_template = ""
                        response_template = ""
                        kwargs = ""

                elif row == "prompt_template:":
                    if not len(prompt_template) == 0:
                        raise AssertionError
                    flag = 0
                elif row == "response_template:":
                    flag = 1
                elif row == "kwargs:":
                    flag = 2
                else:
                    if flag == 0:
                        prompt_template += "{}\n".format(row)
                    elif flag == 1:
                        response_template += "{}\n".format(row)
                    elif flag == 2:
                        kwargs += "{}\n".format(row)
                    else:
                        raise NotImplementedError

    def _generate_prompt(self, entity_dl_path, template_dl_path, split):
        """Yields examples."""
        templates: List[dict] = list()
        for _, template in self._generate_template(template_dl_path, split):
            templates.append(template)

        idx = 0
        for _, sample in self._generate_entity(entity_dl_path, split):
            prompt_response = PromptResponseTemplate(
                dataset_name=self.config.name[:-7],
                templates=templates,
                sample=sample,
            )
            prompt, response = prompt_response.get_prompt_response()
            sample = {
                "prompt": prompt,
                "response": response,
            }
            yield idx, sample
            idx += 1

    def _generate_examples(self, **kwargs):
        """Yields examples."""
        if self.config.name in _entity_urls.keys():
            return self._generate_entity(**kwargs)
        elif self.config.name in _template_urls.keys():
            return self._generate_template(**kwargs)
        elif self.config.name in _prompt_urls.keys():
            return self._generate_prompt(**kwargs)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    pass
