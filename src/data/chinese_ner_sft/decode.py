#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List


def bmeo_decode(tokens: List[str], ner_tags: List[str]):
    """
    This function is for BMEO tags decode.
    reference:
    BIO tags decode.
    https://zhuanlan.zhihu.com/p/147537898
    """
    spans = []

    i = 0
    l = len(ner_tags)
    while i < l:
        tag = ner_tags[i]
        if tag[0] == "B":
            entity_label = tag[2:]
            start_idx = i

            i += 1
            if i < l:
                tag = ner_tags[i]
                while tag[0] == 'M':
                    i += 1
                    if i >= l:
                        break
                    tag = ner_tags[i]

            if tag[0] != "E":
                end_idx = i
            else:
                end_idx = i + 1

            sub_words = tokens[start_idx: end_idx]
            sub_words = [sub_word[2:] if sub_word.startswith('##') else sub_word for sub_word in sub_words]
            entity_text = ''.join(sub_words)

            spans.append({
                "text": entity_text,
                "entity_label": entity_label,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
            i -= 1
        i += 1

    return spans


def bio_decode(tokens: List[str], ner_tags: List[str]):
    """
    This function is for BIO tags decode.
    reference:
    BIO tags decode.
    https://zhuanlan.zhihu.com/p/147537898
    """
    spans = []

    i = 0
    l = len(ner_tags)
    while i < l:
        tag = ner_tags[i]
        if tag[0] == "B":
            entity_label = tag[2:]
            start_idx = i

            i += 1

            if i < l:
                tag = ner_tags[i]

                while tag[0] == 'I':
                    i += 1
                    if i >= l:
                        break
                    tag = ner_tags[i]

            end_idx = i

            sub_words = tokens[start_idx: end_idx]
            sub_words = [sub_word[2:] if sub_word.startswith('##') else sub_word for sub_word in sub_words]
            entity_text = ''.join(sub_words)

            spans.append({
                "text": entity_text,
                "entity_label": entity_label,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
            i -= 1
        i += 1

    return spans


def bioes_decode(tokens: List[str], ner_tags: List[str]):
    """
    This function is for BIOES tags decode.
    reference:
    BIO tags decode.
    https://zhuanlan.zhihu.com/p/147537898
    """
    spans = []

    i = 0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag[0] == "B":
            entity_label = tag[2:]
            start_idx = i

            i += 1
            tag = ner_tags[i]

            while tag[0] == 'I':
                i += 1
                tag = ner_tags[i]

            if tag[0] != "E":
                raise AssertionError

            end_idx = i + 1

            sub_words = tokens[start_idx: end_idx]
            sub_words = [sub_word[2:] if sub_word.startswith('##') else sub_word for sub_word in sub_words]
            entity_text = ''.join(sub_words)

            spans.append({
                "text": entity_text,
                "entity_label": entity_label,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
            i -= 1
        elif tag[0] == "S":
            entity_label = tag[2:]
            spans.append({
                "text": tokens[i],
                "entity_label": entity_label,
                "start_idx": i,
                "end_idx": i
            })
        i += 1

    return spans


if __name__ == '__main__':
    pass
