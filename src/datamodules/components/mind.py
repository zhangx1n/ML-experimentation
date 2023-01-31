#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author : Zhang Xin
@Contact: xinzhang_hp@163.com
@Time : 2022/12/24
"""
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import torch


class MINDDataset(Dataset):
    def __init__(self,
                 behaviors_path,
                 news_path,
                 dataset_attributes,
                 num_words_title,
                 num_words_abstract,
                 num_punctuation,
                 num_clicked_news_a_user):
        super().__init__()

        self.dataset_attributes = dataset_attributes
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.num_punctuation = num_punctuation
        self.num_clicked_news_a_user = num_clicked_news_a_user
        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + self.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(self.dataset_attributes['news']) & {'title', 'abstract', 'title_entities',
                                                                         'abstract_entities', 'title_punctuation'}
            })
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}  # {'n27984': 999}
        self.news2dict = self.news_parsed.to_dict('index')  # {'n27984': {'title': tensor([4873, ...])}}

        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities', 'title_length', 'abstract_length', 'title_punctuation', 'title_number'
        ] for attribute in self.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in self.dataset_attributes['record'])



        # 数据格式变为torch.tensor
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2])
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * self.num_words_title,
            'abstract': [0] * self.num_words_abstract,
            'title_entities': [0] * self.num_words_title,
            'abstract_entities': [0] * self.num_words_abstract,
            'title_length': 0,
            'abstract_length': 0,
            'title_punctuation': [0] * self.num_punctuation,
            'title_number': 0,
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        # 只保留需要的属性
        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in self.dataset_attributes['news']
        }

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in self.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            self.news2dict[x] for x in row.candidate_news.split()
            # [{属性1: tensor[]}, ...]    n_attribute * n_attribute_dim
        ]
        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[:self.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in self.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = self.num_clicked_news_a_user - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding] * repeated_times + item["clicked_news"]
        # len(item["clicked_news"]) = num_clicked_news_a_user

        return item

#
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     behaviors_path = "../data/MINDsmall_train/behaviors_parsed.tsv"
#     news_path = "../data/MINDsmall_train/news_parsed.tsv"
#     dataset = MINDDataset(behaviors_path, news_path)
#     item = dataset.__getitem__(2)
#     print(item)
