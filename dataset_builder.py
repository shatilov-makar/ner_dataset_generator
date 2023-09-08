import pandas as pd
import itertools
import os
import re
from datasets import Dataset, DatasetDict
from nltk.tokenize import word_tokenize

def build_dataset(
    path_to_csv: os.PathLike,
    output_dir: os.PathLike,
) -> None:
    df = pd.read_csv(path_to_csv, index_col=0)
    df.drop_duplicates(inplace=True)
    clear_texts = __clear_texts(
        [e['0'] for e in df.to_dict('records')])
    mapped_texts = __map_tags_with_labels(clear_texts)

    ner_df = pd.DataFrame.from_dict(mapped_texts)

    datasetDict = DatasetDict({
        "train": Dataset.from_pandas(ner_df),
    })
    datasetDict.save_to_disk(output_dir)


def __clear_texts(dataset: list
                  ) -> list:
    init_texts = []
    for text in dataset:
        text = text.replace('[', ' [')
        text = text.replace(']', '] ')
        text = text.replace('  ', ' ')
        init_texts.append(text.strip())
    clear_texts = []

    for text in init_texts:
        opened_ids = []
        closed_ids = []
        words = text.split(' ')
        for i, word in enumerate(words):
            if re.search(r'\[\d]', word) is not None:
                opened_ids.append({'index': i, 'tag': word.strip('[]')})
            elif re.search(r'\[/\d\]', word) is not None:
                closed_ids.append({'index': i, 'tag': word.strip('[/]')})
        if len(opened_ids) == len(closed_ids):
            normal_ids = all(i == 2 for i in [closed['index'] - opened['index']
                                                  for opened, closed in zip(opened_ids, closed_ids)])
            normal_tags = all([opened['tag'] == closed['tag']
                               for opened, closed in zip(opened_ids, closed_ids)])
            if normal_ids and normal_tags and text.count('[') == len(opened_ids) * 2:
                clear_texts.append(text)
    print(f'Found {len(init_texts) - len(clear_texts)} broken sentences.', '\n',
          f'{len(clear_texts)} sentences will be used')
    return clear_texts


def __map_tags_with_labels(
        dataset: list) -> list:
    mapped_dataset = []
    for text in dataset:
        raw = {'tokens': [], 'ner_tags': []}
        tokenized_str = iter(word_tokenize(text))
        for token in tokenized_str:
            if token == '[':
                tag = next(tokenized_str)        
                next(tokenized_str)
                word = next(tokenized_str)
                while word != '[':
                    raw['tokens'].append(word)
                    raw['ner_tags'].append(int(tag))
                    word = next(tokenized_str)
                next(tokenized_str), next(tokenized_str)
            else:
                raw['ner_tags'].append(0)
                raw['tokens'].append(token)
        mapped_dataset.append(raw)
    return mapped_dataset
