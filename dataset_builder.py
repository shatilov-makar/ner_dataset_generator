import pandas as pd
import itertools
import os
import re
from datasets import Dataset, DatasetDict


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
        open_count = 0
        closed_count = 0
        flag = True
        words = text.split(' ')
        for i, word in enumerate(words):
            try:
                if re.search(r'\[\d]', word) is not None:
                    open_count += 1
                    if re.search(r'\[/\d\]', words[i + 1]) is not None \
                            or words[i + 1] == '' \
                            or re.search(r'\[/\d\]', words[i + 2]) is None:
                        flag = False
                        break
                elif re.search(r'\[/\d\]', word) is not None \
                        and re.search(r'\[\d]', words[i - 2]) is not None:
                    closed_count += 1
            except:
                flag = False
                break
        if flag and open_count == closed_count:
            clear_texts.append(text)
    print(f'Found {len(init_texts) - len(clear_texts)} broken sentences.', '\n',
          f'{len(clear_texts)}  will be used')
    return clear_texts


def __map_tags_with_labels(
        dataset: list) -> list:
    mapped_dataset = []
    for text in dataset:
        words = []

        for (name, age) in itertools.zip_longest(re.split(r"[ ]", text, 10000),
                                                 re.findall(r"[ ]", text), fillvalue=''):
            f = name + age
            if f.strip() != '':
                words.append(f.strip())

        raw = {'tokens': [], 'ner_tags': []}
        word_iter = iter(words)

        for word in word_iter:
            if re.search(r'\[\d]', word) is not None:
                plus_1 = next(word_iter)
                if re.search(r'\[/\d\]', plus_1) is None:
                    plus_2 = next(word_iter)
                if re.search(r'\[/\d\]', plus_2) is not None:
                    raw['tokens'].append(plus_1)
                    raw['ner_tags'].append(
                        int(re.sub(r'[ \[ \]]', '', word)))
            elif re.search(r'\[/\d\]', word) is None:
                raw['tokens'].append(word)
                raw['ner_tags'].append(0)

        if len(raw['tokens']) == len(raw['ner_tags']):
            mapped_dataset.append(raw)
    return mapped_dataset
