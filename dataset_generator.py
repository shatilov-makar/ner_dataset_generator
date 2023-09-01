import pandas as pd
import itertools
import os
import re
import torch
from datetime import datetime
from tqdm import trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from typing import Optional, Union

from ner_types import (
    Model,
    Tokenizer,
    Langs
)


class NerDatasetGenerator:
    def __init__(
            self,
            model_name: Optional[Model] = Model.SMALL_MODEL.value,
            tokenizer_name: Optional[Tokenizer] = Tokenizer.DEFAULT_TOKENIZER.value,
    ) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if (torch.cuda.is_available()):
            self.model.cuda()
        else:
            print('No CUDA GPUs are available!')
            return
        self.tokenizer = tokenizer_name

    def translate_sentences(
        self,
        dataset: str,
        src_lang:  Union[Langs, str],
        tgt_lang: Union[Langs, str],
        output_dir: os.PathLike,
        dataset_is_local: bool = False,
        start_from: int = 0,
        stop_at: Optional[int] = None,
        save_steps: Optional[int] = None,
        batch_size: int = 5
    ) -> None:
        dataset = load_from_disk(
            dataset) if dataset_is_local else load_dataset(dataset)
        preprocessed_dataset = self.__preprocess_dataset(
            dataset, start_from, stop_at)
        self.__generate(preprocessed_dataset, batch_size, src_lang,
                        tgt_lang, save_steps, output_dir)

    def __preprocess_dataset(
        self,
        dataset: DatasetDict,
        start_from: int = 0,
        stop_at: Optional[int] = None
    ) -> list:
        print("Preprocessing started....")
        dataset_as_list = []
        for key in dataset:
            df = pd.DataFrame(dataset[key])
            tokens_column, tag_column = '', ''
            tokens_columns = list(
                filter(lambda col: 'token' in col, df.columns))
            tag_columns = list(filter(lambda col: 'tag' in col, df.columns))

            if len(tokens_columns) == 1 and len(tag_columns) == 1:
                tokens_column = tokens_columns[0]
                tag_column = tag_columns[0]
            else:
                print('Columns with tokens or tags not found!')
                return -1

            tokens_count = sum(df[tokens_column].str.len())
            tags_count = sum(df[tag_column].str.len())

            if tokens_count != tags_count:
                print(f'The number of entities in columns with tokens and tags is not the same! \
                        {key}: Entities in "{tokens_column}" = {tokens_count}, \
                        but "{tag_column}" = {tags_count}')
                return -1
            dataset_as_list += df.to_dict('records')
        if stop_at:
            dataset_with_tags = self.__insert_tags(
                dataset_as_list[start_from:stop_at], tokens_column, tag_column)
        else:
            dataset_with_tags = self.__insert_tags(
                dataset_as_list[start_from::], tokens_column, tag_column)
        print(f'Done! Preprocessed {len(dataset_with_tags)} sentences')
        return dataset_with_tags

    def __insert_tags(
        self,
        dataset: list,
        tokens_col: str = 'tokens',
        tags_col: str = 'ner_tags'

    ) -> list:
        dataset_with_tags = []
        for row in dataset:
            text = ''
            for word, tag in zip(row[tokens_col], row[tags_col]):
                word = re.sub(r'[ \[ \]]', '', word)

                if tag != 0:
                    text += f' [{tag}] {word} [/{tag}] '
                elif word in {"(", "[",  '@'}:
                    text += f' {word}'
                elif word in {")", "]",  '.', ',', ':', '!', '?'}:
                    text += f'{word} '
                else:
                    text += f' {word} '
            text = text.replace('  ', ' ')
            text = text.replace(' . ', '. ')
            text = text.replace(' , ', ', ')
            text = text.replace(' ! ', '! ')
            text = text.replace(' : ', ': ')
            dataset_with_tags.append(text.strip())
        return dataset_with_tags

    def __generate(
        self,
        src_dataset: list,
        batch_size: int,
        src_lang: Union[Langs, str],
        tgt_lang: Union[Langs, str],
        save_steps: Optional[int] = None,
        output_dir:  Optional[os.PathLike] = None,
    ) -> None:
        path_to_sentences = f'{output_dir}/sentences'

        if not os.path.exists(path_to_sentences):
            os.makedirs(path_to_sentences)
        output_result = []

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer,  src_lang=src_lang)
        pbar = trange(0, len(src_dataset), batch_size, ncols=80)
        for idx in pbar:
            step = int(idx / batch_size)
            pbar.set_description(f"Translating dataset to {tgt_lang}")
            start_idx = idx
            end_idx = idx + batch_size
            inputs = tokenizer(src_dataset[start_idx: end_idx],
                               padding=True, truncation=True, max_length=100,
                               return_tensors="pt").to('cuda:0')
            bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
            with torch.no_grad():
                translated_tokens = self.model.generate(**inputs,
                                                        forced_bos_token_id=bos_token_id,
                                                        max_length=100,
                                                        num_beams=5,
                                                        num_return_sequences=1,
                                                        early_stopping=True)

            output = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True)
            output_result.extend(output)

            del (inputs)
            torch.cuda.empty_cache()

            if (step > 0 and step % save_steps == 0) \
                    or len(output_result) == len(src_dataset):
                output = pd.DataFrame(output_result)
                dt = datetime.now()
                output.to_csv(
                    f'{path_to_sentences}/{dt.hour}_{dt.minute}__{idx}.csv')

        return output_result
