# Минимальные требования

* Оперативная память: 16 GB ОЗУ
* Видеокарта с объемом видеопамяти в 6 GB

# Как использовать

Устанавливаем зависимости:
```
pip install -r requirements.txt

```

Создаем объект класса **NerDatasetGenerator**, после чего вызываем функцию **translate_sentences**. В аргументах функции указываем гипперпараметры, датасет должен иметь такой же формат представления данных как у [WikiAnn](https://huggingface.co/datasets/wikiann)

Процесс перевода может занять продолжительное время.

```
from dataset_generator import NerDatasetGenerator
from dataset_builder import NerDatasetBuilder


generator = NerDatasetGenerator()

generator.translate_sentences(dataset='E:\\Turk\\Ner\\nerd',
                              dataset_is_local=True,
                              src_lang='kaz_Cyrl',
                              tgt_lang='tur_Latn',
                              generate_count=10_000,
                              save_steps=500,
                              output_dir='turk',
                              batch_size=5)
```


Затем создаем объект NerDatasetBuilder, который превратит сгенерированные предложения в NER-датасет
```
builder = NerDatasetBuilder()
builder.build_dataset(
    path_to_csv="turk\\sentences\\18_22__10_000.csv",
    output_dir='translated_dataset')

```
