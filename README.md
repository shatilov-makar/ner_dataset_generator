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
from dataset_builder import build_dataset


generator = NerDatasetGenerator()

generator.translate_sentences(dataset='E:\\Turk\\ner_dataset',
                              dataset_is_local=True,
                              src_lang='tur_Latn',
                              tgt_lang='kaz_Cyrl',
                              start_from=10_000,
                              stop_at=20_000
                              save_steps=500,
                              output_dir='kaz',
                              batch_size=5)
```


Затем вызываем функцию build_dataset, которая превратит сгенерированные предложения в NER-датасет
```
build_dataset(
    path_to_csv="kaz\\sentences\\18_22__10_000.csv",
    output_dir='translated_dataset')

```
