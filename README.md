# Speaker-Recognizer

Распознавание человека по голосу с записи или буфера.

## Установка

Для запуска нужно установить необходимые пакеты:

```
pip install -r requirements.txt
```

## Запуск

Запуск из директории `source` файла `source.py`

```
cd source/
python source.py
```

Перед началом надо обучить модель методом `save()` или `save_from_path()` сэмплами голоса, после чего тренировать модель `train_model()`.

## Распознавание

Метод `predict()` угадывает какой пользователь на записи

Метод `verify()` проверяет на совпадение
