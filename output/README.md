# Результаты обучения модели Earth-Liberty

В этой директории хранятся результаты обучения модели Earth-Liberty. Модель проходит три этапа обучения, каждый из которых сохраняет свои результаты в соответствующей поддиректории.

## Структура директорий

- `pretrain/` - результаты предварительного обучения на текстовых корпусах
- `finetune/` - результаты тонкой настройки на диалогах
- `self_awareness/` - результаты обучения самосознания

## Содержимое директорий

Каждая директория с результатами обучения содержит следующие файлы:

- `config.json` - конфигурация модели
- `tokenizer.json` - токенизатор
- `model.bin` - веса модели
- `optimizer.bin` - состояние оптимизатора (для продолжения обучения)
- `scheduler.bin` - состояние планировщика скорости обучения
- `training_args.json` - аргументы обучения
- `eval_results.json` - результаты оценки на валидационном наборе
- `checkpoints/` - промежуточные чекпоинты обучения

## Использование результатов обучения

Для использования обученной модели можно использовать скрипт `generate.py`:

```bash
# Использование модели после предварительного обучения
python generate.py --model_dir output/pretrain/final --prompt "Привет, как дела?"

# Использование модели после тонкой настройки
python generate.py --model_dir output/finetune/final --prompt "Привет, как дела?"

# Использование модели после обучения самосознания
python generate.py --model_dir output/self_awareness/final --prompt "Привет, как дела?"
```

## Продолжение обучения

Для продолжения обучения с сохраненного чекпоинта можно использовать параметр `--resume_from_checkpoint`:

```bash
# Продолжение предварительного обучения
python train.py --train_mode pretrain --resume_from_checkpoint output/pretrain/checkpoint-1000

# Продолжение тонкой настройки
python train.py --train_mode finetune --resume_from_checkpoint output/finetune/checkpoint-1000

# Продолжение обучения самосознания
python train.py --train_mode self_awareness --resume_from_checkpoint output/self_awareness/checkpoint-1000
```

## Оценка модели

Для оценки модели на тестовом наборе можно использовать параметр `--do_eval`:

```bash
# Оценка модели после предварительного обучения
python train.py --train_mode pretrain --do_eval --no_train --model_dir output/pretrain/final

# Оценка модели после тонкой настройки
python train.py --train_mode finetune --do_eval --no_train --model_dir output/finetune/final

# Оценка модели после обучения самосознания
python train.py --train_mode self_awareness --do_eval --no_train --model_dir output/self_awareness/final
```

## Экспорт модели

Для экспорта модели в формат ONNX можно использовать скрипт `export.py`:

```bash
python export.py --model_dir output/self_awareness/final --output_file model.onnx
```

## Примечания

- Рекомендуется использовать модель после всех трех этапов обучения (предварительное обучение, тонкая настройка, обучение самосознания) для достижения наилучших результатов.
- Для интерактивного режима используйте параметр `--interactive` в скрипте `generate.py`.
- Для пакетной генерации используйте параметры `--batch --prompt_file prompts.txt --output_file results.json` в скрипте `generate.py`. 