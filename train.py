#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для обучения модели Earth-Liberty.

Поддерживает:
- Предварительное обучение на текстовых корпусах
- Тонкую настройку на диалогах
- Обучение самосознания
- Сохранение и загрузку чекпоинтов
- Логирование метрик
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Optional
import torch
from tqdm.auto import tqdm

# Добавление корневой директории в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.language.config import ModelConfig
from model.language.tokenizer import Tokenizer, TokenizerConfig
from model.language.generator import TextGenerator, GeneratorConfig
from model.language.trainer import Trainer, TrainingArguments
from model.data.dataset import TextDataset, DialogueDataset, SelfAwarenessDataset
from model.data.processor import DataProcessor
from model.data.collator import DataCollator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        Объект с аргументами
    """
    parser = argparse.ArgumentParser(description="Обучение модели Earth-Liberty")
    
    # Общие аргументы
    parser.add_argument("--config", type=str, default="config/model_config.json",
                        help="Путь к конфигурационному файлу модели")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Директория для сохранения результатов")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для воспроизводимости")
    parser.add_argument("--device", type=str, default=None,
                        help="Устройство для обучения (cpu, cuda, mps)")
    
    # Аргументы для обучения
    parser.add_argument("--train_mode", type=str, default="pretrain",
                        choices=["pretrain", "finetune", "self_awareness"],
                        help="Режим обучения")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Размер батча")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Скорость обучения")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Количество шагов для накопления градиентов")
    parser.add_argument("--warmup_steps", type=int, default=10000,
                        help="Количество шагов для разогрева")
    parser.add_argument("--fp16", action="store_true",
                        help="Использовать mixed precision training")
    
    # Аргументы для данных
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Директория с данными")
    parser.add_argument("--processed_data_dir", type=str, default="data/processed",
                        help="Директория для обработанных данных")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Максимальная длина последовательности")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Доля данных для валидации")
    
    # Аргументы для токенизатора
    parser.add_argument("--train_tokenizer", action="store_true",
                        help="Обучить токенизатор перед обучением модели")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Путь к предобученному токенизатору")
    
    # Аргументы для чекпоинтов
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Путь к чекпоинту для продолжения обучения")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Количество шагов между сохранениями чекпоинтов")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Максимальное количество сохраняемых чекпоинтов")
    
    # Аргументы для логирования
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Количество шагов между логированием метрик")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Количество шагов между валидациями")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Использовать Weights & Biases для логирования")
    parser.add_argument("--wandb_project", type=str, default="earth-liberty",
                        help="Название проекта в Weights & Biases")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Название запуска в Weights & Biases")
    
    return parser.parse_args()

def prepare_data(args):
    """
    Подготовка данных для обучения.
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        Кортеж из путей к обучающим и валидационным файлам
    """
    logger.info("Подготовка данных для обучения")
    
    # Создание директорий
    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # Инициализация процессора данных
    processor = DataProcessor(
        data_dir=args.data_dir,
        output_dir=args.processed_data_dir,
        min_length=100,
        max_length=args.max_length * 10,  # Увеличиваем для разделения на чанки
        val_split=args.val_split,
        seed=args.seed
    )
    
    # Подготовка данных в зависимости от режима обучения
    if args.train_mode == "pretrain":
        # Подготовка текстового корпуса
        train_files, val_files = processor.process_text_corpus(
            input_pattern=os.path.join(args.data_dir, "corpus", "*.txt"),
            output_prefix="corpus",
            chunk_size=args.max_length * 2,
            clean_text=True
        )
    
    elif args.train_mode == "finetune":
        # Подготовка корпуса диалогов
        train_files, val_files = processor.process_dialogue_corpus(
            input_pattern=os.path.join(args.data_dir, "dialogues", "*.json"),
            output_prefix="dialogues",
            dialogues_per_file=1000,
            clean_text=True
        )
    
    elif args.train_mode == "self_awareness":
        # Подготовка данных для обучения самосознания
        train_files, val_files = processor.process_self_awareness_data(
            input_pattern=os.path.join(args.data_dir, "self_awareness", "*.json"),
            output_prefix="self_awareness",
            items_per_file=1000,
            clean_text=True
        )
    
    else:
        raise ValueError(f"Неизвестный режим обучения: {args.train_mode}")
    
    logger.info(f"Подготовлено {len(train_files)} обучающих файлов и {len(val_files)} валидационных файлов")
    
    return train_files, val_files

def train_tokenizer(args, train_files):
    """
    Обучение токенизатора.
    
    Args:
        args: Аргументы командной строки
        train_files: Список путей к обучающим файлам
        
    Returns:
        Обученный токенизатор
    """
    logger.info("Обучение токенизатора")
    
    # Инициализация процессора данных
    processor = DataProcessor(
        data_dir=args.data_dir,
        output_dir=args.processed_data_dir,
        min_length=100,
        max_length=args.max_length * 10,
        val_split=args.val_split,
        seed=args.seed
    )
    
    # Подготовка данных для обучения токенизатора
    tokenizer_data_path = processor.prepare_tokenizer_training_data(
        input_files=train_files,
        output_file=os.path.join(args.processed_data_dir, "tokenizer_training_data.txt"),
        max_samples=1000000,
        sample_length=1024
    )
    
    # Загрузка конфигурации модели
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Создание конфигурации токенизатора
    tokenizer_config = TokenizerConfig(
        vocab_size=config_dict.get("vocab_size", 100000),
        min_freq=config_dict.get("min_frequency", 2),
        bpe_vocab_size=config_dict.get("bpe_vocab_size", 50000),
        max_length=config_dict.get("max_position_embeddings", 2048),
        pad_token=config_dict.get("special_tokens", {}).get("pad", "<PAD>"),
        bos_token=config_dict.get("special_tokens", {}).get("bos", "<BOS>"),
        eos_token=config_dict.get("special_tokens", {}).get("eos", "<EOS>"),
        unk_token=config_dict.get("special_tokens", {}).get("unk", "<UNK>"),
        sep_token=config_dict.get("special_tokens", {}).get("sep", "<SEP>"),
        mask_token=config_dict.get("special_tokens", {}).get("mask", "<MASK>"),
        sys_token=config_dict.get("special_tokens", {}).get("sys", "<SYS>"),
        user_token=config_dict.get("special_tokens", {}).get("user", "<USER>"),
        assistant_token=config_dict.get("special_tokens", {}).get("assistant", "<ASSISTANT>"),
        emotion_tokens=config_dict.get("emotion_types", []),
        cache_size=config_dict.get("context_cache_size", 10000),
        use_bpe=config_dict.get("tokenizer_type", "bpe") == "bpe",
        use_sentencepiece=config_dict.get("use_sentencepiece", True)
    )
    
    # Инициализация токенизатора
    tokenizer = Tokenizer(config=tokenizer_config)
    
    # Загрузка данных для обучения
    with open(tokenizer_data_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # Обучение токенизатора
    tokenizer.train(texts)
    
    # Сохранение токенизатора
    tokenizer_output_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_output_path)
    
    logger.info(f"Токенизатор обучен и сохранен в {tokenizer_output_path}")
    
    return tokenizer

def load_tokenizer(args):
    """
    Загрузка токенизатора.
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        Загруженный токенизатор
    """
    # Определение пути к токенизатору
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    
    # Проверка существования файла
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Файл токенизатора не найден: {tokenizer_path}")
    
    # Загрузка конфигурации модели
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Создание конфигурации токенизатора
    tokenizer_config = TokenizerConfig(
        vocab_size=config_dict.get("vocab_size", 100000),
        min_freq=config_dict.get("min_frequency", 2),
        bpe_vocab_size=config_dict.get("bpe_vocab_size", 50000),
        max_length=config_dict.get("max_position_embeddings", 2048),
        pad_token=config_dict.get("special_tokens", {}).get("pad", "<PAD>"),
        bos_token=config_dict.get("special_tokens", {}).get("bos", "<BOS>"),
        eos_token=config_dict.get("special_tokens", {}).get("eos", "<EOS>"),
        unk_token=config_dict.get("special_tokens", {}).get("unk", "<UNK>"),
        sep_token=config_dict.get("special_tokens", {}).get("sep", "<SEP>"),
        mask_token=config_dict.get("special_tokens", {}).get("mask", "<MASK>"),
        sys_token=config_dict.get("special_tokens", {}).get("sys", "<SYS>"),
        user_token=config_dict.get("special_tokens", {}).get("user", "<USER>"),
        assistant_token=config_dict.get("special_tokens", {}).get("assistant", "<ASSISTANT>"),
        emotion_tokens=config_dict.get("emotion_types", []),
        cache_size=config_dict.get("context_cache_size", 10000),
        use_bpe=config_dict.get("tokenizer_type", "bpe") == "bpe",
        use_sentencepiece=config_dict.get("use_sentencepiece", True)
    )
    
    # Инициализация токенизатора
    tokenizer = Tokenizer(config=tokenizer_config)
    
    # Загрузка токенизатора
    tokenizer.load(tokenizer_path)
    
    logger.info(f"Токенизатор загружен из {tokenizer_path}")
    
    return tokenizer

def create_datasets(args, tokenizer, train_files, val_files):
    """
    Создание датасетов для обучения и валидации.
    
    Args:
        args: Аргументы командной строки
        tokenizer: Токенизатор
        train_files: Список путей к обучающим файлам
        val_files: Список путей к валидационным файлам
        
    Returns:
        Кортеж из обучающего и валидационного датасетов
    """
    logger.info("Создание датасетов")
    
    # Определение устройства
    device = get_device(args)
    
    # Создание датасетов в зависимости от режима обучения
    if args.train_mode == "pretrain":
        # Создание датасетов для предварительного обучения
        train_dataset = TextDataset(
            file_paths=train_files,
            tokenizer=tokenizer,
            max_length=args.max_length,
            stride=args.max_length // 2,
            return_tensors=True,
            device=None  # Устройство будет установлено позже
        )
        
        val_dataset = TextDataset(
            file_paths=val_files,
            tokenizer=tokenizer,
            max_length=args.max_length,
            stride=args.max_length // 2,
            return_tensors=True,
            device=None
        )
    
    elif args.train_mode == "finetune":
        # Создание датасетов для тонкой настройки
        train_dataset = DialogueDataset(
            file_paths=train_files,
            tokenizer=tokenizer,
            max_length=args.max_length,
            return_tensors=True,
            device=None
        )
        
        val_dataset = DialogueDataset(
            file_paths=val_files,
            tokenizer=tokenizer,
            max_length=args.max_length,
            return_tensors=True,
            device=None
        )
    
    elif args.train_mode == "self_awareness":
        # Создание датасетов для обучения самосознания
        train_dataset = SelfAwarenessDataset(
            file_paths=train_files,
            tokenizer=tokenizer,
            max_length=args.max_length,
            return_tensors=True,
            device=None,
            emotion_types=tokenizer.config.emotion_tokens
        )
        
        val_dataset = SelfAwarenessDataset(
            file_paths=val_files,
            tokenizer=tokenizer,
            max_length=args.max_length,
            return_tensors=True,
            device=None,
            emotion_types=tokenizer.config.emotion_tokens
        )
    
    else:
        raise ValueError(f"Неизвестный режим обучения: {args.train_mode}")
    
    logger.info(f"Создано {len(train_dataset)} обучающих примеров и {len(val_dataset)} валидационных примеров")
    
    return train_dataset, val_dataset

def create_model(args, tokenizer):
    """
    Создание модели.
    
    Args:
        args: Аргументы командной строки
        tokenizer: Токенизатор
        
    Returns:
        Созданная модель
    """
    logger.info("Создание модели")
    
    # Загрузка конфигурации модели
    config = ModelConfig.load(args.config)
    
    # Создание конфигурации генератора
    generator_config = GeneratorConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        dropout=config.hidden_dropout_prob,
        max_length=config.max_position_embeddings,
        activation=config.hidden_act,
        use_flash_attention=config.use_flash_attention,
        use_multi_query=config.use_multi_query,
        use_rotary=config.use_rotary,
        use_cache=config.use_cache,
        tie_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        layer_norm_epsilon=config.layer_norm_eps
    )
    
    # Создание модели
    model = TextGenerator(config=generator_config)
    
    logger.info(f"Создана модель с {sum(p.numel() for p in model.parameters())} параметрами")
    
    return model, config

def get_device(args):
    """
    Определение устройства для обучения.
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        Устройство для обучения
    """
    if args.device:
        # Использование устройства, указанного в аргументах
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        # Использование CUDA, если доступно
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Использование MPS (для Apple Silicon), если доступно
        device = torch.device("mps")
    else:
        # Использование CPU
        device = torch.device("cpu")
    
    logger.info(f"Используется устройство: {device}")
    
    return device

def main():
    """
    Основная функция для обучения модели.
    """
    # Парсинг аргументов
    args = parse_args()
    
    # Создание директорий
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Подготовка данных
    train_files, val_files = prepare_data(args)
    
    # Обучение или загрузка токенизатора
    if args.train_tokenizer:
        tokenizer = train_tokenizer(args, train_files)
    else:
        tokenizer = load_tokenizer(args)
    
    # Создание датасетов
    train_dataset, val_dataset = create_datasets(args, tokenizer, train_files, val_files)
    
    # Создание модели
    model, config = create_model(args, tokenizer)
    
    # Создание коллатора данных
    data_collator = DataCollator(
        pad_token_id=tokenizer.token_to_id(tokenizer.config.pad_token),
        label_pad_token_id=-100,
        max_length=args.max_length,
        return_tensors=True,
        device=None,  # Устройство будет установлено тренером
        pad_to_multiple_of=8 if args.fp16 else None
    )
    
    # Создание аргументов для обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        seed=args.seed,
        fp16=args.fp16,
        fp16_opt_level="O1",
        local_rank=-1
    )
    
    # Создание тренера
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Загрузка чекпоинта, если указан
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Обучение модели
    trainer.train()
    
    # Финальная валидация
    eval_metrics = trainer.evaluate()
    
    # Вывод результатов
    logger.info(f"Обучение завершено. Метрики валидации: {eval_metrics}")
    
    # Сохранение финальной модели
    trainer.save_checkpoint("final")
    
    logger.info(f"Финальная модель сохранена в {os.path.join(args.output_dir, 'final')}")

if __name__ == "__main__":
    main() 