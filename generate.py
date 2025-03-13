#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для генерации текста с помощью обученной модели Earth-Liberty.

Поддерживает:
- Генерацию текста на основе промпта
- Различные стратегии генерации (temperature, top-k, top-p)
- Интерактивный режим
- Пакетную генерацию
- Эмоциональное состояние
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
from model.language.tokenizer import Tokenizer
from model.language.generator import TextGenerator
from model.language.context import ContextManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/generation.log"),
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
    parser = argparse.ArgumentParser(description="Генерация текста с помощью модели Earth-Liberty")
    
    # Общие аргументы
    parser.add_argument("--model_dir", type=str, default="output/final",
                        help="Директория с моделью")
    parser.add_argument("--device", type=str, default=None,
                        help="Устройство для генерации (cpu, cuda, mps)")
    
    # Аргументы для генерации
    parser.add_argument("--prompt", type=str, default=None,
                        help="Промпт для генерации")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Файл с промптами для пакетной генерации")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Файл для сохранения результатов")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Максимальная длина генерируемого текста")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Температура для генерации")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Параметр top-k для генерации")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Параметр top-p для генерации")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Штраф за повторения")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Количество генерируемых последовательностей")
    parser.add_argument("--do_sample", action="store_true",
                        help="Использовать сэмплирование вместо жадного декодирования")
    
    # Аргументы для эмоционального состояния
    parser.add_argument("--emotion", type=str, default=None,
                        help="Эмоциональное состояние в формате 'emotion1:value1,emotion2:value2'")
    parser.add_argument("--self_awareness_level", type=float, default=0.5,
                        help="Уровень самосознания (от 0 до 1)")
    
    # Аргументы для режима
    parser.add_argument("--interactive", action="store_true",
                        help="Интерактивный режим")
    parser.add_argument("--batch", action="store_true",
                        help="Пакетный режим")
    
    return parser.parse_args()

def load_model(model_dir: str, device: torch.device) -> tuple:
    """
    Загрузка модели из директории.
    
    Args:
        model_dir: Директория с моделью
        device: Устройство для загрузки модели
        
    Returns:
        Кортеж из модели, токенизатора и конфигурации
    """
    logger.info(f"Загрузка модели из {model_dir}")
    
    # Проверка существования директории
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Директория с моделью не найдена: {model_dir}")
    
    # Загрузка конфигурации
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    
    config = ModelConfig.load(config_path)
    
    # Загрузка токенизатора
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Файл токенизатора не найден: {tokenizer_path}")
    
    tokenizer = Tokenizer()
    tokenizer.load(tokenizer_path)
    
    # Загрузка модели
    model_state_path = os.path.join(model_dir, "model_state.pt")
    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Файл состояния модели не найден: {model_state_path}")
    
    # Создание модели
    from model.language.generator import GeneratorConfig
    
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
    
    model = TextGenerator(config=generator_config)
    
    # Загрузка состояния модели
    model_state = torch.load(model_state_path, map_location=device)
    model.load_state_dict(model_state["model_state_dict"])
    
    # Перемещение модели на устройство
    model.to(device)
    
    # Установка режима оценки
    model.eval()
    
    logger.info(f"Модель загружена: {sum(p.numel() for p in model.parameters())} параметров")
    
    return model, tokenizer, config

def parse_emotional_state(emotion_str: Optional[str], emotion_types: List[str]) -> Dict[str, float]:
    """
    Парсинг строки с эмоциональным состоянием.
    
    Args:
        emotion_str: Строка с эмоциональным состоянием в формате 'emotion1:value1,emotion2:value2'
        emotion_types: Список поддерживаемых типов эмоций
        
    Returns:
        Словарь с эмоциональным состоянием
    """
    if not emotion_str:
        # Возвращаем нейтральное эмоциональное состояние
        return {emotion: 0.5 for emotion in emotion_types}
    
    # Парсинг строки
    emotional_state = {}
    
    for item in emotion_str.split(","):
        if ":" not in item:
            continue
        
        emotion, value_str = item.split(":", 1)
        emotion = emotion.strip()
        
        try:
            value = float(value_str.strip())
            # Ограничение значения от 0 до 1
            value = max(0.0, min(1.0, value))
            emotional_state[emotion] = value
        except ValueError:
            logger.warning(f"Неверное значение для эмоции {emotion}: {value_str}")
    
    # Добавление отсутствующих эмоций
    for emotion in emotion_types:
        if emotion not in emotional_state:
            emotional_state[emotion] = 0.5
    
    return emotional_state

def generate_text(
    model: TextGenerator,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    num_return_sequences: int,
    do_sample: bool,
    emotional_state: Optional[Dict[str, float]] = None,
    device: torch.device = torch.device("cpu")
) -> List[str]:
    """
    Генерация текста с помощью модели.
    
    Args:
        model: Модель для генерации
        tokenizer: Токенизатор
        prompt: Промпт для генерации
        max_length: Максимальная длина генерируемого текста
        temperature: Температура для генерации
        top_k: Параметр top-k для генерации
        top_p: Параметр top-p для генерации
        repetition_penalty: Штраф за повторения
        num_return_sequences: Количество генерируемых последовательностей
        do_sample: Использовать сэмплирование вместо жадного декодирования
        emotional_state: Эмоциональное состояние
        device: Устройство для генерации
        
    Returns:
        Список сгенерированных текстов
    """
    # Токенизация промпта
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], device=device)
    
    # Генерация текста
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            emotional_state=emotional_state
        )
    
    # Декодирование результатов
    generated_texts = []
    
    for output_sequence in output_sequences:
        # Декодирование токенов
        generated_text = tokenizer.decode(output_sequence.tolist(), skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

def interactive_mode(
    model: TextGenerator,
    tokenizer: Tokenizer,
    config: ModelConfig,
    args: argparse.Namespace,
    device: torch.device
):
    """
    Интерактивный режим генерации текста.
    
    Args:
        model: Модель для генерации
        tokenizer: Токенизатор
        config: Конфигурация модели
        args: Аргументы командной строки
        device: Устройство для генерации
    """
    logger.info("Запуск интерактивного режима")
    
    # Создание менеджера контекста
    context_manager = ContextManager(
        max_history=config.max_history_length,
        embedding_dim=config.context_embedding_dim,
        use_hierarchical=config.use_hierarchical_context
    )
    
    # Парсинг эмоционального состояния
    emotional_state = parse_emotional_state(args.emotion, config.emotion_types)
    
    print("\n" + "=" * 50)
    print("Интерактивный режим генерации текста Earth-Liberty")
    print("Введите текст для генерации или 'exit' для выхода")
    print("=" * 50 + "\n")
    
    # Вывод текущего эмоционального состояния
    print("Текущее эмоциональное состояние:")
    for emotion, value in emotional_state.items():
        print(f"  - {emotion}: {value:.2f}")
    
    print(f"Уровень самосознания: {args.self_awareness_level:.2f}")
    print("\n" + "-" * 50 + "\n")
    
    # Цикл генерации
    while True:
        # Получение ввода пользователя
        user_input = input("Вы: ")
        
        # Проверка на выход
        if user_input.lower() in ["exit", "quit", "выход"]:
            break
        
        # Проверка на команды
        if user_input.startswith("/"):
            command = user_input[1:].strip().lower()
            
            if command.startswith("emotion "):
                # Изменение эмоционального состояния
                emotion_str = command[8:].strip()
                emotional_state = parse_emotional_state(emotion_str, config.emotion_types)
                
                print("\nЭмоциональное состояние изменено:")
                for emotion, value in emotional_state.items():
                    print(f"  - {emotion}: {value:.2f}")
                
                continue
            
            elif command.startswith("awareness "):
                # Изменение уровня самосознания
                try:
                    args.self_awareness_level = float(command[10:].strip())
                    args.self_awareness_level = max(0.0, min(1.0, args.self_awareness_level))
                    print(f"\nУровень самосознания изменен: {args.self_awareness_level:.2f}")
                except ValueError:
                    print("\nНеверное значение для уровня самосознания")
                
                continue
            
            elif command == "help":
                # Вывод справки
                print("\nДоступные команды:")
                print("  /emotion emotion1:value1,emotion2:value2 - изменение эмоционального состояния")
                print("  /awareness value - изменение уровня самосознания (от 0 до 1)")
                print("  /help - вывод справки")
                print("  exit, quit, выход - выход из программы")
                
                continue
        
        # Обновление контекста
        context_manager.update_context([{"role": "user", "content": user_input}])
        
        # Генерация текста
        try:
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=1,
                do_sample=args.do_sample,
                emotional_state=emotional_state,
                device=device
            )
            
            # Вывод результата
            print(f"\nEarth-Liberty: {generated_texts[0]}")
            
            # Обновление контекста
            context_manager.update_context([{"role": "assistant", "content": generated_texts[0]}])
            
        except Exception as e:
            logger.error(f"Ошибка при генерации текста: {e}")
            print(f"\nПроизошла ошибка при генерации текста: {e}")
        
        print("\n" + "-" * 50 + "\n")

def batch_mode(
    model: TextGenerator,
    tokenizer: Tokenizer,
    config: ModelConfig,
    args: argparse.Namespace,
    device: torch.device
):
    """
    Пакетный режим генерации текста.
    
    Args:
        model: Модель для генерации
        tokenizer: Токенизатор
        config: Конфигурация модели
        args: Аргументы командной строки
        device: Устройство для генерации
    """
    logger.info("Запуск пакетного режима")
    
    # Проверка наличия файла с промптами
    if not args.prompt_file:
        raise ValueError("Не указан файл с промптами для пакетного режима")
    
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Файл с промптами не найден: {args.prompt_file}")
    
    # Проверка наличия файла для сохранения результатов
    if not args.output_file:
        raise ValueError("Не указан файл для сохранения результатов")
    
    # Загрузка промптов
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Загружено {len(prompts)} промптов")
    
    # Парсинг эмоционального состояния
    emotional_state = parse_emotional_state(args.emotion, config.emotion_types)
    
    # Генерация текста для каждого промпта
    results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Генерация")):
        try:
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=args.num_return_sequences,
                do_sample=args.do_sample,
                emotional_state=emotional_state,
                device=device
            )
            
            # Сохранение результатов
            for j, text in enumerate(generated_texts):
                results.append({
                    "prompt": prompt,
                    "generated_text": text,
                    "sequence_id": j,
                    "prompt_id": i
                })
            
        except Exception as e:
            logger.error(f"Ошибка при генерации текста для промпта {i}: {e}")
            results.append({
                "prompt": prompt,
                "error": str(e),
                "prompt_id": i
            })
    
    # Сохранение результатов
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты сохранены в {args.output_file}")

def get_device(args):
    """
    Определение устройства для генерации.
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        Устройство для генерации
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
    Основная функция для генерации текста.
    """
    # Парсинг аргументов
    args = parse_args()
    
    # Создание директорий
    os.makedirs("logs", exist_ok=True)
    
    # Определение устройства
    device = get_device(args)
    
    # Загрузка модели
    model, tokenizer, config = load_model(args.model_dir, device)
    
    # Выбор режима
    if args.interactive:
        # Интерактивный режим
        interactive_mode(model, tokenizer, config, args, device)
    
    elif args.batch:
        # Пакетный режим
        batch_mode(model, tokenizer, config, args, device)
    
    else:
        # Одиночная генерация
        if not args.prompt:
            raise ValueError("Не указан промпт для генерации")
        
        # Парсинг эмоционального состояния
        emotional_state = parse_emotional_state(args.emotion, config.emotion_types)
        
        # Генерация текста
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_return_sequences,
            do_sample=args.do_sample,
            emotional_state=emotional_state,
            device=device
        )
        
        # Вывод результатов
        for i, text in enumerate(generated_texts):
            print(f"\nРезультат {i+1}:")
            print(text)
            print("\n" + "-" * 50)
        
        # Сохранение результатов, если указан файл
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "prompt": args.prompt,
                    "generated_texts": generated_texts,
                    "parameters": {
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "do_sample": args.do_sample,
                        "emotional_state": emotional_state,
                        "self_awareness_level": args.self_awareness_level
                    }
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Результаты сохранены в {args.output_file}")

if __name__ == "__main__":
    main() 