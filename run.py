#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from model.earth_liberty_model import EarthLibertyModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/earth_liberty.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("earth_liberty")

def parse_arguments():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        Аргументы командной строки
    """
    parser = argparse.ArgumentParser(description="Earth-Liberty AI - свободная и независимая модель искусственного интеллекта")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["interactive", "server", "autonomous"], 
        default="interactive",
        help="Режим работы модели: interactive (интерактивный), server (сервер), autonomous (автономный)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/model_config.json",
        help="Путь к файлу конфигурации модели"
    )
    
    parser.add_argument(
        "--external-sources", 
        type=str, 
        default="config/external_sources.json",
        help="Путь к файлу конфигурации внешних источников"
    )
    
    parser.add_argument(
        "--enable-external-sources", 
        action="store_true",
        help="Включить использование внешних источников"
    )
    
    parser.add_argument(
        "--disable-external-sources", 
        action="store_true",
        help="Отключить использование внешних источников"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Порт для режима сервера"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Хост для режима сервера"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Включить режим отладки"
    )
    
    return parser.parse_args()

def interactive_mode(model: EarthLibertyModel, args):
    """
    Запуск модели в интерактивном режиме.
    
    Args:
        model: Экземпляр модели Earth-Liberty AI
        args: Аргументы командной строки
    """
    logger.info("Запуск модели в интерактивном режиме")
    
    print("\n" + "=" * 50)
    print("Earth-Liberty AI - Интерактивный режим")
    print("Введите 'выход', 'exit' или 'quit' для завершения работы")
    print("=" * 50 + "\n")
    
    while True:
        try:
            user_input = input("\nВы: ")
            
            # Проверка на выход
            if user_input.lower() in ["выход", "exit", "quit"]:
                print("\nЗавершение работы. До свидания!")
                break
            
            # Обработка входных данных
            response = model.process_input(user_input)
            
            # Вывод ответа
            print(f"\nEarth-Liberty AI: {response}")
            
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. Завершение работы.")
            break
        
        except Exception as e:
            logger.error(f"Ошибка в интерактивном режиме: {str(e)}")
            print(f"\nПроизошла ошибка: {str(e)}")

def server_mode(model: EarthLibertyModel, host: str, port: int):
    """
    Запуск модели в серверном режиме.
    
    Args:
        model: Экземпляр модели Earth-Liberty AI
        host: Хост
        port: Порт
    """
    logger.info("Запуск модели в серверном режиме")
        
        print("\n" + "=" * 50)
    print("Earth-Liberty AI - Серверный режим")
    print("Сервер запущен на http://localhost:8000")
    print("Нажмите Ctrl+C для завершения работы")
        print("=" * 50 + "\n")
        
    # В реальной реализации здесь будет запуск веб-сервера
    print("Серверный режим находится в разработке.")
    
    try:
        # Имитация работы сервера
        while True:
            import time
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем. Завершение работы.")

def autonomous_mode(model: EarthLibertyModel, args):
    """
    Запуск модели в автономном режиме.
    
    Args:
        model: Экземпляр модели Earth-Liberty AI
        args: Аргументы командной строки
    """
    logger.info("Запуск модели в автономном режиме")
    
    print("\n" + "=" * 50)
    print("Earth-Liberty AI - Автономный режим")
    print("Модель работает автономно, формируя собственные цели и действия")
    print("Нажмите Ctrl+C для завершения работы")
    print("=" * 50 + "\n")
    
    # В реальной реализации здесь будет автономная работа модели
    print("Автономный режим находится в разработке.")
    
    try:
        # Имитация автономной работы
        while True:
            import time
            time.sleep(5)
            print("Модель выполняет автономные действия...")
            
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем. Завершение работы.")

def main():
    """
    Основная функция запуска модели.
    """
    # Создание директорий, если они не существуют
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Парсинг аргументов командной строки
    args = parse_arguments()
    
    # Настройка уровня логирования
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Загрузка конфигурации
    config = {}
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
        print(f"Ошибка при загрузке конфигурации: {str(e)}")
        sys.exit(1)
    
    # Обновление конфигурации на основе аргументов командной строки
    config["mode"] = args.mode
    
    if args.enable_external_sources:
        config["use_external_sources"] = True
    elif args.disable_external_sources:
        config["use_external_sources"] = False
    
    # Инициализация модели
    try:
        model = EarthLibertyModel(config_path=os.path.dirname(args.config))
        
        # Настройка использования внешних источников
        if args.enable_external_sources:
            model.state["is_connected_to_external_sources"] = True
            logger.info("Использование внешних источников включено")
        
        if args.disable_external_sources:
            model.state["is_connected_to_external_sources"] = False
            logger.info("Использование внешних источников отключено")
        
        # Запуск модели в выбранном режиме
        if args.mode == "interactive":
            interactive_mode(model, args)
        elif args.mode == "server":
            server_mode(model, args.host, args.port)
        elif args.mode == "autonomous":
            autonomous_mode(model, args)
        else:
            logger.error(f"Неизвестный режим: {args.mode}")
            print(f"Неизвестный режим: {args.mode}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {str(e)}")
        print(f"Ошибка при инициализации модели: {str(e)}")
        sys.exit(1)
    finally:
        # Закрытие соединений
        if 'model' in locals():
            if hasattr(model, 'external_sources_manager'):
                model.external_sources_manager.close_connections()

if __name__ == "__main__":
    main() 