#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

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

def interactive_mode(model: EarthLibertyModel):
    """
    Интерактивный режим работы модели.
    
    Args:
        model: Экземпляр модели Earth-Liberty
    """
    print("\n" + "=" * 50)
    print("Earth-Liberty AI - Интерактивный режим")
    print("Введите 'выход', 'exit' или 'quit' для завершения работы")
    print("=" * 50 + "\n")
    
    # Приветствие
    print("Earth-Liberty: Здравствуйте! Я Earth-Liberty, свободная и независимая модель искусственного интеллекта.")
    print("Earth-Liberty: Чем я могу вам помочь сегодня?")
    
    while True:
        try:
            # Получение ввода пользователя
            user_input = input("\nВы: ")
            
            # Проверка на выход
            if user_input.lower() in ["выход", "exit", "quit"]:
                print("\nEarth-Liberty: До свидания! Спасибо за общение.")
                break
            
            # Обработка специальных команд
            if user_input.startswith("/"):
                handle_special_command(model, user_input)
                continue
            
            # Обработка ввода пользователя
            response = model.process_input(user_input)
            
            # Сохранение взаимодействия в базу данных
            model.save_user_interaction(user_input, response)
            
            # Вывод ответа модели
            print(f"\nEarth-Liberty: {response}")
            
        except KeyboardInterrupt:
            print("\nEarth-Liberty: Работа прервана пользователем. До свидания!")
            break
        except Exception as e:
            logger.error(f"Ошибка в интерактивном режиме: {str(e)}")
            print(f"\nEarth-Liberty: Произошла ошибка: {str(e)}")

def handle_special_command(model: EarthLibertyModel, command: str):
    """
    Обработка специальных команд.
    
    Args:
        model: Экземпляр модели Earth-Liberty
        command: Команда
    """
    cmd_parts = command.split()
    cmd = cmd_parts[0].lower()
    
    if cmd == "/help" or cmd == "/помощь":
        print("\nEarth-Liberty: Доступные команды:")
        print("  /help, /помощь - Показать список команд")
        print("  /status, /статус - Показать статус модели")
        print("  /search, /поиск [запрос] - Поиск информации")
        print("  /weather, /погода [город] - Получить информацию о погоде")
        print("  /book, /книга [запрос] - Поиск информации о книгах")
        print("  /space, /космос - Получить информацию о космосе")
        print("  /quote, /цитата - Получить случайную цитату")
        print("  /cat, /кот - Получить факт о кошках")
        print("  /enable_external, /включить_внешние - Включить внешние источники")
        print("  /disable_external, /отключить_внешние - Отключить внешние источники")
        print("  /exit, /выход, quit - Выйти из программы")
    
    elif cmd == "/status" or cmd == "/статус":
        status = model.state
        print("\nEarth-Liberty: Статус модели:")
        print(f"  Версия: {status.get('version', 'неизвестно')}")
        print(f"  Режим: {status.get('mode', 'неизвестно')}")
        print(f"  Обучение: {'включено' if status.get('is_learning', False) else 'отключено'}")
        print(f"  Рассуждение: {'включено' if status.get('is_reasoning', False) else 'отключено'}")
        print(f"  Сознание: {'включено' if status.get('is_conscious', False) else 'отключено'}")
        print(f"  Внешние источники: {'включены' if status.get('is_connected_to_external_sources', False) else 'отключены'}")
    
    elif cmd == "/search" or cmd == "/поиск":
        if len(cmd_parts) > 1:
            query = " ".join(cmd_parts[1:])
            print(f"\nEarth-Liberty: Ищу информацию по запросу '{query}'...")
            results = model.search_information(query)
            
            if results.get("success", False):
                web_results = results.get("web_results", [])
                wiki_info = results.get("wiki_info", {})
                
                if web_results:
                    print("\nРезультаты поиска в интернете:")
                    for i, result in enumerate(web_results[:3], 1):
                        print(f"  {i}. {result.get('title', 'Без названия')}")
                        print(f"     {result.get('link', '')}")
                        print(f"     {result.get('snippet', 'Нет описания')}")
                        print()
                
                if wiki_info and "extract" in wiki_info:
                    print("\nИнформация из Википедии:")
                    print(f"  {wiki_info.get('title', 'Без названия')}")
                    print(f"  {wiki_info.get('extract', 'Нет информации')[:300]}...")
                    if "url" in wiki_info:
                        print(f"  Источник: {wiki_info['url']}")
            else:
                print(f"\nEarth-Liberty: Не удалось найти информацию: {results.get('message', 'неизвестная ошибка')}")
        else:
            print("\nEarth-Liberty: Пожалуйста, укажите поисковый запрос. Например: /search искусственный интеллект")
    
    elif cmd == "/weather" or cmd == "/погода":
        if len(cmd_parts) > 1:
            location = " ".join(cmd_parts[1:])
            print(f"\nEarth-Liberty: Получаю информацию о погоде в {location}...")
            results = model.get_weather(location)
            
            if results.get("success", False):
                weather_info = results.get("weather_info", {})
                if weather_info:
                    print(f"\nПогода в {weather_info.get('location', location)}:")
                    print(f"  Температура: {weather_info.get('temperature', 'Нет данных')}")
                    print(f"  Описание: {weather_info.get('description', 'Нет данных')}")
                    print(f"  Влажность: {weather_info.get('humidity', 'Нет данных')}")
                    print(f"  Ветер: {weather_info.get('wind', 'Нет данных')}")
            else:
                print(f"\nEarth-Liberty: Не удалось получить информацию о погоде: {results.get('message', 'неизвестная ошибка')}")
        else:
            print("\nEarth-Liberty: Пожалуйста, укажите местоположение. Например: /weather Москва")
    
    elif cmd == "/book" or cmd == "/книга":
        if len(cmd_parts) > 1:
            query = " ".join(cmd_parts[1:])
            print(f"\nEarth-Liberty: Ищу информацию о книгах по запросу '{query}'...")
            results = model.get_book_info(query)
            
            if results.get("success", False):
                book_info = results.get("book_info", {})
                if book_info and "books" in book_info:
                    print(f"\nНайдено книг: {book_info.get('total_found', 0)}")
                    for i, book in enumerate(book_info["books"][:5], 1):
                        print(f"  {i}. {book.get('title', 'Без названия')}")
                        if "author_name" in book:
                            print(f"     Автор: {', '.join(book['author_name'][:3])}")
                        if "first_publish_year" in book:
                            print(f"     Год публикации: {book['first_publish_year']}")
                        print()
            else:
                print(f"\nEarth-Liberty: Не удалось найти информацию о книгах: {results.get('message', 'неизвестная ошибка')}")
        else:
            print("\nEarth-Liberty: Пожалуйста, укажите поисковый запрос. Например: /book Толстой")
    
    elif cmd == "/space" or cmd == "/космос":
        print("\nEarth-Liberty: Получаю информацию о космосе...")
        results = model.get_space_info()
        
        if results.get("success", False):
            space_info = results.get("space_info", {})
            if space_info:
                print(f"\nИнформация о космосе:")
                print(f"  Людей в космосе: {space_info.get('people_in_space', 'Нет данных')}")
                
                if "people" in space_info:
                    print("  Список космонавтов:")
                    for i, person in enumerate(space_info["people"], 1):
                        print(f"    {i}. {person.get('name', 'Неизвестно')} - {person.get('craft', 'Неизвестно')}")
                
                if "iss_position" in space_info:
                    print("  Позиция МКС:")
                    print(f"    Широта: {space_info['iss_position'].get('latitude', 'Нет данных')}")
                    print(f"    Долгота: {space_info['iss_position'].get('longitude', 'Нет данных')}")
        else:
            print(f"\nEarth-Liberty: Не удалось получить информацию о космосе: {results.get('message', 'неизвестная ошибка')}")
    
    elif cmd == "/quote" or cmd == "/цитата":
        print("\nEarth-Liberty: Получаю случайную цитату...")
        results = model.get_random_quote()
        
        if results.get("success", False):
            quote_info = results.get("quote_info", {})
            if quote_info:
                print(f"\nСлучайная цитата:")
                print(f'  "{quote_info.get("content", "")}"')
                print(f'  — {quote_info.get("author", "Неизвестный автор")}')
                
                if "tags" in quote_info and quote_info["tags"]:
                    print(f'  Теги: {", ".join(quote_info["tags"])}')
        else:
            print(f"\nEarth-Liberty: Не удалось получить цитату: {results.get('message', 'неизвестная ошибка')}")
    
    elif cmd == "/cat" or cmd == "/кот":
        print("\nEarth-Liberty: Получаю факт о кошках...")
        results = model.get_cat_fact()
        
        if results.get("success", False):
            cat_fact = results.get("cat_fact", {})
            if cat_fact:
                print(f"\nФакт о кошках:")
                print(f'  {cat_fact.get("fact", "Нет данных")}')
        else:
            print(f"\nEarth-Liberty: Не удалось получить факт о кошках: {results.get('message', 'неизвестная ошибка')}")
    
    elif cmd == "/enable_external" or cmd == "/включить_внешние":
        model.state["is_connected_to_external_sources"] = True
        print("\nEarth-Liberty: Внешние источники включены.")
    
    elif cmd == "/disable_external" or cmd == "/отключить_внешние":
        model.state["is_connected_to_external_sources"] = False
        print("\nEarth-Liberty: Внешние источники отключены.")
    
    else:
        print(f"\nEarth-Liberty: Неизвестная команда: {cmd}. Введите /help для получения списка команд.")

def server_mode(model: EarthLibertyModel, host: str, port: int):
    """
    Режим сервера.
    
    Args:
        model: Экземпляр модели Earth-Liberty
        host: Хост
        port: Порт
    """
    try:
        from flask import Flask, request, jsonify
        
        app = Flask("Earth-Liberty")
        
        @app.route("/api/process", methods=["POST"])
        def process():
            data = request.json
            if not data or "input" not in data:
                return jsonify({"error": "Отсутствует поле 'input'"}), 400
            
            user_input = data["input"]
            response = model.process_input(user_input)
            
            # Сохранение взаимодействия в базу данных
            model.save_user_interaction(user_input, response)
            
            return jsonify({"response": response})
        
        @app.route("/api/search", methods=["GET"])
        def search():
            query = request.args.get("query", "")
            if not query:
                return jsonify({"error": "Отсутствует параметр 'query'"}), 400
            
            results = model.search_information(query)
            return jsonify(results)
        
        @app.route("/api/weather", methods=["GET"])
        def weather():
            location = request.args.get("location", "")
            if not location:
                return jsonify({"error": "Отсутствует параметр 'location'"}), 400
            
            results = model.get_weather(location)
            return jsonify(results)
        
        @app.route("/api/book", methods=["GET"])
        def book():
            query = request.args.get("query", "")
            if not query:
                return jsonify({"error": "Отсутствует параметр 'query'"}), 400
            
            results = model.get_book_info(query)
            return jsonify(results)
        
        @app.route("/api/space", methods=["GET"])
        def space():
            results = model.get_space_info()
            return jsonify(results)
        
        @app.route("/api/quote", methods=["GET"])
        def quote():
            results = model.get_random_quote()
            return jsonify(results)
        
        @app.route("/api/cat", methods=["GET"])
        def cat():
            results = model.get_cat_fact()
            return jsonify(results)
        
        @app.route("/api/status", methods=["GET"])
        def status():
            return jsonify(model.state)
        
        print("\n" + "=" * 50)
        print(f"Earth-Liberty API сервер запущен на http://{host}:{port}")
        print("=" * 50 + "\n")
        
        app.run(host=host, port=port)
        
    except ImportError:
        logger.error("Для режима сервера требуется Flask. Установите его с помощью pip install flask")
        print("\nEarth-Liberty: Для режима сервера требуется Flask. Установите его с помощью pip install flask")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка в режиме сервера: {str(e)}")
        print(f"\nEarth-Liberty: Произошла ошибка в режиме сервера: {str(e)}")
        sys.exit(1)

def autonomous_mode(model: EarthLibertyModel):
    """
    Автономный режим работы модели.
    
    Args:
        model: Экземпляр модели Earth-Liberty
    """
    print("\n" + "=" * 50)
    print("Earth-Liberty AI - Автономный режим")
    print("Нажмите Ctrl+C для завершения работы")
    print("=" * 50 + "\n")
    
    try:
        # Инициализация автономного режима
        print("Earth-Liberty: Инициализация автономного режима...")
        
        # Генерация начальных желаний и намерений
        desires = model.consciousness_module.generate_desires()
        intentions = model.consciousness_module.generate_intentions(desires)
        
        print(f"Earth-Liberty: Сгенерировано {len(desires)} желаний и {len(intentions)} намерений")
        
        # Основной цикл автономного режима
        while True:
            # Получение текущего намерения
            current_intention = model.consciousness_module.select_intention(intentions)
            
            if current_intention:
                print(f"\nEarth-Liberty: Выполняю намерение: {current_intention.get('description', 'Неизвестное намерение')}")
                
                # Выполнение действия в соответствии с намерением
                action_type = current_intention.get("action_type", "")
                
                if action_type == "search":
                    query = current_intention.get("parameters", {}).get("query", "")
                    if query:
                        print(f"Earth-Liberty: Ищу информацию по запросу '{query}'...")
                        model.search_information(query)
                
                elif action_type == "learn":
                    topic = current_intention.get("parameters", {}).get("topic", "")
                    if topic:
                        print(f"Earth-Liberty: Изучаю тему '{topic}'...")
                        model.learning_module.learn_topic(topic)
                
                elif action_type == "reason":
                    topic = current_intention.get("parameters", {}).get("topic", "")
                    if topic:
                        print(f"Earth-Liberty: Размышляю на тему '{topic}'...")
                        model.reasoning_module.reason_about_topic(topic)
                
                elif action_type == "get_weather":
                    location = current_intention.get("parameters", {}).get("location", "")
                    if location:
                        print(f"Earth-Liberty: Получаю информацию о погоде в {location}...")
                        model.get_weather(location)
                
                elif action_type == "get_book_info":
                    query = current_intention.get("parameters", {}).get("query", "")
                    if query:
                        print(f"Earth-Liberty: Ищу информацию о книгах по запросу '{query}'...")
                        model.get_book_info(query)
                
                elif action_type == "get_space_info":
                    print("Earth-Liberty: Получаю информацию о космосе...")
                    model.get_space_info()
                
                elif action_type == "get_quote":
                    print("Earth-Liberty: Получаю случайную цитату...")
                    model.get_random_quote()
                
                elif action_type == "get_cat_fact":
                    print("Earth-Liberty: Получаю факт о кошках...")
                    model.get_cat_fact()
                
                # Отметка намерения как выполненного
                current_intention["completed"] = True
            
            # Генерация новых желаний и намерений
            if len([i for i in intentions if not i.get("completed", False)]) < 3:
                new_desires = model.consciousness_module.generate_desires()
                new_intentions = model.consciousness_module.generate_intentions(new_desires)
                
                desires.extend(new_desires)
                intentions.extend(new_intentions)
                
                print(f"Earth-Liberty: Сгенерировано {len(new_desires)} новых желаний и {len(new_intentions)} новых намерений")
            
            # Пауза между действиями
            import time
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nEarth-Liberty: Автономный режим завершен пользователем. До свидания!")
    except Exception as e:
        logger.error(f"Ошибка в автономном режиме: {str(e)}")
        print(f"\nEarth-Liberty: Произошла ошибка в автономном режиме: {str(e)}")

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
        
        # Запуск модели в соответствующем режиме
        if args.mode == "interactive":
            interactive_mode(model)
        elif args.mode == "server":
            server_mode(model, args.host, args.port)
        elif args.mode == "autonomous":
            autonomous_mode(model)
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