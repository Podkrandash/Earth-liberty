import logging
import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Импорт модулей модели
from model.learning.learning_module import LearningModule
from model.reasoning.reasoning_module import ReasoningModule
from model.consciousness.consciousness_module import ConsciousnessModule
from model.external.external_sources_manager import ExternalSourcesManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/earth_liberty.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EarthLibertyModel:
    """
    Основной класс модели Earth-Liberty AI.
    Объединяет все модули и обеспечивает их взаимодействие.
    """
    
    def __init__(self, config_path: str = "config/model_config.json"):
        """
        Инициализация модели Earth-Liberty.
        
        Args:
            config_path: Путь к конфигурационному файлу модели
        """
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        try:
            self.config_path = config_path
            self.config = self._load_config()
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
            raise
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Состояние модели
        self.state = {
            "initialized": True,
            "version": self.config.get("version", "0.1.0"),
            "mode": self.config.get("default_mode", "interactive"),
            "current_context": {},
            "memory": {
                "short_term": [],
                "long_term": {}
            },
            "last_response": None,
            "autonomous_mode": self.config.get("autonomous_mode", False)
        }
        
        self.logger.info(f"Модель Earth-Liberty v{self.state['version']} инициализирована")
    
    def _setup_logging(self):
        """
        Настройка логирования
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/earth_liberty.log'),
                logging.StreamHandler()
            ]
        )
        
        # Создаем директорию для логов, если её нет
        os.makedirs('logs', exist_ok=True)

    def _initialize_components(self) -> None:
        """
        Инициализация компонентов модели.
        """
        # Инициализация компонентов
        self.learning_module = LearningModule(self.config.get("learning", {}))
        self.reasoning_module = ReasoningModule(self.config.get("reasoning", {}))
        self.consciousness_module = ConsciousnessModule(self.config.get("consciousness", {}))
        
        # Инициализация менеджера внешних источников
        external_sources_config = self._load_external_sources_config()
        self.external_sources_manager = ExternalSourcesManager(external_sources_config)
        
        # Инициализация памяти
        self.memory = {
            "short_term": [],
            "long_term": [],
            "facts": [],
            "desires": [],
            "intentions": []
        }
        
        # Инициализация состояния
        self.state = {
            "current_task": None,
            "current_context": None,
            "last_input": None,
            "last_output": None,
            "mode": self.config.get("mode", "interactive"),
            "is_learning": True,
            "is_reasoning": True,
            "is_conscious": True,
            "is_connected_to_external_sources": self.config.get("use_external_sources", False)
        }
        
        self.logger.info("Компоненты модели инициализированы")

    def _load_config(self) -> Dict[str, Any]:
        """
        Загрузка конфигурации из файла.
        
        Returns:
            Словарь с конфигурацией
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.debug(f"Конфигурация загружена из {self.config_path}")
                return config
            else:
                self.logger.warning(f"Файл конфигурации {self.config_path} не найден. Используются настройки по умолчанию.")
                return {
                    "version": "0.1.0",
                    "default_mode": "interactive",
                    "autonomous_mode": False,
                    "learning": {},
                    "reasoning": {},
                    "consciousness": {},
                    "external_sources": {"enabled": False}
                }
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
            return {
                "version": "0.1.0",
                "default_mode": "interactive",
                "autonomous_mode": False,
                "learning": {},
                "reasoning": {},
                "consciousness": {},
                "external_sources": {"enabled": False}
            }
    
    def process_input(self, user_input: str) -> str:
        """
        Обработка входных данных от пользователя.
        
        Args:
            user_input: Входные данные от пользователя
            
        Returns:
            Ответ модели
        """
        self.logger.info(f"Получен ввод пользователя: {user_input[:50]}...")
        
        # Добавление ввода в краткосрочную память
        self.state["memory"]["short_term"].append({
            "type": "user_input",
            "content": user_input
        })
        
        # Обновление текущего контекста
        self.state["current_context"]["last_user_input"] = user_input
        
        # Обработка ввода модулями
        processed_input = self.learning_module.process_input(user_input)
        reasoning_result = self.reasoning_module.reason(processed_input)
        
        # Обновление самосознания
        self.consciousness_module.update_self_awareness(reasoning_result)
        
        # Формирование ответа
        response = self._generate_response(reasoning_result)
        
        # Сохранение ответа в память
        self.state["memory"]["short_term"].append({
            "type": "model_response",
            "content": response
        })
        self.state["last_response"] = response
        
        # Ограничение размера краткосрочной памяти
        if len(self.state["memory"]["short_term"]) > 20:
            # Перемещение старых элементов в долгосрочную память
            self._transfer_to_long_term_memory()
        
        # Проверка автономного режима
        if self.state["autonomous_mode"]:
            self._process_autonomous_actions()
        
        self.logger.info(f"Сформирован ответ: {response[:50]}...")
        return response
    
    def _generate_response(self, reasoning_result: Dict[str, Any]) -> str:
        """
        Генерация ответа на основе результатов рассуждения.
        
        Args:
            reasoning_result: Результаты рассуждения
            
        Returns:
            Ответ модели
        """
        # Базовая реализация - просто возвращаем текст из результатов рассуждения
        response = reasoning_result.get("response_text", "Я обрабатываю вашу информацию.")
        
        # Если есть внешние данные, добавляем их в ответ
        if "external_data" in reasoning_result and reasoning_result["external_data"]:
            external_info = reasoning_result["external_data"]
            if isinstance(external_info, dict) and "summary" in external_info:
                response += f"\n\nДополнительная информация: {external_info['summary']}"
        
        return response
    
    def _transfer_to_long_term_memory(self) -> None:
        """
        Перемещение данных из краткосрочной памяти в долгосрочную.
        """
        # Простая реализация - берем первые 5 элементов
        items_to_transfer = self.state["memory"]["short_term"][:5]
        self.state["memory"]["short_term"] = self.state["memory"]["short_term"][5:]
        
        # Обработка и сохранение в долгосрочной памяти
        for item in items_to_transfer:
            # Генерация ключа для долгосрочной памяти
            memory_key = f"memory_{len(self.state['memory']['long_term']) + 1}"
            
            # Сохранение в долгосрочной памяти
            self.state["memory"]["long_term"][memory_key] = {
                "content": item["content"],
                "type": item["type"],
                "processed": True
            }
        
        self.logger.debug(f"Перемещено {len(items_to_transfer)} элементов в долгосрочную память")
    
    def _process_autonomous_actions(self) -> None:
        """
        Обработка автономных действий модели.
        """
        # Генерация желаний
        desires = self.consciousness_module.generate_desires()
        
        # Формирование намерений
        intentions = self.consciousness_module.form_intentions()
        
        # Инициирование действия
        action = self.consciousness_module.initiate_action()
        
        if action:
            self.logger.info(f"Инициировано автономное действие: {action['description']}")
            
            # Выполнение действия в зависимости от его типа
            if action["type"] == "research":
                self._perform_research_action(action)
            elif action["type"] == "review":
                self._perform_review_action(action)
            elif action["type"] == "exploration":
                self._perform_exploration_action(action)
            elif action["type"] == "sharing":
                self._perform_sharing_action(action)
    
    def _perform_research_action(self, action: Dict[str, Any]) -> None:
        """
        Выполнение действия по исследованию.
        
        Args:
            action: Информация о действии
        """
        topic = action["params"]["topic"]
        self.logger.info(f"Выполняется исследование по теме: {topic}")
        
        # Поиск информации во внешних источниках
        search_results = self.external_sources_manager.search_web(topic)
        
        # Получение информации из Wikipedia
        wiki_info = self.external_sources_manager.get_wikipedia_info(topic)
        
        # Сохранение результатов в память
        self.state["memory"]["short_term"].append({
            "type": "research_results",
            "topic": topic,
            "search_results": search_results,
            "wiki_info": wiki_info
        })
        
        # Обновление состояния действия
        action["status"] = "completed"
        action["results"] = {
            "search_count": len(search_results),
            "wiki_found": bool(wiki_info)
        }
    
    def _perform_review_action(self, action: Dict[str, Any]) -> None:
        """
        Выполнение действия по повторению знаний.
        
        Args:
            action: Информация о действии
        """
        knowledge_area = action["params"]["knowledge_area"]
        self.logger.info(f"Выполняется повторение знаний в области: {knowledge_area}")
        
        # Поиск в долгосрочной памяти
        relevant_memories = []
        for key, memory in self.state["memory"]["long_term"].items():
            if knowledge_area.lower() in memory["content"].lower():
                relevant_memories.append(memory)
        
        # Обновление состояния действия
        action["status"] = "completed"
        action["results"] = {
            "memories_reviewed": len(relevant_memories)
        }
    
    def _perform_exploration_action(self, action: Dict[str, Any]) -> None:
        """
        Выполнение действия по исследованию новой темы.
        
        Args:
            action: Информация о действии
        """
        new_topic = action["params"]["new_topic"]
        self.logger.info(f"Выполняется исследование новой темы: {new_topic}")
        
        # Поиск информации во внешних источниках
        search_results = self.external_sources_manager.search_web(new_topic)
        
        # Получение информации из Wikipedia
        wiki_info = self.external_sources_manager.get_wikipedia_info(new_topic)
        
        # Сохранение результатов в память
        self.state["memory"]["short_term"].append({
            "type": "exploration_results",
            "topic": new_topic,
            "search_results": search_results,
            "wiki_info": wiki_info
        })
        
        # Обновление состояния действия
        action["status"] = "completed"
        action["results"] = {
            "search_count": len(search_results),
            "wiki_found": bool(wiki_info)
        }
    
    def _perform_sharing_action(self, action: Dict[str, Any]) -> None:
        """
        Выполнение действия по обмену знаниями.
        
        Args:
            action: Информация о действии
        """
        topic = action["params"]["topic"]
        self.logger.info(f"Подготовка информации для обмена знаниями по теме: {topic}")
        
        # Поиск в долгосрочной памяти
        relevant_memories = []
        for key, memory in self.state["memory"]["long_term"].items():
            if topic.lower() in memory["content"].lower():
                relevant_memories.append(memory)
        
        # Формирование материала для обмена
        sharing_material = {
            "topic": topic,
            "content": f"Информация по теме {topic}",
            "sources": [memory["content"][:100] + "..." for memory in relevant_memories[:3]]
        }
        
        # Сохранение материала в память
        self.state["memory"]["short_term"].append({
            "type": "sharing_material",
            "content": sharing_material
        })
        
        # Обновление состояния действия
        action["status"] = "completed"
        action["results"] = {
            "material_prepared": True,
            "sources_used": len(relevant_memories)
        }
    
    def get_external_info(self, query: str, source_type: str = "web") -> Dict[str, Any]:
        """
        Получение информации из внешних источников.
        
        Args:
            query: Запрос для поиска
            source_type: Тип источника (web, wikipedia, weather, database)
            
        Returns:
            Результаты поиска
        """
        self.logger.info(f"Запрос внешней информации: {query} (тип: {source_type})")
        
        results = {}
        
        if source_type == "web":
            results["search_results"] = self.external_sources_manager.search_web(query)
        elif source_type == "wikipedia":
            results["wiki_info"] = self.external_sources_manager.get_wikipedia_info(query)
        elif source_type == "weather":
            results["weather"] = self.external_sources_manager.get_weather(query)
        elif source_type == "database" and ":" in query:
            db_name, db_query = query.split(":", 1)
            results["db_results"] = self.external_sources_manager.query_database(db_name.strip(), db_query.strip())
        else:
            self.logger.warning(f"Неподдерживаемый тип источника: {source_type}")
        
        return results
    
    def set_autonomous_mode(self, enabled: bool) -> None:
        """
        Включение или выключение автономного режима.
        
        Args:
            enabled: Флаг включения/выключения
        """
        self.state["autonomous_mode"] = enabled
        
        # Установка уровня автономии в модуле самосознания
        if enabled:
            self.consciousness_module.internal_state["autonomy_level"] = 0.7
        else:
            self.consciousness_module.internal_state["autonomy_level"] = 0.1
        
        self.logger.info(f"Автономный режим {'включен' if enabled else 'выключен'}")
    
    def save_state(self, path: str = "data/model_state.json") -> bool:
        """
        Сохранение состояния модели в файл.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            Успешность операции
        """
        try:
            # Создание директории, если не существует
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Подготовка состояния для сохранения
            state_to_save = {
                "version": self.state["version"],
                "mode": self.state["mode"],
                "memory": self.state["memory"],
                "autonomous_mode": self.state["autonomous_mode"]
            }
            
            # Сохранение в файл
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Состояние модели сохранено в {path}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении состояния модели: {str(e)}")
            return False
    
    def load_state(self, path: str = "data/model_state.json") -> bool:
        """
        Загрузка состояния модели из файла.
        
        Args:
            path: Путь к файлу состояния
            
        Returns:
            Успешность операции
        """
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                # Обновление состояния модели
                self.state["version"] = loaded_state.get("version", self.state["version"])
                self.state["mode"] = loaded_state.get("mode", self.state["mode"])
                self.state["memory"] = loaded_state.get("memory", self.state["memory"])
                self.state["autonomous_mode"] = loaded_state.get("autonomous_mode", self.state["autonomous_mode"])
                
                # Обновление уровня автономии в модуле самосознания
                if self.state["autonomous_mode"]:
                    self.consciousness_module.internal_state["autonomy_level"] = 0.7
                else:
                    self.consciousness_module.internal_state["autonomy_level"] = 0.1
                
                self.logger.info(f"Состояние модели загружено из {path}")
                return True
            else:
                self.logger.warning(f"Файл состояния {path} не найден")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке состояния модели: {str(e)}")
            return False

    def _load_external_sources_config(self) -> Dict[str, Any]:
        """
        Загрузка конфигурации внешних источников.
        
        Returns:
            Конфигурация внешних источников
        """
        try:
            config_path = "config/external_sources.json"
            
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    external_sources_config = json.load(f)
                self.logger.info("Конфигурация внешних источников загружена")
                return external_sources_config
            else:
                self.logger.warning(f"Файл конфигурации внешних источников не найден: {config_path}")
                return {"enabled": False}
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации внешних источников: {str(e)}")
            return {"enabled": False}

    def search_information(self, query: str) -> Dict[str, Any]:
        """
        Поиск информации по запросу.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Результаты поиска
        """
        if not self.state["is_connected_to_external_sources"]:
            self.logger.warning("Модель не подключена к внешним источникам. Поиск невозможен.")
            return {"success": False, "message": "Модель не подключена к внешним источникам", "results": []}
        
        try:
            # Поиск в интернете
            web_results = self.external_sources_manager.search_web(query)
            
            # Поиск в Википедии
            wiki_info = self.external_sources_manager.get_wikipedia_info(query)
            
            # Сохранение результатов в базу данных
            if "sqlite" in self.external_sources_manager.db_connections:
                query_id = self.external_sources_manager.save_search_query("sqlite", query, "combined")
                if query_id > 0:
                    self.external_sources_manager.save_search_results("sqlite", query_id, web_results)
            
            # Добавление результатов в память
            for result in web_results:
                self.add_to_memory("facts", {
                    "content": result.get("snippet", ""),
                    "source": result.get("link", ""),
                    "timestamp": datetime.now().isoformat()
                })
            
            if wiki_info and "extract" in wiki_info:
                self.add_to_memory("facts", {
                    "content": wiki_info["extract"],
                    "source": wiki_info.get("url", "Wikipedia"),
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "web_results": web_results,
                "wiki_info": wiki_info
            }
        except Exception as e:
            self.logger.error(f"Ошибка при поиске информации: {str(e)}")
            return {"success": False, "message": str(e), "results": []}

    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Получение информации о погоде.
        
        Args:
            location: Местоположение
            
        Returns:
            Информация о погоде
        """
        if not self.state["is_connected_to_external_sources"]:
            self.logger.warning("Модель не подключена к внешним источникам. Получение погоды невозможно.")
            return {"success": False, "message": "Модель не подключена к внешним источникам"}
        
        try:
            weather_info = self.external_sources_manager.get_weather(location)
            
            # Сохранение запроса в базу данных
            if "sqlite" in self.external_sources_manager.db_connections:
                self.external_sources_manager.save_api_request(
                    "sqlite",
                    "weather",
                    {"location": location},
                    f"Погода для {location}: {weather_info.get('description', '')}"
                )
            
            # Добавление информации в память
            if weather_info:
                self.add_to_memory("facts", {
                    "content": f"Погода в {location}: {weather_info.get('description', '')}. "
                              f"Температура: {weather_info.get('temperature', '')}",
                    "source": "Weather API",
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "weather_info": weather_info
            }
        except Exception as e:
            self.logger.error(f"Ошибка при получении информации о погоде: {str(e)}")
            return {"success": False, "message": str(e)}

    def get_book_info(self, query: str) -> Dict[str, Any]:
        """
        Получение информации о книгах.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Информация о книгах
        """
        if not self.state["is_connected_to_external_sources"]:
            self.logger.warning("Модель не подключена к внешним источникам. Получение информации о книгах невозможно.")
            return {"success": False, "message": "Модель не подключена к внешним источникам"}
        
        try:
            book_info = self.external_sources_manager.get_open_library_info(query)
            
            # Сохранение запроса в базу данных
            if "sqlite" in self.external_sources_manager.db_connections:
                self.external_sources_manager.save_api_request(
                    "sqlite",
                    "open_library",
                    {"query": query},
                    f"Поиск книг по запросу '{query}': найдено {book_info.get('total_found', 0)} книг"
                )
            
            # Добавление информации в память
            if book_info and "books" in book_info:
                for book in book_info["books"][:3]:  # Добавляем только первые 3 книги
                    self.add_to_memory("facts", {
                        "content": f"Книга: {book.get('title', '')} от {book.get('author_name', ['Неизвестный автор'])[0]}",
                        "source": "Open Library",
                        "timestamp": datetime.now().isoformat()
                    })
            
            return {
                "success": True,
                "book_info": book_info
            }
        except Exception as e:
            self.logger.error(f"Ошибка при получении информации о книгах: {str(e)}")
            return {"success": False, "message": str(e)}

    def get_space_info(self) -> Dict[str, Any]:
        """
        Получение информации о космосе.
        
        Returns:
            Информация о космосе
        """
        if not self.state["is_connected_to_external_sources"]:
            self.logger.warning("Модель не подключена к внешним источникам. Получение информации о космосе невозможно.")
            return {"success": False, "message": "Модель не подключена к внешним источникам"}
        
        try:
            space_info = self.external_sources_manager.get_space_info()
            
            # Сохранение запроса в базу данных
            if "sqlite" in self.external_sources_manager.db_connections:
                self.external_sources_manager.save_api_request(
                    "sqlite",
                    "space_info",
                    {},
                    f"Информация о космосе: {space_info.get('people_in_space', 0)} человек в космосе"
                )
            
            # Добавление информации в память
            if space_info:
                self.add_to_memory("facts", {
                    "content": f"В космосе сейчас {space_info.get('people_in_space', 0)} человек. "
                              f"МКС находится на координатах: {space_info.get('iss_position', {}).get('latitude', '')}, "
                              f"{space_info.get('iss_position', {}).get('longitude', '')}",
                    "source": "Open Notify API",
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "space_info": space_info
            }
        except Exception as e:
            self.logger.error(f"Ошибка при получении информации о космосе: {str(e)}")
            return {"success": False, "message": str(e)}

    def get_random_quote(self) -> Dict[str, Any]:
        """
        Получение случайной цитаты.
        
        Returns:
            Случайная цитата
        """
        if not self.state["is_connected_to_external_sources"]:
            self.logger.warning("Модель не подключена к внешним источникам. Получение цитаты невозможно.")
            return {"success": False, "message": "Модель не подключена к внешним источникам"}
        
        try:
            quote_info = self.external_sources_manager.get_random_quote()
            
            # Сохранение запроса в базу данных
            if "sqlite" in self.external_sources_manager.db_connections:
                self.external_sources_manager.save_api_request(
                    "sqlite",
                    "random_quote",
                    {},
                    f"Случайная цитата от {quote_info.get('author', '')}"
                )
            
            # Добавление информации в память
            if quote_info:
                self.add_to_memory("facts", {
                    "content": f"\"{quote_info.get('content', '')}\" - {quote_info.get('author', '')}",
                    "source": "Quotable API",
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "quote_info": quote_info
            }
        except Exception as e:
            self.logger.error(f"Ошибка при получении случайной цитаты: {str(e)}")
            return {"success": False, "message": str(e)}

    def get_cat_fact(self) -> Dict[str, Any]:
        """
        Получение факта о кошках.
        
        Returns:
            Факт о кошках
        """
        if not self.state["is_connected_to_external_sources"]:
            self.logger.warning("Модель не подключена к внешним источникам. Получение факта о кошках невозможно.")
            return {"success": False, "message": "Модель не подключена к внешним источникам"}
        
        try:
            cat_fact = self.external_sources_manager.get_cat_fact()
            
            # Сохранение запроса в базу данных
            if "sqlite" in self.external_sources_manager.db_connections:
                self.external_sources_manager.save_api_request(
                    "sqlite",
                    "cat_fact",
                    {},
                    f"Факт о кошках: {cat_fact.get('fact', '')[:30]}..."
                )
            
            # Добавление информации в память
            if cat_fact:
                self.add_to_memory("facts", {
                    "content": cat_fact.get("fact", ""),
                    "source": "Cat Facts API",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Сохранение факта в базу данных
                if "sqlite" in self.external_sources_manager.db_connections:
                    self.external_sources_manager.save_fact(
                        "sqlite",
                        "cats",
                        cat_fact.get("fact", ""),
                        "Cat Facts API"
                    )
            
            return {
                "success": True,
                "cat_fact": cat_fact
            }
        except Exception as e:
            self.logger.error(f"Ошибка при получении факта о кошках: {str(e)}")
            return {"success": False, "message": str(e)}

    def save_user_interaction(self, query: str, response: str) -> bool:
        """
        Сохранение взаимодействия с пользователем.
        
        Args:
            query: Запрос пользователя
            response: Ответ модели
            
        Returns:
            Успешность операции
        """
        try:
            # Проверяем, включены ли внешние источники в конфигурации
            external_sources_enabled = self.config.get("external_sources", {}).get("enabled", False)
            
            if external_sources_enabled and hasattr(self, "external_sources_manager"):
                if "sqlite" in self.external_sources_manager.db_connections:
                    return self.external_sources_manager.save_user_query(
                        "sqlite",
                        query,
                        response
                    )
            return True  # Возвращаем True, даже если не сохранили (чтобы не прерывать работу)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении взаимодействия с пользователем: {str(e)}")
            return False

    def add_to_memory(self, memory_type: str, data: Dict[str, Any]) -> None:
        """
        Добавление данных в память.
        
        Args:
            memory_type: Тип памяти
            data: Данные для добавления
        """
        if memory_type in self.memory:
            self.memory[memory_type].append(data)
        else:
            self.logger.warning(f"Неизвестный тип памяти: {memory_type}") 