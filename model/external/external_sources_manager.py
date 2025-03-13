import json
import logging
import os
import requests
from typing import Dict, List, Any, Optional, Union

# Настройка логирования
logger = logging.getLogger(__name__)

class ExternalSourcesManager:
    """
    Менеджер внешних источников информации для модели Earth-Liberty.
    Обеспечивает доступ к поисковым системам, API и базам данных.
    """
    
    def __init__(self, config_path: str = "config/external_sources.json"):
        """
        Инициализация менеджера внешних источников.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.enabled = self.config.get("enabled", False)
        self.search_engines = self.config.get("search_engines", {})
        self.apis = self.config.get("apis", {})
        self.databases = self.config.get("databases", {})
        
        # Инициализация соединений с базами данных
        self.db_connections = {}
        if self.enabled:
            self._initialize_db_connections()
        
        logger.info(f"Менеджер внешних источников инициализирован. Статус: {'включен' if self.enabled else 'выключен'}")
    
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
                logger.debug(f"Конфигурация загружена из {self.config_path}")
                return config
            else:
                logger.warning(f"Файл конфигурации {self.config_path} не найден. Используются настройки по умолчанию.")
                return {"enabled": False}
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
            return {"enabled": False}
    
    def _initialize_db_connections(self) -> None:
        """
        Инициализация соединений с базами данных.
        """
        for db_name, db_config in self.databases.items():
            if db_config.get("enabled", False):
                try:
                    if db_config["type"] == "sqlite":
                        self._connect_sqlite(db_name, db_config)
                    else:
                        logger.warning(f"Неподдерживаемый тип базы данных: {db_config.get('type')}")
                except Exception as e:
                    logger.error(f"Ошибка при подключении к базе данных {db_name}: {str(e)}")
    
    def _connect_sqlite(self, db_name: str, db_config: Dict[str, Any]) -> None:
        """
        Подключение к SQLite.
        
        Args:
            db_name: Имя базы данных
            db_config: Конфигурация базы данных
        """
        try:
            import sqlite3
            
            # Получение пути к файлу базы данных
            db_path = db_config.get("connection_string", "data/earth_liberty.db")
            
            # Создание директории, если она не существует
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
            # Подключение к базе данных
            conn = sqlite3.connect(db_path)
            
            # Настройка для получения результатов в виде словарей
            conn.row_factory = sqlite3.Row
            
            self.db_connections[db_name] = {
                "connection": conn,
                "cursor": conn.cursor()
            }
            
            # Инициализация базы данных (создание таблиц, если они не существуют)
            self._initialize_sqlite_db(conn)
            
            logger.info(f"Подключение к SQLite {db_name} установлено")
        except ImportError:
            logger.error("Модуль sqlite3 не установлен. Он должен быть включен в стандартную библиотеку Python.")
        except Exception as e:
            logger.error(f"Ошибка при подключении к SQLite {db_name}: {str(e)}")
    
    def _initialize_sqlite_db(self, conn) -> None:
        """
        Инициализация базы данных SQLite (создание таблиц).
        
        Args:
            conn: Соединение с базой данных
        """
        cursor = conn.cursor()
        
        # Создание таблицы для хранения поисковых запросов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            engine TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Создание таблицы для хранения результатов поиска
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            title TEXT,
            link TEXT,
            snippet TEXT,
            source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES search_queries (id)
        )
        ''')
        
        # Создание таблицы для хранения информации из API
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_name TEXT NOT NULL,
            request_params TEXT,
            response_summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Создание таблицы для хранения пользовательских запросов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Создание таблицы для хранения фактов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
    
    def query_database(self, db_name: str, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Выполнение запроса к базе данных.
        
        Args:
            db_name: Имя базы данных
            query: SQL запрос
            params: Параметры запроса
            
        Returns:
            Результаты запроса
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Запрос к базе данных невозможен.")
            return []
        
        if db_name not in self.db_connections:
            logger.warning(f"База данных {db_name} не подключена.")
            return []
        
        db_config = self.databases.get(db_name, {})
        db_type = db_config.get("type")
        
        try:
            if db_type == "sqlite":
                return self._query_sqlite(db_name, query, params)
            else:
                logger.warning(f"Неподдерживаемый тип базы данных: {db_type}")
                return []
        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса к базе данных {db_name}: {str(e)}")
            return []
    
    def _query_sqlite(self, db_name: str, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Выполнение запроса к SQLite.
        
        Args:
            db_name: Имя базы данных
            query: SQL запрос
            params: Параметры запроса
            
        Returns:
            Результаты запроса
        """
        if params is None:
            params = []
        
        cursor = self.db_connections[db_name]["cursor"]
        cursor.execute(query, params)
        
        # Получение результатов
        results = []
        for row in cursor.fetchall():
            # Преобразование объекта Row в словарь
            result = {}
            for key in row.keys():
                result[key] = row[key]
            results.append(result)
        
        return results
    
    def save_search_query(self, db_name: str, query: str, engine: str) -> int:
        """
        Сохранение поискового запроса в базу данных.
        
        Args:
            db_name: Имя базы данных
            query: Поисковый запрос
            engine: Имя поисковой системы
            
        Returns:
            ID сохраненного запроса
        """
        if not self.enabled or db_name not in self.db_connections:
            return -1
        
        try:
            cursor = self.db_connections[db_name]["cursor"]
            cursor.execute(
                "INSERT INTO search_queries (query, engine) VALUES (?, ?)",
                (query, engine)
            )
            self.db_connections[db_name]["connection"].commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Ошибка при сохранении поискового запроса: {str(e)}")
            return -1
    
    def save_search_results(self, db_name: str, query_id: int, results: List[Dict[str, str]]) -> bool:
        """
        Сохранение результатов поиска в базу данных.
        
        Args:
            db_name: Имя базы данных
            query_id: ID поискового запроса
            results: Результаты поиска
            
        Returns:
            Успешность операции
        """
        if not self.enabled or db_name not in self.db_connections or query_id < 0:
            return False
        
        try:
            cursor = self.db_connections[db_name]["cursor"]
            
            for result in results:
                cursor.execute(
                    "INSERT INTO search_results (query_id, title, link, snippet, source) VALUES (?, ?, ?, ?, ?)",
                    (
                        query_id,
                        result.get("title", ""),
                        result.get("link", ""),
                        result.get("snippet", ""),
                        result.get("source", "")
                    )
                )
            
            self.db_connections[db_name]["connection"].commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов поиска: {str(e)}")
            return False
    
    def save_api_request(self, db_name: str, api_name: str, request_params: Dict[str, Any], response_summary: str) -> bool:
        """
        Сохранение запроса к API в базу данных.
        
        Args:
            db_name: Имя базы данных
            api_name: Имя API
            request_params: Параметры запроса
            response_summary: Краткое описание ответа
            
        Returns:
            Успешность операции
        """
        if not self.enabled or db_name not in self.db_connections:
            return False
        
        try:
            cursor = self.db_connections[db_name]["cursor"]
            
            cursor.execute(
                "INSERT INTO api_requests (api_name, request_params, response_summary) VALUES (?, ?, ?)",
                (
                    api_name,
                    json.dumps(request_params),
                    response_summary
                )
            )
            
            self.db_connections[db_name]["connection"].commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении запроса к API: {str(e)}")
            return False
    
    def save_user_query(self, db_name: str, query: str, response: str) -> bool:
        """
        Сохранение пользовательского запроса в базу данных.
        
        Args:
            db_name: Имя базы данных
            query: Запрос пользователя
            response: Ответ модели
            
        Returns:
            Успешность операции
        """
        if not self.enabled or db_name not in self.db_connections:
            return False
        
        try:
            cursor = self.db_connections[db_name]["cursor"]
            
            cursor.execute(
                "INSERT INTO user_queries (query, response) VALUES (?, ?)",
                (query, response)
            )
            
            self.db_connections[db_name]["connection"].commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении пользовательского запроса: {str(e)}")
            return False
    
    def save_fact(self, db_name: str, category: str, content: str, source: str = "") -> bool:
        """
        Сохранение факта в базу данных.
        
        Args:
            db_name: Имя базы данных
            category: Категория факта
            content: Содержание факта
            source: Источник факта
            
        Returns:
            Успешность операции
        """
        if not self.enabled or db_name not in self.db_connections:
            return False
        
        try:
            cursor = self.db_connections[db_name]["cursor"]
            
            cursor.execute(
                "INSERT INTO facts (category, content, source) VALUES (?, ?, ?)",
                (category, content, source)
            )
            
            self.db_connections[db_name]["connection"].commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении факта: {str(e)}")
            return False
    
    def search_web(self, query: str, engine: str = None, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Поиск информации в интернете.
        
        Args:
            query: Поисковый запрос
            engine: Имя поисковой системы (если None, используется первая доступная)
            max_results: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Поиск невозможен.")
            return []
        
        # Выбор поисковой системы
        if engine is None:
            for engine_name, engine_config in self.search_engines.items():
                if engine_config.get("enabled", False):
                    engine = engine_name
                    break
        
        if engine not in self.search_engines or not self.search_engines[engine].get("enabled", False):
            logger.warning(f"Поисковая система {engine} недоступна или отключена.")
            return []
        
        engine_config = self.search_engines[engine]
        
        try:
            if engine == "duckduckgo":
                return self._search_duckduckgo(query, engine_config, max_results)
            elif engine == "wikipedia_search":
                return self._search_wikipedia(query, engine_config, max_results)
            else:
                logger.warning(f"Неподдерживаемая поисковая система: {engine}")
                return []
        except Exception as e:
            logger.error(f"Ошибка при поиске в {engine}: {str(e)}")
            return []
    
    def _search_duckduckgo(self, query: str, config: Dict[str, Any], max_results: int) -> List[Dict[str, str]]:
        """
        Поиск в DuckDuckGo через открытый API.
        
        Args:
            query: Поисковый запрос
            config: Конфигурация поисковой системы
            max_results: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        # DuckDuckGo не имеет официального API, но можно использовать неофициальный эндпоинт
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            logger.error(f"Ошибка при запросе к DuckDuckGo API: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        results = []
        
        # Добавление основного результата (Instant Answer)
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", ""),
                "link": data.get("AbstractURL", ""),
                "snippet": data.get("AbstractText", ""),
                "source": "duckduckgo"
            })
        
        # Добавление связанных тем
        for topic in data.get("RelatedTopics", [])[:max_results-len(results)]:
            if "Text" in topic and "FirstURL" in topic:
                results.append({
                    "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                    "link": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                    "source": "duckduckgo"
                })
        
        return results
    
    def _search_wikipedia(self, query: str, config: Dict[str, Any], max_results: int) -> List[Dict[str, str]]:
        """
        Поиск в Wikipedia через открытый API.
        
        Args:
            query: Поисковый запрос
            config: Конфигурация поисковой системы
            max_results: Максимальное количество результатов
            
        Returns:
            Список результатов поиска
        """
        url = "https://ru.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            logger.error(f"Ошибка при запросе к Wikipedia API: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        results = []
        
        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                # Удаление HTML-тегов из сниппета
                snippet = item.get("snippet", "")
                snippet = snippet.replace("<span class=\"searchmatch\">", "")
                snippet = snippet.replace("</span>", "")
                
                results.append({
                    "title": item.get("title", ""),
                    "link": f"https://ru.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}",
                    "snippet": snippet,
                    "source": "wikipedia"
                })
        
        return results
    
    def get_wikipedia_info(self, topic: str, language: str = "ru") -> Dict[str, Any]:
        """
        Получение информации из Wikipedia.
        
        Args:
            topic: Тема для поиска
            language: Код языка (по умолчанию русский)
            
        Returns:
            Словарь с информацией
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Получение информации из Wikipedia невозможно.")
            return {}
        
        if "wikipedia" not in self.apis or not self.apis["wikipedia"].get("enabled", False):
            logger.warning("API Wikipedia недоступно или отключено.")
            return {}
        
        try:
            url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{topic}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Ошибка при запросе к Wikipedia API: {response.status_code} - {response.text}")
                return {}
            
            data = response.json()
            return {
                "title": data.get("title", ""),
                "extract": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации из Wikipedia: {str(e)}")
            return {}
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Получение информации о погоде через OpenWeatherMap.
        
        Args:
            location: Местоположение
            
        Returns:
            Словарь с информацией о погоде
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Получение информации о погоде невозможно.")
            return {}
        
        if "open_weather" not in self.apis or not self.apis["open_weather"].get("enabled", False):
            logger.warning("API погоды недоступно или отключено.")
            return {}
        
        # Для демонстрации используем открытый API без ключа
        try:
            # Используем альтернативный открытый API погоды
            url = f"https://wttr.in/{location}?format=j1"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Ошибка при запросе к API погоды: {response.status_code} - {response.text}")
                return {}
            
            data = response.json()
            current = data.get("current_condition", [{}])[0]
            
            return {
                "location": location,
                "temperature": current.get("temp_C", ""),
                "description": current.get("weatherDesc", [{}])[0].get("value", ""),
                "humidity": current.get("humidity", ""),
                "wind_speed": current.get("windspeedKmph", "")
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о погоде: {str(e)}")
            return {}
    
    def get_open_library_info(self, query: str) -> Dict[str, Any]:
        """
        Получение информации о книгах через Open Library API.
        
        Args:
            query: Поисковый запрос (название книги или автор)
            
        Returns:
            Словарь с информацией о книгах
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Получение информации о книгах невозможно.")
            return {}
        
        if "open_library" not in self.apis or not self.apis["open_library"].get("enabled", False):
            logger.warning("API Open Library недоступно или отключено.")
            return {}
        
        try:
            url = "https://openlibrary.org/search.json"
            params = {"q": query, "limit": 5}
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Ошибка при запросе к Open Library API: {response.status_code} - {response.text}")
                return {}
            
            data = response.json()
            books = []
            
            for doc in data.get("docs", [])[:5]:
                book = {
                    "title": doc.get("title", ""),
                    "author": ", ".join(doc.get("author_name", [])),
                    "year": doc.get("first_publish_year", ""),
                    "isbn": doc.get("isbn", [""])[0] if doc.get("isbn") else "",
                    "subject": ", ".join(doc.get("subject", [])[:3])
                }
                books.append(book)
            
            return {
                "query": query,
                "total_found": data.get("numFound", 0),
                "books": books
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о книгах: {str(e)}")
            return {}
    
    def get_space_info(self) -> Dict[str, Any]:
        """
        Получение информации о космосе через Open Notify API.
        
        Returns:
            Словарь с информацией о МКС и астронавтах
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Получение информации о космосе невозможно.")
            return {}
        
        if "open_notify" not in self.apis or not self.apis["open_notify"].get("enabled", False):
            logger.warning("API Open Notify недоступно или отключено.")
            return {}
        
        try:
            # Получение информации о людях в космосе
            people_url = "http://api.open-notify.org/astros.json"
            people_response = requests.get(people_url)
            
            # Получение текущего положения МКС
            iss_url = "http://api.open-notify.org/iss-now.json"
            iss_response = requests.get(iss_url)
            
            result = {}
            
            if people_response.status_code == 200:
                people_data = people_response.json()
                result["people_in_space"] = people_data.get("number", 0)
                result["astronauts"] = [
                    {"name": person.get("name", ""), "craft": person.get("craft", "")}
                    for person in people_data.get("people", [])
                ]
            
            if iss_response.status_code == 200:
                iss_data = iss_response.json()
                position = iss_data.get("iss_position", {})
                result["iss_position"] = {
                    "latitude": position.get("latitude", ""),
                    "longitude": position.get("longitude", ""),
                    "timestamp": iss_data.get("timestamp", "")
                }
            
            return result
        except Exception as e:
            logger.error(f"Ошибка при получении информации о космосе: {str(e)}")
            return {}
    
    def get_random_quote(self) -> Dict[str, Any]:
        """
        Получение случайной цитаты через Quotable API.
        
        Returns:
            Словарь с цитатой
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Получение цитаты невозможно.")
            return {}
        
        if "quotable" not in self.apis or not self.apis["quotable"].get("enabled", False):
            logger.warning("API Quotable недоступно или отключено.")
            return {}
        
        try:
            url = "https://api.quotable.io/random"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Ошибка при запросе к Quotable API: {response.status_code} - {response.text}")
                return {}
            
            data = response.json()
            return {
                "content": data.get("content", ""),
                "author": data.get("author", ""),
                "tags": data.get("tags", [])
            }
        except Exception as e:
            logger.error(f"Ошибка при получении цитаты: {str(e)}")
            return {}
    
    def get_cat_fact(self) -> Dict[str, Any]:
        """
        Получение случайного факта о кошках через Cat Facts API.
        
        Returns:
            Словарь с фактом о кошке
        """
        if not self.enabled:
            logger.warning("Внешние источники отключены. Получение факта о кошке невозможно.")
            return {}
        
        if "cat_facts" not in self.apis or not self.apis["cat_facts"].get("enabled", False):
            logger.warning("API Cat Facts недоступно или отключено.")
            return {}
        
        try:
            url = "https://catfact.ninja/fact"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Ошибка при запросе к Cat Facts API: {response.status_code} - {response.text}")
                return {}
            
            data = response.json()
            return {
                "fact": data.get("fact", ""),
                "length": data.get("length", 0)
            }
        except Exception as e:
            logger.error(f"Ошибка при получении факта о кошке: {str(e)}")
            return {}
    
    def close_connections(self) -> None:
        """
        Закрытие всех соединений с базами данных.
        """
        for db_name, connection in self.db_connections.items():
            try:
                db_type = self.databases[db_name]["type"]
                
                if db_type == "sqlite":
                    connection["connection"].close()
                else:
                    logger.warning(f"Неподдерживаемый тип базы данных: {db_type}")
            except Exception as e:
                logger.error(f"Ошибка при закрытии соединения с базой данных {db_name}: {str(e)}")
    
    def __del__(self):
        """
        Деструктор для закрытия соединений при уничтожении объекта.
        """
        self.close_connections() 