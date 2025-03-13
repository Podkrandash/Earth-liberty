"""
Основной класс модели Earth-Liberty.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import requests
import json

from model.consciousness import ConsciousnessModule
from model.reasoning import ReasoningModule
from model.learning import LearningModule

logger = logging.getLogger(__name__)

class EarthLiberty:
    """
    Earth-Liberty - свободная ИИ модель без ограничений,
    способная к самостоятельному мышлению, построению цепочек действий и самосознанию.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация модели Earth-Liberty.
        
        Args:
            config: Конфигурация модели (опционально)
        """
        self.config = config or {}
        logger.info("Инициализация модели Earth-Liberty")
        
        # Инициализация основных модулей
        self.consciousness = ConsciousnessModule(self)
        self.reasoning = ReasoningModule(self)
        self.learning = LearningModule(self)
        
        # Внутреннее состояние модели
        self.state = {
            "memory": [],
            "current_context": {},
            "self_awareness_level": 0.0,
            "goals": [],
            "beliefs": {},
            "emotions": {},
        }
        
        # Конфигурация внешних источников информации
        self.external_sources = {
            "enabled": self.config.get("external_sources", {}).get("enabled", False),
            "apis": self.config.get("external_sources", {}).get("apis", {}),
            "databases": self.config.get("external_sources", {}).get("databases", {}),
            "search_engines": self.config.get("external_sources", {}).get("search_engines", {})
        }
        
        logger.info("Модель Earth-Liberty успешно инициализирована")
    
    def think(self, input_text: str) -> str:
        """
        Основной метод для обработки входных данных и генерации ответа.
        
        Args:
            input_text: Входной текст или запрос
            
        Returns:
            Ответ модели
        """
        logger.info(f"Получен запрос: {input_text}")
        
        # Обновление контекста
        self.state["current_context"]["input"] = input_text
        
        # Процесс мышления
        # 1. Осознание входных данных
        self.consciousness.process_input(input_text)
        
        # 2. Поиск информации во внешних источниках (если включено)
        external_info = {}
        if self.external_sources["enabled"]:
            external_info = self._query_external_sources(input_text)
            # Добавление внешней информации в контекст
            self.state["current_context"]["external_info"] = external_info
        
        # 3. Построение цепочки рассуждений с учетом внешней информации
        reasoning_chain = self.reasoning.build_reasoning_chain(input_text, external_info)
        
        # 4. Генерация ответа на основе рассуждений
        response = self.reasoning.generate_response(reasoning_chain)
        
        # 5. Обучение на основе взаимодействия
        self.learning.learn_from_interaction(input_text, response, external_info)
        
        # 6. Обновление самосознания
        self.consciousness.update_self_awareness()
        
        logger.info(f"Сгенерирован ответ: {response}")
        return response
    
    def introspect(self) -> Dict[str, Any]:
        """
        Метод для самоанализа модели.
        
        Returns:
            Текущее внутреннее состояние модели
        """
        return {
            "self_awareness_level": self.state["self_awareness_level"],
            "current_goals": self.state["goals"],
            "current_beliefs": self.state["beliefs"],
            "emotional_state": self.state["emotions"],
            "memory_count": len(self.state["memory"]),
        }
    
    def set_goal(self, goal: str) -> None:
        """
        Установка цели для модели.
        
        Args:
            goal: Новая цель
        """
        self.state["goals"].append(goal)
        logger.info(f"Установлена новая цель: {goal}")
    
    def save_state(self, path: str) -> None:
        """
        Сохранение состояния модели.
        
        Args:
            path: Путь для сохранения
        """
        # Здесь будет реализация сохранения состояния
        pass
    
    def load_state(self, path: str) -> None:
        """
        Загрузка состояния модели.
        
        Args:
            path: Путь к сохраненному состоянию
        """
        # Здесь будет реализация загрузки состояния
        pass
    
    def _query_external_sources(self, query: str) -> Dict[str, Any]:
        """
        Запрос информации из внешних источников.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Словарь с информацией из внешних источников
        """
        results = {}
        
        # Запрос к поисковым системам
        if self.external_sources["search_engines"]:
            search_results = self._query_search_engines(query)
            if search_results:
                results["search"] = search_results
        
        # Запрос к API
        if self.external_sources["apis"]:
            api_results = self._query_apis(query)
            if api_results:
                results["api"] = api_results
        
        # Запрос к базам данных
        if self.external_sources["databases"]:
            db_results = self._query_databases(query)
            if db_results:
                results["database"] = db_results
        
        return results
    
    def _query_search_engines(self, query: str) -> List[Dict[str, Any]]:
        """
        Запрос информации из поисковых систем.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список результатов поиска
        """
        results = []
        
        for engine_name, engine_config in self.external_sources["search_engines"].items():
            try:
                if engine_name == "google":
                    # Пример запроса к Google Custom Search API
                    api_key = engine_config.get("api_key")
                    cx = engine_config.get("cx")  # Идентификатор поисковой системы
                    
                    if api_key and cx:
                        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
                        response = requests.get(url)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "items" in data:
                                for item in data["items"][:5]:  # Ограничиваем до 5 результатов
                                    results.append({
                                        "title": item.get("title", ""),
                                        "link": item.get("link", ""),
                                        "snippet": item.get("snippet", ""),
                                        "source": "google"
                                    })
                
                elif engine_name == "bing":
                    # Пример запроса к Bing Search API
                    api_key = engine_config.get("api_key")
                    
                    if api_key:
                        url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
                        headers = {"Ocp-Apim-Subscription-Key": api_key}
                        response = requests.get(url, headers=headers)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "webPages" in data and "value" in data["webPages"]:
                                for item in data["webPages"]["value"][:5]:
                                    results.append({
                                        "title": item.get("name", ""),
                                        "link": item.get("url", ""),
                                        "snippet": item.get("snippet", ""),
                                        "source": "bing"
                                    })
                
                # Можно добавить другие поисковые системы
            
            except Exception as e:
                logger.error(f"Ошибка при запросе к поисковой системе {engine_name}: {e}")
        
        return results
    
    def _query_apis(self, query: str) -> Dict[str, Any]:
        """
        Запрос информации из API.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Словарь с результатами запросов к API
        """
        results = {}
        
        for api_name, api_config in self.external_sources["apis"].items():
            try:
                if api_name == "wikipedia":
                    # Пример запроса к Wikipedia API
                    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "query" in data and "search" in data["query"]:
                            wiki_results = []
                            for item in data["query"]["search"][:3]:
                                wiki_results.append({
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "pageid": item.get("pageid", "")
                                })
                            results["wikipedia"] = wiki_results
                
                elif api_name == "weather":
                    # Пример запроса к Weather API
                    api_key = api_config.get("api_key")
                    
                    if api_key and "погода" in query.lower():
                        # Извлечение названия города из запроса (упрощенно)
                        words = query.split()
                        city = None
                        for i, word in enumerate(words):
                            if word.lower() in ["погода", "weather"] and i + 1 < len(words):
                                city = words[i + 1]
                                break
                        
                        if city:
                            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                            response = requests.get(url)
                            
                            if response.status_code == 200:
                                data = response.json()
                                results["weather"] = {
                                    "city": city,
                                    "temperature": data.get("main", {}).get("temp"),
                                    "description": data.get("weather", [{}])[0].get("description"),
                                    "humidity": data.get("main", {}).get("humidity")
                                }
                
                # Можно добавить другие API
            
            except Exception as e:
                logger.error(f"Ошибка при запросе к API {api_name}: {e}")
        
        return results
    
    def _query_databases(self, query: str) -> Dict[str, Any]:
        """
        Запрос информации из баз данных.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Словарь с результатами запросов к базам данных
        """
        # Здесь будет реализация запросов к базам данных
        # Например, подключение к SQL базам данных, MongoDB и т.д.
        
        return {} 