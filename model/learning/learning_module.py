"""
Модуль обучения для модели Earth-Liberty.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class LearningModule:
    """
    Модуль обучения для модели Earth-Liberty.
    Отвечает за:
    - Адаптацию модели на основе опыта
    - Обновление внутренних представлений
    - Формирование новых знаний
    - Оптимизацию процессов мышления
    """
    
    def __init__(self, parent_model):
        """
        Инициализация модуля обучения.
        
        Args:
            parent_model: Родительская модель Earth-Liberty
        """
        self.parent = parent_model
        self.learning_state = {
            "interaction_history": [],
            "learned_patterns": {},
            "adaptation_level": 0.1,
            "learning_rate": 0.05,
            "knowledge_base": {
                "concepts": {},
                "relations": [],
                "rules": []
            }
        }
        logger.info("Модуль обучения инициализирован")
    
    def learn_from_interaction(self, input_text: str, response: str, external_info: Dict[str, Any] = None) -> None:
        """
        Обучение на основе взаимодействия.
        
        Args:
            input_text: Входной текст
            response: Сгенерированный ответ
            external_info: Информация из внешних источников (опционально)
        """
        # Запись взаимодействия в историю
        interaction = {
            "input": input_text,
            "response": response,
            "timestamp": "current_time",  # В реальной системе здесь будет реальное время
            "context": self.parent.state["current_context"].copy()
        }
        
        # Добавление информации из внешних источников
        if external_info:
            interaction["external_info"] = external_info
        
        self.learning_state["interaction_history"].append(interaction)
        
        # Извлечение паттернов из взаимодействия
        patterns = self._extract_patterns(input_text, response)
        
        # Извлечение дополнительных паттернов из внешней информации
        if external_info:
            external_patterns = self._extract_patterns_from_external_info(external_info)
            patterns.extend(external_patterns)
        
        # Обновление базы знаний
        self._update_knowledge_base(patterns)
        
        # Адаптация параметров модели
        self._adapt_model_parameters()
        
        logger.debug(f"Обучение на основе взаимодействия завершено, извлечено {len(patterns)} паттернов")
    
    def optimize_reasoning(self) -> None:
        """
        Оптимизация процессов рассуждения на основе накопленного опыта.
        """
        # Анализ истории рассуждений
        if hasattr(self.parent, "reasoning") and hasattr(self.parent.reasoning, "reasoning_state"):
            reasoning_chains = self.parent.reasoning.reasoning_state.get("reasoning_chains", [])
            
            if reasoning_chains:
                # Оценка эффективности различных стратегий рассуждения
                strategy_effectiveness = self._evaluate_reasoning_strategies(reasoning_chains)
                
                # Корректировка весов стратегий рассуждения
                self._adjust_reasoning_weights(strategy_effectiveness)
                
                logger.debug("Процессы рассуждения оптимизированы")
    
    def update_beliefs(self) -> None:
        """
        Обновление системы убеждений на основе накопленного опыта.
        """
        # Анализ истории взаимодействий
        if self.learning_state["interaction_history"]:
            # Извлечение ключевых концепций
            concepts = self._extract_key_concepts()
            
            # Обновление убеждений
            for concept, confidence in concepts.items():
                if concept not in self.parent.state["beliefs"]:
                    self.parent.state["beliefs"][concept] = confidence
                else:
                    # Постепенное обновление уверенности в убеждении
                    current_confidence = self.parent.state["beliefs"][concept]
                    updated_confidence = (
                        (1 - self.learning_state["learning_rate"]) * current_confidence +
                        self.learning_state["learning_rate"] * confidence
                    )
                    self.parent.state["beliefs"][concept] = updated_confidence
            
            logger.debug(f"Система убеждений обновлена, текущее количество убеждений: {len(self.parent.state['beliefs'])}")
    
    def _extract_patterns(self, input_text: str, response: str) -> List[Dict[str, Any]]:
        """
        Извлечение паттернов из взаимодействия.
        
        Args:
            input_text: Входной текст
            response: Сгенерированный ответ
            
        Returns:
            Список извлеченных паттернов
        """
        patterns = []
        
        # Простой анализ входных данных и ответа
        input_words = input_text.lower().split()
        response_words = response.lower().split()
        
        # Поиск повторяющихся слов
        common_words = set(input_words) & set(response_words)
        
        # Формирование паттернов на основе общих слов
        for word in common_words:
            if len(word) > 3:  # Игнорируем короткие слова
                pattern = {
                    "type": "word_association",
                    "trigger": word,
                    "context": input_text,
                    "response_fragment": response,
                    "confidence": 0.6 + random.random() * 0.2  # Случайная уверенность от 0.6 до 0.8
                }
                patterns.append(pattern)
        
        # В реальной системе здесь будет более сложный алгоритм извлечения паттернов
        
        return patterns
    
    def _update_knowledge_base(self, patterns: List[Dict[str, Any]]) -> None:
        """
        Обновление базы знаний на основе извлеченных паттернов.
        
        Args:
            patterns: Список извлеченных паттернов
        """
        for pattern in patterns:
            if pattern["type"] == "word_association":
                # Обновление концепций
                concept = pattern["trigger"]
                if concept not in self.learning_state["knowledge_base"]["concepts"]:
                    self.learning_state["knowledge_base"]["concepts"][concept] = {
                        "occurrences": 1,
                        "confidence": pattern["confidence"],
                        "contexts": [pattern["context"]],
                        "related_concepts": []
                    }
                else:
                    # Обновление существующей концепции
                    concept_data = self.learning_state["knowledge_base"]["concepts"][concept]
                    concept_data["occurrences"] += 1
                    concept_data["confidence"] = (
                        (concept_data["occurrences"] - 1) * concept_data["confidence"] +
                        pattern["confidence"]
                    ) / concept_data["occurrences"]
                    concept_data["contexts"].append(pattern["context"])
                    
                    # Ограничение количества сохраняемых контекстов
                    if len(concept_data["contexts"]) > 10:
                        concept_data["contexts"] = concept_data["contexts"][-10:]
    
    def _adapt_model_parameters(self) -> None:
        """
        Адаптация параметров модели на основе накопленного опыта.
        """
        # Увеличение уровня адаптации с каждым взаимодействием
        history_length = len(self.learning_state["interaction_history"])
        self.learning_state["adaptation_level"] = min(
            1.0, 
            0.1 + history_length / 200  # Простая формула роста
        )
        
        # Корректировка скорости обучения
        if history_length > 100:
            self.learning_state["learning_rate"] = max(0.01, 0.05 - history_length / 10000)
    
    def _evaluate_reasoning_strategies(self, reasoning_chains: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Оценка эффективности различных стратегий рассуждения.
        
        Args:
            reasoning_chains: История цепочек рассуждений
            
        Returns:
            Словарь с оценками эффективности стратегий
        """
        # Инициализация оценок стратегий
        strategies = {
            "deductive": 0.5,
            "inductive": 0.5,
            "abductive": 0.5,
            "analogical": 0.5,
            "counterfactual": 0.5
        }
        
        # В реальной системе здесь будет более сложный алгоритм оценки
        
        return strategies
    
    def _adjust_reasoning_weights(self, strategy_effectiveness: Dict[str, float]) -> None:
        """
        Корректировка весов стратегий рассуждения.
        
        Args:
            strategy_effectiveness: Оценки эффективности стратегий
        """
        # В реальной системе здесь будет реализация корректировки весов
        pass
    
    def _extract_key_concepts(self) -> Dict[str, float]:
        """
        Извлечение ключевых концепций из истории взаимодействий.
        
        Returns:
            Словарь с ключевыми концепциями и уровнями уверенности
        """
        concepts = {}
        
        # Анализ концепций в базе знаний
        for concept, data in self.learning_state["knowledge_base"]["concepts"].items():
            if data["occurrences"] > 2:  # Только концепции, встречающиеся более 2 раз
                concepts[concept] = data["confidence"]
        
        return concepts
    
    def _extract_patterns_from_external_info(self, external_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Извлечение паттернов из внешней информации.
        
        Args:
            external_info: Информация из внешних источников
            
        Returns:
            Список извлеченных паттернов
        """
        patterns = []
        
        # Извлечение паттернов из результатов поиска
        if "search" in external_info:
            search_results = external_info["search"]
            for result in search_results:
                # Извлечение ключевых слов из заголовка и сниппета
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                source = result.get("source", "unknown")
                
                # Создание паттерна для заголовка
                if title:
                    pattern = {
                        "type": "external_knowledge",
                        "subtype": "search_title",
                        "content": title,
                        "source": source,
                        "confidence": 0.7,
                        "metadata": {
                            "link": result.get("link", "")
                        }
                    }
                    patterns.append(pattern)
                
                # Создание паттерна для сниппета
                if snippet:
                    pattern = {
                        "type": "external_knowledge",
                        "subtype": "search_snippet",
                        "content": snippet,
                        "source": source,
                        "confidence": 0.6,
                        "metadata": {
                            "link": result.get("link", "")
                        }
                    }
                    patterns.append(pattern)
        
        # Извлечение паттернов из результатов API
        if "api" in external_info:
            api_results = external_info["api"]
            
            # Обработка результатов Wikipedia
            if "wikipedia" in api_results:
                wiki_results = api_results["wikipedia"]
                for result in wiki_results:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    
                    if title and snippet:
                        pattern = {
                            "type": "external_knowledge",
                            "subtype": "wikipedia",
                            "content": f"{title}: {snippet}",
                            "source": "wikipedia",
                            "confidence": 0.8,
                            "metadata": {
                                "pageid": result.get("pageid", "")
                            }
                        }
                        patterns.append(pattern)
            
            # Обработка результатов погоды
            if "weather" in api_results:
                weather = api_results["weather"]
                city = weather.get("city", "")
                temperature = weather.get("temperature")
                description = weather.get("description", "")
                
                if city and temperature is not None:
                    pattern = {
                        "type": "external_knowledge",
                        "subtype": "weather",
                        "content": f"Погода в {city}: {temperature}°C, {description}",
                        "source": "weather_api",
                        "confidence": 0.9,
                        "metadata": weather
                    }
                    patterns.append(pattern)
        
        # Извлечение паттернов из результатов баз данных
        if "database" in external_info:
            db_results = external_info["database"]
            # Здесь будет код для обработки результатов из баз данных
        
        return patterns 