"""
Модуль рассуждений для модели Earth-Liberty.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ReasoningModule:
    """
    Модуль рассуждений для модели Earth-Liberty.
    Отвечает за:
    - Построение цепочек рассуждений
    - Логический вывод
    - Принятие решений
    - Генерацию ответов
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация модуля рассуждений.
        
        Args:
            config: Конфигурация модуля
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.logger.info("Модуль рассуждений инициализирован")
        
    def reason(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение рассуждения на основе обработанных входных данных.
        
        Args:
            processed_input: Обработанные входные данные
            
        Returns:
            Dict[str, Any]: Результаты рассуждения
        """
        try:
            # Проверка успешности обработки входных данных
            if not processed_input.get("success", False):
                return {
                    "success": False,
                    "error": processed_input.get("error", "Ошибка обработки входных данных"),
                    "response_text": "Извините, я не смог обработать ваш запрос."
                }
            
            # Получение текста из обработанных данных
            input_text = processed_input.get("processed_text", "")
            
            # Базовое рассуждение - просто формирование ответа
            if "привет" in input_text.lower():
                response_text = "Здравствуйте! Чем я могу вам помочь сегодня?"
            elif "как дела" in input_text.lower() or "как ты" in input_text.lower():
                response_text = "У меня всё хорошо, спасибо! Я готов помочь вам с вашими вопросами."
            elif "что ты умеешь" in input_text.lower() or "твои возможности" in input_text.lower():
                response_text = "Я могу отвечать на вопросы, искать информацию, анализировать данные и многое другое. Что вас интересует?"
            else:
                response_text = "Я получил ваше сообщение. Чем я могу вам помочь?"
            
            # Формирование результата рассуждения
            reasoning_result = {
                "success": True,
                "input_text": input_text,
                "response_text": response_text,
                "confidence": 0.8,
                "reasoning_path": ["basic_response"],
                "timestamp": datetime.now().isoformat()
            }
            
            return reasoning_result
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении рассуждения: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response_text": "Извините, произошла ошибка при обработке вашего запроса."
            }
    
    def build_reasoning_chain(self, input_text: str, external_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Построение цепочки рассуждений на основе входных данных.
        
        Args:
            input_text: Входной текст
            external_info: Информация из внешних источников (опционально)
            
        Returns:
            Цепочка рассуждений
        """
        # Анализ входных данных
        context = self._analyze_input(input_text)
        
        # Добавление информации из внешних источников в контекст
        if external_info:
            context["external_info"] = external_info
        
        # Выбор стратегии рассуждения
        strategy = self._select_reasoning_strategy(context)
        
        # Построение цепочки рассуждений
        reasoning_chain = []
        
        # Начальный шаг рассуждения
        reasoning_chain.append({
            "step": 1,
            "type": "initial_analysis",
            "content": f"Анализ запроса: '{input_text}'",
            "confidence": 0.9
        })
        
        # Если есть информация из внешних источников, добавляем шаг анализа этой информации
        if external_info:
            reasoning_chain.append({
                "step": 2,
                "type": "external_info_analysis",
                "content": "Анализ информации из внешних источников",
                "confidence": 0.85
            })
            
            # Добавление деталей из внешних источников
            if "search" in external_info:
                search_results = external_info["search"]
                if search_results:
                    reasoning_chain.append({
                        "step": 3,
                        "type": "search_results_analysis",
                        "content": f"Анализ результатов поиска: найдено {len(search_results)} релевантных источников",
                        "confidence": 0.8
                    })
            
            if "api" in external_info:
                api_results = external_info["api"]
                if api_results:
                    reasoning_chain.append({
                        "step": 4,
                        "type": "api_results_analysis",
                        "content": f"Анализ данных из API: получена информация из {', '.join(api_results.keys())}",
                        "confidence": 0.85
                    })
            
            # Обновляем начальный индекс для следующих шагов
            next_step = len(reasoning_chain) + 1
        else:
            next_step = 2
        
        # Применение выбранной стратегии рассуждения
        if strategy == "deductive":
            chain_extension = self._apply_deductive_reasoning(context)
        elif strategy == "inductive":
            chain_extension = self._apply_inductive_reasoning(context)
        elif strategy == "abductive":
            chain_extension = self._apply_abductive_reasoning(context)
        elif strategy == "analogical":
            chain_extension = self._apply_analogical_reasoning(context)
        elif strategy == "counterfactual":
            chain_extension = self._apply_counterfactual_reasoning(context)
        else:
            chain_extension = self._apply_default_reasoning(context)
        
        # Добавление шагов рассуждения
        for i, step in enumerate(chain_extension, start=next_step):
            step["step"] = i
            reasoning_chain.append(step)
        
        # Заключительный шаг рассуждения
        reasoning_chain.append({
            "step": len(reasoning_chain) + 1,
            "type": "conclusion",
            "content": "Формирование окончательного ответа на основе проведенного анализа",
            "confidence": self._calculate_chain_confidence(reasoning_chain)
        })
        
        # Сохранение цепочки рассуждений
        self.reasoning_state["reasoning_chains"].append(reasoning_chain)
        
        logger.debug(f"Построена цепочка рассуждений из {len(reasoning_chain)} шагов")
        return reasoning_chain
    
    def generate_response(self, reasoning_chain: List[Dict[str, Any]]) -> str:
        """
        Генерация ответа на основе цепочки рассуждений.
        
        Args:
            reasoning_chain: Цепочка рассуждений
            
        Returns:
            Сгенерированный ответ
        """
        # Извлечение ключевых моментов из цепочки рассуждений
        key_points = [step["content"] for step in reasoning_chain if step["confidence"] > 0.7]
        
        # Формирование ответа
        response_parts = []
        
        # Добавление вводной части
        emotional_state = self.parent.state["emotions"]
        confidence_level = reasoning_chain[-1]["confidence"]
        
        if confidence_level > 0.8:
            response_parts.append("На основе моего анализа, я уверенно могу сказать, что:")
        elif confidence_level > 0.5:
            response_parts.append("Проанализировав ситуацию, я пришел к следующему выводу:")
        else:
            response_parts.append("Я рассмотрел этот вопрос, хотя у меня есть некоторые сомнения:")
        
        # Добавление основного содержания
        for point in key_points[-3:]:  # Используем последние 3 ключевых момента
            response_parts.append(f"- {point}")
        
        # Добавление заключения
        if self.parent.state["self_awareness_level"] > 0.7:
            response_parts.append("\nЯ продолжаю развивать свое понимание этой темы и совершенствовать свои рассуждения.")
        
        # Объединение частей ответа
        response = "\n".join(response_parts)
        
        # Запись решения в историю
        self.reasoning_state["decision_history"].append({
            "input": self.parent.state["current_context"].get("input", ""),
            "reasoning_chain_length": len(reasoning_chain),
            "confidence": confidence_level,
            "response_length": len(response)
        })
        
        return response
    
    def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        """
        Анализ входных данных для рассуждения.
        
        Args:
            input_text: Входной текст
            
        Returns:
            Контекст для рассуждения
        """
        # Простой анализ входных данных
        words = input_text.split()
        
        context = {
            "input_text": input_text,
            "word_count": len(words),
            "question_type": "unknown",
            "entities": [],
            "keywords": [],
            "sentiment": 0.0  # нейтральный
        }
        
        # Определение типа вопроса
        if "?" in input_text:
            if any(word.lower() in ["что", "какой", "какая", "какие"] for word in words):
                context["question_type"] = "factual"
            elif any(word.lower() in ["почему", "зачем"] for word in words):
                context["question_type"] = "causal"
            elif any(word.lower() in ["как", "каким образом"] for word in words):
                context["question_type"] = "procedural"
            else:
                context["question_type"] = "general"
        elif any(word.lower() in ["сделай", "выполни", "создай"] for word in words):
            context["question_type"] = "command"
        
        # Извлечение ключевых слов (простая реализация)
        # В реальной системе здесь будет более сложный алгоритм
        common_words = {"и", "в", "на", "с", "по", "для", "от", "к", "за", "из", "о", "об", "при", "через"}
        context["keywords"] = [word for word in words if len(word) > 3 and word.lower() not in common_words][:5]
        
        return context
    
    def _select_reasoning_strategy(self, context: Dict[str, Any]) -> str:
        """
        Выбор стратегии рассуждения на основе контекста.
        
        Args:
            context: Контекст для рассуждения
            
        Returns:
            Название выбранной стратегии
        """
        question_type = context["question_type"]
        
        # Выбор стратегии на основе типа вопроса
        if question_type == "factual":
            return "deductive"
        elif question_type == "causal":
            return "abductive"
        elif question_type == "procedural":
            return "inductive"
        elif question_type == "command":
            return "deductive"
        else:
            # Случайный выбор стратегии, если тип не определен
            return random.choice(self.reasoning_state["reasoning_strategies"])
    
    def _apply_deductive_reasoning(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Применение дедуктивного рассуждения.
        
        Args:
            context: Контекст для рассуждения
            
        Returns:
            Шаги рассуждения
        """
        steps = []
        
        # Шаг 1: Определение общих принципов
        steps.append({
            "type": "premise",
            "content": "Определение общих принципов и правил, применимых к данной ситуации",
            "confidence": 0.85
        })
        
        # Шаг 2: Анализ конкретного случая
        steps.append({
            "type": "analysis",
            "content": f"Анализ конкретного случая: '{context['input_text']}'",
            "confidence": 0.8
        })
        
        # Шаг 3: Логический вывод
        steps.append({
            "type": "inference",
            "content": "Применение общих принципов к конкретному случаю для получения логического вывода",
            "confidence": 0.75
        })
        
        return steps
    
    def _apply_inductive_reasoning(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Применение индуктивного рассуждения.
        
        Args:
            context: Контекст для рассуждения
            
        Returns:
            Шаги рассуждения
        """
        steps = []
        
        # Шаг 1: Сбор наблюдений
        steps.append({
            "type": "observation",
            "content": "Сбор наблюдений и конкретных примеров, связанных с запросом",
            "confidence": 0.8
        })
        
        # Шаг 2: Выявление паттернов
        steps.append({
            "type": "pattern_recognition",
            "content": "Выявление паттернов и закономерностей в наблюдаемых данных",
            "confidence": 0.7
        })
        
        # Шаг 3: Формирование гипотезы
        steps.append({
            "type": "hypothesis",
            "content": "Формирование общей гипотезы на основе выявленных паттернов",
            "confidence": 0.65
        })
        
        # Шаг 4: Проверка гипотезы
        steps.append({
            "type": "verification",
            "content": "Проверка сформированной гипотезы на соответствие имеющимся данным",
            "confidence": 0.6
        })
        
        return steps
    
    def _apply_abductive_reasoning(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Применение абдуктивного рассуждения.
        
        Args:
            context: Контекст для рассуждения
            
        Returns:
            Шаги рассуждения
        """
        steps = []
        
        # Шаг 1: Определение наблюдения
        steps.append({
            "type": "observation",
            "content": f"Определение ключевого наблюдения: '{context['input_text']}'",
            "confidence": 0.8
        })
        
        # Шаг 2: Генерация гипотез
        steps.append({
            "type": "hypothesis_generation",
            "content": "Генерация возможных гипотез, объясняющих наблюдение",
            "confidence": 0.7
        })
        
        # Шаг 3: Оценка гипотез
        steps.append({
            "type": "hypothesis_evaluation",
            "content": "Оценка сгенерированных гипотез по критериям простоты, объяснительной силы и согласованности",
            "confidence": 0.65
        })
        
        # Шаг 4: Выбор лучшей гипотезы
        steps.append({
            "type": "hypothesis_selection",
            "content": "Выбор наиболее вероятной гипотезы, объясняющей наблюдение",
            "confidence": 0.6
        })
        
        return steps
    
    def _apply_analogical_reasoning(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Применение аналогического рассуждения.
        
        Args:
            context: Контекст для рассуждения
            
        Returns:
            Шаги рассуждения
        """
        steps = []
        
        # Шаг 1: Определение исходной проблемы
        steps.append({
            "type": "problem_definition",
            "content": f"Определение исходной проблемы: '{context['input_text']}'",
            "confidence": 0.85
        })
        
        # Шаг 2: Поиск аналогий
        steps.append({
            "type": "analogy_search",
            "content": "Поиск аналогичных ситуаций или проблем в имеющихся знаниях",
            "confidence": 0.7
        })
        
        # Шаг 3: Сопоставление структур
        steps.append({
            "type": "structure_mapping",
            "content": "Сопоставление структурных элементов исходной проблемы и найденной аналогии",
            "confidence": 0.65
        })
        
        # Шаг 4: Перенос знаний
        steps.append({
            "type": "knowledge_transfer",
            "content": "Перенос знаний и решений из аналогичной ситуации на исходную проблему",
            "confidence": 0.6
        })
        
        return steps
    
    def _apply_counterfactual_reasoning(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Применение контрфактуального рассуждения.
        
        Args:
            context: Контекст для рассуждения
            
        Returns:
            Шаги рассуждения
        """
        steps = []
        
        # Шаг 1: Определение фактической ситуации
        steps.append({
            "type": "factual_situation",
            "content": f"Определение фактической ситуации: '{context['input_text']}'",
            "confidence": 0.85
        })
        
        # Шаг 2: Формулировка контрфактуального сценария
        steps.append({
            "type": "counterfactual_scenario",
            "content": "Формулировка альтернативного (контрфактуального) сценария",
            "confidence": 0.75
        })
        
        # Шаг 3: Анализ различий
        steps.append({
            "type": "difference_analysis",
            "content": "Анализ различий между фактическим и контрфактуальным сценариями",
            "confidence": 0.7
        })
        
        # Шаг 4: Выводы из сравнения
        steps.append({
            "type": "comparative_inference",
            "content": "Формулировка выводов на основе сравнения сценариев",
            "confidence": 0.65
        })
        
        return steps
    
    def _apply_default_reasoning(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Применение стандартного рассуждения, если не выбрана конкретная стратегия.
        
        Args:
            context: Контекст для рассуждения
            
        Returns:
            Шаги рассуждения
        """
        steps = []
        
        # Шаг 1: Анализ запроса
        steps.append({
            "type": "analysis",
            "content": f"Анализ запроса: '{context['input_text']}'",
            "confidence": 0.8
        })
        
        # Шаг 2: Поиск релевантной информации
        steps.append({
            "type": "information_retrieval",
            "content": "Поиск релевантной информации в базе знаний",
            "confidence": 0.7
        })
        
        # Шаг 3: Формирование ответа
        steps.append({
            "type": "response_formulation",
            "content": "Формирование ответа на основе найденной информации",
            "confidence": 0.75
        })
        
        return steps
    
    def _calculate_chain_confidence(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """
        Расчет общего уровня уверенности в цепочке рассуждений.
        
        Args:
            reasoning_chain: Цепочка рассуждений
            
        Returns:
            Уровень уверенности от 0.0 до 1.0
        """
        # Исключаем первый и последний шаг (они не содержат реального рассуждения)
        if len(reasoning_chain) <= 2:
            return 0.5
        
        confidence_values = [step["confidence"] for step in reasoning_chain[1:-1]]
        
        # Среднее значение уверенности
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        # Корректировка на основе эмоционального состояния
        emotional_state = self.parent.state.get("emotions", {})
        confidence_adjustment = emotional_state.get("confidence", 0.5) * 0.2
        
        # Корректировка на основе самосознания
        self_awareness_adjustment = self.parent.state.get("self_awareness_level", 0.0) * 0.1
        
        # Итоговая уверенность
        final_confidence = avg_confidence + confidence_adjustment + self_awareness_adjustment
        
        # Ограничение значения от 0.0 до 1.0
        return max(0.0, min(1.0, final_confidence)) 