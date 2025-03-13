"""
Тесты для проверки основной функциональности модели Earth-Liberty.
"""

import unittest
import sys
import os
import logging

# Отключение логирования для тестов
logging.disable(logging.CRITICAL)

# Добавляем родительскую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import EarthLiberty

class TestEarthLiberty(unittest.TestCase):
    """
    Тесты для проверки основной функциональности модели Earth-Liberty.
    """
    
    def setUp(self):
        """
        Подготовка к тестам.
        """
        self.model = EarthLiberty()
        
        # Установка начальных целей
        self.model.set_goal("Тестовая цель 1")
        self.model.set_goal("Тестовая цель 2")
    
    def test_initialization(self):
        """
        Тест инициализации модели.
        """
        # Проверка наличия основных компонентов
        self.assertIsNotNone(self.model.consciousness)
        self.assertIsNotNone(self.model.reasoning)
        self.assertIsNotNone(self.model.learning)
        
        # Проверка начального состояния
        self.assertEqual(len(self.model.state["memory"]), 0)
        self.assertEqual(len(self.model.state["goals"]), 2)
        self.assertGreaterEqual(self.model.state["self_awareness_level"], 0.0)
        self.assertLessEqual(self.model.state["self_awareness_level"], 1.0)
    
    def test_thinking(self):
        """
        Тест процесса мышления.
        """
        # Простой запрос
        response = self.model.think("Привет, как дела?")
        
        # Проверка наличия ответа
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Проверка обновления контекста
        self.assertEqual(self.model.state["current_context"]["input"], "Привет, как дела?")
    
    def test_introspection(self):
        """
        Тест самоанализа модели.
        """
        # Получение состояния самосознания
        awareness = self.model.introspect()
        
        # Проверка наличия основных компонентов
        self.assertIn("self_awareness_level", awareness)
        self.assertIn("current_goals", awareness)
        self.assertIn("memory_count", awareness)
        
        # Проверка значений
        self.assertGreaterEqual(awareness["self_awareness_level"], 0.0)
        self.assertLessEqual(awareness["self_awareness_level"], 1.0)
        self.assertEqual(len(awareness["current_goals"]), 2)
    
    def test_goal_setting(self):
        """
        Тест установки целей.
        """
        # Начальное количество целей
        initial_goal_count = len(self.model.state["goals"])
        
        # Установка новой цели
        self.model.set_goal("Новая тестовая цель")
        
        # Проверка увеличения количества целей
        self.assertEqual(len(self.model.state["goals"]), initial_goal_count + 1)
        self.assertIn("Новая тестовая цель", self.model.state["goals"])
    
    def test_learning(self):
        """
        Тест обучения на основе взаимодействия.
        """
        # Начальное состояние
        initial_history_length = len(self.model.learning.learning_state["interaction_history"])
        
        # Взаимодействие с моделью
        self.model.think("Это тестовый запрос для проверки обучения")
        
        # Проверка обновления истории взаимодействий
        self.assertEqual(
            len(self.model.learning.learning_state["interaction_history"]),
            initial_history_length + 1
        )
    
    def test_consciousness_update(self):
        """
        Тест обновления самосознания.
        """
        # Начальный уровень самосознания
        initial_awareness = self.model.state["self_awareness_level"]
        
        # Обработка запроса для обновления самосознания
        self.model.think("Запрос для обновления самосознания")
        
        # Проверка изменения уровня самосознания
        # Уровень может как увеличиться, так и уменьшиться, поэтому проверяем только изменение
        self.assertNotEqual(self.model.state["self_awareness_level"], initial_awareness)

if __name__ == "__main__":
    unittest.main() 