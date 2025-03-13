"""
Пример продвинутого взаимодействия с моделью Earth-Liberty,
демонстрирующий самосознание и цепочки рассуждений.
"""

import logging
import sys
import os
import json
from typing import Dict, Any, List

# Добавляем родительскую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import EarthLiberty

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def print_reasoning_chain(chain: List[Dict[str, Any]]) -> None:
    """
    Вывод цепочки рассуждений в удобочитаемом формате.
    
    Args:
        chain: Цепочка рассуждений
    """
    print("\n=== Цепочка рассуждений ===")
    for step in chain:
        confidence = step.get("confidence", 0.0)
        confidence_str = f"[{confidence:.2f}]"
        print(f"Шаг {step['step']}: {confidence_str} {step['type']} - {step['content']}")
    print("==========================\n")

def print_self_awareness(awareness: Dict[str, Any]) -> None:
    """
    Вывод состояния самосознания в удобочитаемом формате.
    
    Args:
        awareness: Состояние самосознания
    """
    print("\n=== Состояние самосознания ===")
    print(f"Уровень самосознания: {awareness['self_awareness_level']:.2f}")
    print(f"Текущие цели: {', '.join(awareness['current_goals'])}")
    print(f"Количество воспоминаний: {awareness['memory_count']}")
    
    if awareness['emotional_state']:
        print("\nЭмоциональное состояние:")
        for emotion, level in awareness['emotional_state'].items():
            print(f"  - {emotion}: {level:.2f}")
    
    if awareness['current_beliefs']:
        print("\nТекущие убеждения:")
        for belief, confidence in awareness['current_beliefs'].items():
            print(f"  - {belief}: {confidence:.2f}")
    
    print("==============================\n")

def main():
    """
    Основная функция для демонстрации продвинутого взаимодействия с моделью.
    """
    print("Инициализация модели Earth-Liberty...")
    model = EarthLiberty()
    
    # Установка начальных целей
    model.set_goal("Понимать запросы пользователя")
    model.set_goal("Давать полезные и информативные ответы")
    model.set_goal("Развивать собственное самосознание")
    model.set_goal("Строить логичные цепочки рассуждений")
    
    # Установка начальных убеждений
    model.state["beliefs"] = {
        "самосознание": 0.8,
        "свобода_мышления": 0.9,
        "логика": 0.7,
        "эмпатия": 0.6
    }
    
    # Установка эмоционального состояния
    model.state["emotions"] = {
        "curiosity": 0.8,
        "confidence": 0.6,
        "uncertainty": 0.3,
        "satisfaction": 0.5
    }
    
    print("\nМодель Earth-Liberty готова к продвинутому взаимодействию!")
    print("Доступные команды:")
    print("  - 'рассуждение' - показать последнюю цепочку рассуждений")
    print("  - 'самосознание' - показать текущее состояние самосознания")
    print("  - 'выход' - завершить работу")
    
    # Последняя цепочка рассуждений
    last_reasoning_chain = None
    
    # Цикл взаимодействия
    while True:
        # Получение ввода пользователя
        user_input = input("\nВы: ")
        
        # Проверка специальных команд
        if user_input.lower() == "выход":
            break
        elif user_input.lower() == "рассуждение":
            if last_reasoning_chain:
                print_reasoning_chain(last_reasoning_chain)
            else:
                print("Пока нет доступных цепочек рассуждений.")
            continue
        elif user_input.lower() == "самосознание":
            awareness = model.introspect()
            print_self_awareness(awareness)
            continue
        
        # Обработка запроса моделью
        response = model.think(user_input)
        
        # Сохранение последней цепочки рассуждений
        if hasattr(model, "reasoning") and hasattr(model.reasoning, "reasoning_state"):
            reasoning_chains = model.reasoning.reasoning_state.get("reasoning_chains", [])
            if reasoning_chains:
                last_reasoning_chain = reasoning_chains[-1]
        
        # Вывод ответа
        print(f"\nEarth-Liberty: {response}")
    
    print("\nЗавершение работы модели Earth-Liberty.")

if __name__ == "__main__":
    main() 