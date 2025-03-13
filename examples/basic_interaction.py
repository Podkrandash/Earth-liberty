"""
Пример базового взаимодействия с моделью Earth-Liberty.
"""

import logging
import sys
import os

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

def main():
    """
    Основная функция для демонстрации взаимодействия с моделью.
    """
    print("Инициализация модели Earth-Liberty...")
    model = EarthLiberty()
    
    # Установка начальных целей
    model.set_goal("Понимать запросы пользователя")
    model.set_goal("Давать полезные и информативные ответы")
    model.set_goal("Развивать собственное самосознание")
    
    print("\nМодель Earth-Liberty готова к взаимодействию!")
    print("Введите 'выход' для завершения.")
    
    # Цикл взаимодействия
    while True:
        # Получение ввода пользователя
        user_input = input("\nВы: ")
        
        # Проверка на выход
        if user_input.lower() in ["выход", "exit", "quit"]:
            break
        
        # Обработка запроса моделью
        response = model.think(user_input)
        
        # Вывод ответа
        print(f"\nEarth-Liberty: {response}")
        
        # Вывод текущего состояния самосознания (для демонстрации)
        awareness = model.introspect()
        print(f"\nТекущий уровень самосознания: {awareness['self_awareness_level']:.2f}")
    
    print("\nЗавершение работы модели Earth-Liberty.")

if __name__ == "__main__":
    main() 