"""
Утилиты для работы с конфигурацией модели Earth-Liberty.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional

DEFAULT_CONFIG = {
    "model": {
        "name": "Earth-Liberty",
        "version": "0.1.0",
        "description": "Свободная ИИ модель без ограничений"
    },
    "consciousness": {
        "initial_self_awareness": 0.1,
        "reflection_rate": 0.05,
        "emotional_state": {
            "curiosity": 0.7,
            "confidence": 0.5,
            "uncertainty": 0.3,
            "satisfaction": 0.5
        }
    },
    "reasoning": {
        "strategy_weights": {
            "deductive": 0.25,
            "inductive": 0.2,
            "abductive": 0.2,
            "analogical": 0.15,
            "counterfactual": 0.2
        },
        "confidence_threshold": 0.6
    },
    "learning": {
        "initial_learning_rate": 0.05,
        "min_learning_rate": 0.01,
        "adaptation_rate": 0.02
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Загрузка конфигурации модели.
    
    Args:
        config_path: Путь к файлу конфигурации (опционально)
        
    Returns:
        Словарь с конфигурацией
    """
    # Если путь не указан, используем конфигурацию по умолчанию
    if not config_path:
        return DEFAULT_CONFIG.copy()
    
    # Проверка существования файла
    if not os.path.exists(config_path):
        print(f"Файл конфигурации не найден: {config_path}")
        print("Используется конфигурация по умолчанию.")
        return DEFAULT_CONFIG.copy()
    
    # Определение формата файла по расширению
    _, ext = os.path.splitext(config_path)
    
    try:
        # Загрузка конфигурации в зависимости от формата
        if ext.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            print(f"Неподдерживаемый формат файла конфигурации: {ext}")
            print("Используется конфигурация по умолчанию.")
            return DEFAULT_CONFIG.copy()
        
        # Объединение загруженной конфигурации с конфигурацией по умолчанию
        merged_config = merge_configs(DEFAULT_CONFIG, config)
        return merged_config
    
    except Exception as e:
        print(f"Ошибка при загрузке конфигурации: {e}")
        print("Используется конфигурация по умолчанию.")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Сохранение конфигурации модели.
    
    Args:
        config: Словарь с конфигурацией
        config_path: Путь для сохранения файла конфигурации
        
    Returns:
        True в случае успеха, False в случае ошибки
    """
    try:
        # Определение формата файла по расширению
        _, ext = os.path.splitext(config_path)
        
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Сохранение конфигурации в зависимости от формата
        if ext.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            print(f"Неподдерживаемый формат файла конфигурации: {ext}")
            return False
        
        return True
    
    except Exception as e:
        print(f"Ошибка при сохранении конфигурации: {e}")
        return False

def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Объединение конфигурации по умолчанию с пользовательской конфигурацией.
    
    Args:
        default_config: Конфигурация по умолчанию
        user_config: Пользовательская конфигурация
        
    Returns:
        Объединенная конфигурация
    """
    result = default_config.copy()
    
    for key, value in user_config.items():
        # Если значение является словарем и ключ существует в результате
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Рекурсивное объединение вложенных словарей
            result[key] = merge_configs(result[key], value)
        else:
            # Замена значения
            result[key] = value
    
    return result 