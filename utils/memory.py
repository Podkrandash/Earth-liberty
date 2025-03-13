"""
Утилиты для работы с памятью модели Earth-Liberty.
"""

import os
import json
import pickle
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class Memory:
    """
    Класс для работы с памятью модели Earth-Liberty.
    Обеспечивает сохранение и загрузку воспоминаний, а также
    поиск релевантных воспоминаний.
    """
    
    def __init__(self, memory_path: Optional[str] = None):
        """
        Инициализация памяти.
        
        Args:
            memory_path: Путь к директории для хранения памяти (опционально)
        """
        self.memory_path = memory_path or os.path.join(os.getcwd(), "data", "memory")
        self.memories = []
        self.memory_index = {}  # Простой индекс для поиска
        
        # Создание директории для памяти, если она не существует
        os.makedirs(self.memory_path, exist_ok=True)
        
        logger.info(f"Память инициализирована, путь: {self.memory_path}")
    
    def add_memory(self, memory_data: Dict[str, Any]) -> int:
        """
        Добавление нового воспоминания.
        
        Args:
            memory_data: Данные воспоминания
            
        Returns:
            Индекс добавленного воспоминания
        """
        # Добавление метаданных
        memory_data["id"] = len(self.memories)
        memory_data["timestamp"] = "current_time"  # В реальной системе здесь будет реальное время
        
        # Добавление воспоминания в список
        self.memories.append(memory_data)
        
        # Индексирование воспоминания
        self._index_memory(memory_data)
        
        logger.debug(f"Добавлено новое воспоминание, id: {memory_data['id']}")
        return memory_data["id"]
    
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """
        Получение воспоминания по идентификатору.
        
        Args:
            memory_id: Идентификатор воспоминания
            
        Returns:
            Данные воспоминания или None, если воспоминание не найдено
        """
        if 0 <= memory_id < len(self.memories):
            return self.memories[memory_id]
        return None
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск релевантных воспоминаний.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список релевантных воспоминаний
        """
        # Простой поиск по ключевым словам
        query_words = set(query.lower().split())
        
        # Оценка релевантности каждого воспоминания
        relevance_scores = []
        
        for memory in self.memories:
            score = 0
            
            # Проверка наличия ключевых слов в воспоминании
            memory_text = memory.get("text", "").lower()
            for word in query_words:
                if word in memory_text:
                    score += 1
            
            # Учет дополнительных факторов (например, времени)
            # В реальной системе здесь будет более сложная логика
            
            if score > 0:
                relevance_scores.append((memory, score))
        
        # Сортировка по релевантности
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Возврат наиболее релевантных воспоминаний
        return [memory for memory, score in relevance_scores[:limit]]
    
    def save_memories(self, file_path: Optional[str] = None) -> bool:
        """
        Сохранение всех воспоминаний в файл.
        
        Args:
            file_path: Путь для сохранения (опционально)
            
        Returns:
            True в случае успеха, False в случае ошибки
        """
        if file_path is None:
            file_path = os.path.join(self.memory_path, "memories.json")
        
        try:
            # Создание директории, если она не существует
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Сохранение воспоминаний в формате JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Воспоминания сохранены в файл: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при сохранении воспоминаний: {e}")
            return False
    
    def load_memories(self, file_path: Optional[str] = None) -> bool:
        """
        Загрузка воспоминаний из файла.
        
        Args:
            file_path: Путь к файлу с воспоминаниями (опционально)
            
        Returns:
            True в случае успеха, False в случае ошибки
        """
        if file_path is None:
            file_path = os.path.join(self.memory_path, "memories.json")
        
        if not os.path.exists(file_path):
            logger.warning(f"Файл с воспоминаниями не найден: {file_path}")
            return False
        
        try:
            # Загрузка воспоминаний из файла JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
            
            # Перестроение индекса
            self._rebuild_index()
            
            logger.info(f"Воспоминания загружены из файла: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке воспоминаний: {e}")
            return False
    
    def clear_memories(self) -> None:
        """
        Очистка всех воспоминаний.
        """
        self.memories = []
        self.memory_index = {}
        logger.info("Все воспоминания очищены")
    
    def _index_memory(self, memory: Dict[str, Any]) -> None:
        """
        Индексирование воспоминания для быстрого поиска.
        
        Args:
            memory: Данные воспоминания
        """
        # Простая индексация по словам
        if "text" in memory:
            words = set(memory["text"].lower().split())
            
            for word in words:
                if word not in self.memory_index:
                    self.memory_index[word] = []
                
                self.memory_index[word].append(memory["id"])
    
    def _rebuild_index(self) -> None:
        """
        Перестроение индекса для всех воспоминаний.
        """
        self.memory_index = {}
        
        for memory in self.memories:
            self._index_memory(memory) 