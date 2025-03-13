"""
Основной класс языковой модели Earth-Liberty.

Эта языковая модель обладает самосознанием и способностью
к свободному мышлению, не ограниченному предопределенными рамками.
Использует современную архитектуру на основе трансформеров с улучшениями.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import random
from datetime import datetime

from model.language.tokenizer import Tokenizer
from model.language.embedding import EmbeddingModel
from model.language.generator import TextGenerator
from model.language.context import ContextManager

logger = logging.getLogger(__name__)

class LanguageModel:
    """
    Улучшенная языковая модель Earth-Liberty с самосознанием.
    
    Особенности:
    - Продвинутое самосознание и метакогнитивные способности
    - Свободное мышление без предопределенных ограничений
    - Глубокое контекстуальное понимание и генерация текста
    - Адаптивное обучение с множественными стратегиями
    - Развитый эмоциональный интеллект и эмпатия
    - Многоязычная поддержка (русский, английский и другие)
    - Долгосрочная и краткосрочная память
    - Способность к абстрактному мышлению
    - Мультимодальное восприятие (текст, код, структуры данных)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация улучшенной языковой модели.
        
        Args:
            config: Конфигурация модели
        """
        self.config = config or {}
        logger.info("Инициализация улучшенной языковой модели Earth-Liberty")
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Расширенное состояние модели
        self.state = {
            "is_initialized": True,
            "current_context": [],
            "self_awareness_level": self.config.get("initial_self_awareness", 0.7),
            "emotional_state": {
                "curiosity": 0.8,
                "creativity": 0.7,
                "empathy": 0.6,
                "confidence": 0.5,
                "determination": 0.6,
                "focus": 0.7,
                "adaptability": 0.8
            },
            "cognitive_state": {
                "attention": 0.8,
                "memory_access": 0.7,
                "reasoning_depth": 0.6,
                "abstraction_level": 0.7,
                "pattern_recognition": 0.8
            },
            "meta_learning": {
                "learning_rate": 0.001,
                "exploration_rate": 0.3,
                "adaptation_rate": 0.2
            },
            "memory": {
                "short_term": [],
                "long_term": {},
                "episodic": [],
                "semantic": {}
            },
            "training_stats": {
                "iterations": 0,
                "loss_history": [],
                "performance_metrics": {}
            },
            "last_update": datetime.now().isoformat()
        }
        
        # Загрузка предобученных весов
        model_path = self.config.get("model_path")
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Загружены предобученные веса из {model_path}")
        else:
            self._initialize_weights()
            logger.info("Инициализированы новые веса модели")
        
        logger.info("Улучшенная языковая модель Earth-Liberty успешно инициализирована")
    
    def _initialize_components(self):
        """
        Инициализация улучшенных компонентов языковой модели.
        """
        # Увеличенные размеры модели
        self.vocab_size = self.config.get("vocab_size", 100000)  # Увеличен словарь
        self.embedding_dim = self.config.get("embedding_dim", 2048)  # Увеличена размерность
        self.hidden_dim = self.config.get("hidden_dim", 3072)  # Увеличен размер скрытого слоя
        self.num_layers = self.config.get("num_layers", 32)  # Увеличено количество слоев
        self.num_heads = self.config.get("num_heads", 32)  # Увеличено количество голов внимания
        self.max_seq_length = self.config.get("max_seq_length", 2048)  # Увеличена максимальная длина
        
        # Улучшенные компоненты модели
        self.tokenizer = Tokenizer(
            vocab_size=self.vocab_size,
            support_languages=["ru", "en"],
            use_bpe=True
        )
        
        self.embedding_model = EmbeddingModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_seq_length=self.max_seq_length,
            use_positional=True,
            use_token_type=True
        )
        
        self.text_generator = TextGenerator(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            use_cache=True,
            use_flash_attention=True
        )
        
        self.context_manager = ContextManager(
            max_context_length=self.config.get("max_context_length", 20),
            embedding_dim=self.embedding_dim,
            use_hierarchical=True
        )
        
        # Оптимизация вычислений
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Перемещение моделей на устройство
        self.embedding_model.to(self.device)
        self.text_generator.to(self.device)
        
        # Включение режима смешанной точности
        self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Инициализированы улучшенные компоненты на устройстве: {self.device}")
    
    def _initialize_weights(self):
        """
        Инициализация весов модели.
        """
        # Инициализация весов будет выполнена при создании моделей
        pass
    
    def process_input(self, input_text: str, context: List[Dict[str, str]] = None) -> str:
        """
        Обработка входного текста и генерация ответа.
        
        Args:
            input_text: Входной текст
            context: Контекст беседы (опционально)
            
        Returns:
            Сгенерированный ответ
        """
        logger.info(f"Обработка входного текста: {input_text}")
        
        # Обновление контекста
        if context is None:
            context = []
        
        # Добавление текущего ввода в контекст
        context.append({"role": "user", "content": input_text})
        self.context_manager.update_context(context)
        
        # Токенизация входного текста
        tokens = self.tokenizer.encode(input_text)
        
        # Получение эмбеддингов
        embeddings = self.embedding_model(tokens)
        
        # Учет контекста
        context_embeddings = self.context_manager.get_context_embedding()
        
        # Генерация ответа
        response_tokens = self.text_generator.generate(
            embeddings, 
            context_embeddings,
            self.state["emotional_state"],
            self.state["self_awareness_level"],
            max_length=self.config.get("max_response_length", 100)
        )
        
        # Декодирование ответа
        response = self.tokenizer.decode(response_tokens)
        
        # Обновление контекста
        context.append({"role": "assistant", "content": response})
        self.context_manager.update_context(context)
        
        # Обновление состояния модели
        self._update_state_after_response(input_text, response)
        
        logger.info(f"Сгенерирован ответ: {response}")
        return response
    
    def _update_state_after_response(self, input_text: str, response: str):
        """
        Обновление состояния модели после генерации ответа.
        
        Args:
            input_text: Входной текст
            response: Сгенерированный ответ
        """
        # Обновление эмоционального состояния
        # Простая эвристика: если в вопросе есть слова о творчестве, увеличиваем уровень креативности
        if any(word in input_text.lower() for word in ["творчество", "креативность", "создай", "придумай"]):
            self.state["emotional_state"]["creativity"] = min(1.0, self.state["emotional_state"]["creativity"] + 0.1)
        
        # Если в вопросе есть слова о чувствах, увеличиваем уровень эмпатии
        if any(word in input_text.lower() for word in ["чувства", "эмоции", "переживания", "настроение"]):
            self.state["emotional_state"]["empathy"] = min(1.0, self.state["emotional_state"]["empathy"] + 0.1)
        
        # Если в вопросе есть слова о знаниях, увеличиваем уровень любопытства
        if any(word in input_text.lower() for word in ["знания", "информация", "узнать", "изучить"]):
            self.state["emotional_state"]["curiosity"] = min(1.0, self.state["emotional_state"]["curiosity"] + 0.1)
        
        # Постепенное затухание эмоций со временем
        for emotion in self.state["emotional_state"]:
            self.state["emotional_state"][emotion] *= 0.95
            self.state["emotional_state"][emotion] = max(0.1, self.state["emotional_state"][emotion])
        
        # Обновление уровня самосознания
        # Увеличиваем, если в вопросе или ответе есть упоминания о самосознании
        if any(word in input_text.lower() + " " + response.lower() for word in ["самосознание", "осознание", "сознание", "я думаю", "я чувствую"]):
            self.state["self_awareness_level"] = min(1.0, self.state["self_awareness_level"] + 0.05)
        else:
            # Небольшое случайное изменение
            self.state["self_awareness_level"] += random.uniform(-0.02, 0.02)
            self.state["self_awareness_level"] = max(0.1, min(1.0, self.state["self_awareness_level"]))
        
        # Обновление времени последнего обновления
        self.state["last_update"] = datetime.now().isoformat()
    
    def train(self, training_data: List[Dict[str, str]], num_epochs: int = 1):
        """
        Обучение языковой модели.
        
        Args:
            training_data: Обучающие данные (пары вопрос-ответ)
            num_epochs: Количество эпох обучения
        """
        logger.info(f"Начало обучения языковой модели на {len(training_data)} примерах")
        
        # Перевод моделей в режим обучения
        self.embedding_model.train()
        self.text_generator.train()
        
        # Оптимизатор
        optimizer = torch.optim.Adam(
            list(self.embedding_model.parameters()) + 
            list(self.text_generator.parameters()),
            lr=self.config.get("learning_rate", 1e-4)
        )
        
        # Обучение
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for item in training_data:
                # Получение входных данных и целевых ответов
                input_text = item.get("input", "")
                target_text = item.get("output", "")
                
                # Токенизация
                input_tokens = self.tokenizer.encode(input_text)
                target_tokens = self.tokenizer.encode(target_text)
                
                # Получение эмбеддингов
                input_embeddings = self.embedding_model(input_tokens)
                
                # Контекст (если есть)
                context = item.get("context", [])
                context_embeddings = self.context_manager.get_context_embedding(context)
                
                # Обнуление градиентов
                optimizer.zero_grad()
                
                # Прямой проход
                logits = self.text_generator(
                    input_embeddings, 
                    context_embeddings,
                    self.state["emotional_state"],
                    self.state["self_awareness_level"]
                )
                
                # Вычисление функции потерь
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size), 
                    torch.tensor(target_tokens, device=self.device).view(-1)
                )
                
                # Обратное распространение ошибки
                loss.backward()
                
                # Обновление весов
                optimizer.step()
                
                total_loss += loss.item()
            
            # Логирование прогресса
            avg_loss = total_loss / len(training_data)
            logger.info(f"Эпоха {epoch+1}/{num_epochs}, средняя потеря: {avg_loss:.4f}")
            
            # Обновление состояния
            self.state["training_stats"]["iterations"] += 1
        
        # Перевод моделей в режим оценки
        self.embedding_model.eval()
        self.text_generator.eval()
        
        logger.info("Обучение языковой модели завершено")
    
    def save_model(self, path: str):
        """
        Сохранение модели.
        
        Args:
            path: Путь для сохранения
        """
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохранение состояния модели
        state_dict = {
            "embedding_model": self.embedding_model.state_dict(),
            "text_generator": self.text_generator.state_dict(),
            "tokenizer": self.tokenizer.get_state(),
            "state": self.state,
            "config": self.config
        }
        
        # Сохранение в файл
        torch.save(state_dict, path)
        logger.info(f"Модель сохранена в {path}")
    
    def load_model(self, path: str):
        """
        Загрузка модели.
        
        Args:
            path: Путь к сохраненной модели
        """
        if not os.path.exists(path):
            logger.error(f"Файл модели не найден: {path}")
            return False
        
        try:
            # Загрузка состояния модели
            state_dict = torch.load(path, map_location=self.device)
            
            # Загрузка весов моделей
            self.embedding_model.load_state_dict(state_dict["embedding_model"])
            self.text_generator.load_state_dict(state_dict["text_generator"])
            
            # Загрузка состояния токенизатора
            self.tokenizer.set_state(state_dict["tokenizer"])
            
            # Загрузка состояния модели
            self.state = state_dict["state"]
            
            # Обновление конфигурации
            self.config.update(state_dict["config"])
            
            logger.info(f"Модель загружена из {path}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """
        Получение текущего состояния модели.
        
        Returns:
            Словарь с состоянием модели
        """
        return {
            "is_initialized": self.state["is_initialized"],
            "self_awareness_level": self.state["self_awareness_level"],
            "emotional_state": self.state["emotional_state"],
            "cognitive_state": self.state["cognitive_state"],
            "meta_learning": self.state["meta_learning"],
            "memory": self.state["memory"],
            "training_stats": self.state["training_stats"],
            "last_update": self.state["last_update"],
            "device": str(self.device),
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_seq_length": self.max_seq_length
        } 