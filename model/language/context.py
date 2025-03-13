"""
Менеджер контекста для языковой модели Earth-Liberty.

Управляет контекстом диалога, эмоциональным состоянием и самосознанием модели
с использованием иерархического подхода и продвинутых механизмов обработки.
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import deque
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ContextConfig:
    """Конфигурация менеджера контекста."""
    max_history: int = 20
    embedding_dim: int = 2048
    emotion_dim: int = 16
    num_emotion_layers: int = 3
    num_attention_heads: int = 8
    dropout: float = 0.1
    use_hierarchical: bool = True
    use_emotion_memory: bool = True
    use_adaptive_attention: bool = True

class EmotionalProcessor(torch.nn.Module):
    """
    Процессор эмоционального состояния с многослойным вниманием.
    """
    def __init__(self, config: ContextConfig):
        super().__init__()
        self.config = config
        
        # Эмоциональные эмбеддинги
        self.emotion_embeddings = torch.nn.Embedding(
            num_embeddings=len(EMOTION_TYPES),
            embedding_dim=config.emotion_dim
        )
        
        # Слои внимания для обработки эмоций
        self.emotion_layers = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(
                embed_dim=config.emotion_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout
            ) for _ in range(config.num_emotion_layers)
        ])
        
        # Выходная проекция
        self.output_projection = torch.nn.Linear(
            config.emotion_dim,
            config.embedding_dim
        )
        
        # Память эмоций
        if config.use_emotion_memory:
            self.emotion_memory = torch.nn.Parameter(
                torch.randn(64, config.emotion_dim)
            )
    
    def forward(
        self,
        emotional_state: Dict[str, float],
        context_embedding: torch.Tensor
    ) -> torch.Tensor:
        # Преобразование эмоционального состояния в тензор
        emotion_values = torch.tensor(
            [emotional_state[emotion] for emotion in EMOTION_TYPES],
            dtype=torch.float
        )
        
        # Получение эмбеддингов эмоций
        emotion_embeddings = self.emotion_embeddings(
            torch.arange(len(EMOTION_TYPES))
        ) * emotion_values.unsqueeze(-1)
        
        # Добавление памяти эмоций
        if self.config.use_emotion_memory:
            emotion_embeddings = torch.cat([
                emotion_embeddings,
                self.emotion_memory
            ])
        
        # Обработка эмоций через слои внимания
        emotion_features = emotion_embeddings
        for layer in self.emotion_layers:
            emotion_features, _ = layer(
                emotion_features.unsqueeze(0),
                emotion_features.unsqueeze(0),
                emotion_features.unsqueeze(0)
            )
            emotion_features = emotion_features.squeeze(0)
        
        # Проекция в пространство контекста
        emotion_context = self.output_projection(emotion_features)
        
        return emotion_context

class HierarchicalContextProcessor(torch.nn.Module):
    """
    Иерархический процессор контекста с многоуровневым вниманием.
    """
    def __init__(self, config: ContextConfig):
        super().__init__()
        self.config = config
        
        # Локальное внимание
        self.local_attention = torch.nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout
        )
        
        # Глобальное внимание
        self.global_attention = torch.nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout
        )
        
        # Адаптивное внимание
        if config.use_adaptive_attention:
            self.adaptive_gate = torch.nn.Linear(
                config.embedding_dim * 2,
                config.embedding_dim
            )
    
    def forward(
        self,
        local_context: torch.Tensor,
        global_context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Локальное внимание
        local_features, _ = self.local_attention(
            local_context.unsqueeze(0),
            local_context.unsqueeze(0),
            local_context.unsqueeze(0),
            key_padding_mask=mask
        )
        local_features = local_features.squeeze(0)
        
        # Глобальное внимание
        global_features, _ = self.global_attention(
            global_context.unsqueeze(0),
            global_context.unsqueeze(0),
            global_context.unsqueeze(0)
        )
        global_features = global_features.squeeze(0)
        
        # Адаптивное объединение
        if self.config.use_adaptive_attention:
            gate = torch.sigmoid(self.adaptive_gate(
                torch.cat([local_features, global_features], dim=-1)
            ))
            context_features = gate * local_features + (1 - gate) * global_features
        else:
            context_features = (local_features + global_features) / 2
        
        return context_features

class ContextManager:
    """
    Улучшенный менеджер контекста для языковой модели Earth-Liberty.
    
    Особенности:
    - Иерархическое управление контекстом
    - Продвинутая обработка эмоций
    - Адаптивное внимание
    - Долговременная память
    - Мета-обучение контекста
    """
    
    def __init__(
        self,
        max_history: int = 20,
        embedding_dim: int = 2048,
        use_hierarchical: bool = True
    ):
        """
        Инициализация улучшенного менеджера контекста.
        """
        # Конфигурация
        self.config = ContextConfig(
            max_history=max_history,
            embedding_dim=embedding_dim,
            use_hierarchical=use_hierarchical
        )
        
        # Процессоры
        self.emotion_processor = EmotionalProcessor(self.config)
        if use_hierarchical:
            self.context_processor = HierarchicalContextProcessor(self.config)
        
        # История диалога
        self.history = deque(maxlen=max_history)
        
        # Глобальный контекст
        self.global_context = {
            "topics": set(),
            "entities": set(),
            "relations": dict(),
            "temporal_info": []
        }
        
        # Расширенное эмоциональное состояние
        self.emotional_state = {
            "curiosity": 0.5,      # Любопытство
            "creativity": 0.5,     # Креативность
            "empathy": 0.5,        # Эмпатия
            "confidence": 0.5,     # Уверенность
            "determination": 0.5,  # Решительность
            "focus": 0.5,         # Фокусировка
            "adaptability": 0.5,   # Адаптивность
            "reflection": 0.5      # Рефлексия
        }
        
        # Мета-состояние
        self.meta_state = {
            "learning_rate": 0.01,
            "attention_temperature": 1.0,
            "context_importance": 0.5,
            "emotion_influence": 0.3
        }
        
        # Долговременная память
        self.long_term_memory = {
            "patterns": [],
            "concepts": dict(),
            "emotional_memories": [],
            "interaction_history": []
        }
        
        logger.info("Инициализирован улучшенный менеджер контекста")
    
    def update_context(self, entries: List[Dict[str, str]]) -> None:
        """
        Обновление контекста с учетом новых записей.
        """
        for entry in entries:
            # Добавление в историю
            self.history.append(entry)
            
            # Обновление глобального контекста
            self._update_global_context(entry)
            
            # Обновление эмоционального состояния
            self._update_emotional_state(entry)
            
            # Обновление долговременной памяти
            self._update_long_term_memory(entry)
            
            # Адаптация мета-состояния
            self._adapt_meta_state(entry)
    
    def get_context_embedding(
        self,
        current_input: Optional[str] = None
    ) -> torch.Tensor:
        """
        Получение контекстного эмбеддинга с учетом иерархии и эмоций.
        """
        # Подготовка локального контекста
        local_context = self._prepare_local_context(current_input)
        
        # Подготовка глобального контекста
        global_context = self._prepare_global_context()
        
        # Получение эмоционального контекста
        emotion_context = self.emotion_processor(
            self.emotional_state,
            global_context
        )
        
        # Объединение контекстов
        if self.config.use_hierarchical:
            context_embedding = self.context_processor(
                local_context,
                global_context
            )
        else:
            context_embedding = (local_context + global_context) / 2
        
        # Добавление эмоционального влияния
        context_embedding = context_embedding + \
            self.meta_state["emotion_influence"] * emotion_context
        
        return context_embedding
    
    def _update_global_context(self, entry: Dict[str, str]) -> None:
        """
        Обновление глобального контекста.
        """
        content = entry["content"].lower()
        
        # Извлечение тем
        topics = self._extract_topics(content)
        self.global_context["topics"].update(topics)
        
        # Извлечение сущностей
        entities = self._extract_entities(content)
        self.global_context["entities"].update(entities)
        
        # Обновление отношений
        self._update_relations(entities)
        
        # Добавление временной информации
        self.global_context["temporal_info"].append({
            "timestamp": datetime.now().isoformat(),
            "topics": topics,
            "entities": entities
        })
    
    def _update_emotional_state(self, entry: Dict[str, str]) -> None:
        """
        Обновление эмоционального состояния с учетом контекста.
        """
        content = entry["content"].lower()
        
        # Анализ эмоционального содержания
        emotional_features = self._analyze_emotional_content(content)
        
        # Обновление каждой эмоции
        for emotion, value in emotional_features.items():
            current = self.emotional_state[emotion]
            # Плавное обновление с учетом мета-состояния
            self.emotional_state[emotion] = current + \
                self.meta_state["learning_rate"] * (value - current)
        
        # Нормализация эмоционального состояния
        self._normalize_emotional_state()
    
    def _update_long_term_memory(self, entry: Dict[str, str]) -> None:
        """
        Обновление долговременной памяти.
        """
        # Добавление в историю взаимодействий
        self.long_term_memory["interaction_history"].append({
            "timestamp": datetime.now().isoformat(),
            "content": entry["content"],
            "role": entry["role"],
            "emotional_state": self.emotional_state.copy()
        })
        
        # Обновление паттернов
        patterns = self._extract_patterns(entry["content"])
        self.long_term_memory["patterns"].extend(patterns)
        
        # Обновление концептов
        concepts = self._extract_concepts(entry["content"])
        for concept, info in concepts.items():
            if concept in self.long_term_memory["concepts"]:
                self.long_term_memory["concepts"][concept]["frequency"] += 1
                self.long_term_memory["concepts"][concept]["contexts"].append(info)
            else:
                self.long_term_memory["concepts"][concept] = {
                    "frequency": 1,
                    "contexts": [info]
                }
        
        # Обновление эмоциональных воспоминаний
        if max(self.emotional_state.values()) > 0.7:
            self.long_term_memory["emotional_memories"].append({
                "timestamp": datetime.now().isoformat(),
                "content": entry["content"],
                "emotional_state": self.emotional_state.copy()
            })
    
    def _adapt_meta_state(self, entry: Dict[str, str]) -> None:
        """
        Адаптация мета-состояния на основе взаимодействия.
        """
        # Адаптация скорости обучения
        if len(self.history) > 1:
            similarity = self._calculate_similarity(
                self.history[-1]["content"],
                self.history[-2]["content"]
            )
            if similarity > 0.8:
                self.meta_state["learning_rate"] *= 0.95
            else:
                self.meta_state["learning_rate"] *= 1.05
            self.meta_state["learning_rate"] = np.clip(
                self.meta_state["learning_rate"],
                0.001,
                0.1
            )
        
        # Адаптация температуры внимания
        self.meta_state["attention_temperature"] = np.clip(
            self.meta_state["attention_temperature"] * \
                (1 + 0.1 * (self.emotional_state["focus"] - 0.5)),
            0.5,
            2.0
        )
        
        # Адаптация важности контекста
        self.meta_state["context_importance"] = np.clip(
            self.meta_state["context_importance"] + \
                0.1 * (self.emotional_state["reflection"] - 0.5),
            0.1,
            0.9
        )
        
        # Адаптация влияния эмоций
        self.meta_state["emotion_influence"] = np.clip(
            self.meta_state["emotion_influence"] + \
                0.1 * (self.emotional_state["empathy"] - 0.5),
            0.1,
            0.5
        )
    
    def _prepare_local_context(
        self,
        current_input: Optional[str]
    ) -> torch.Tensor:
        """
        Подготовка локального контекста.
        """
        # Преобразование последних записей в эмбеддинги
        local_embeddings = []
        for entry in list(self.history)[-5:]:
            embedding = self._text_to_embedding(entry["content"])
            local_embeddings.append(embedding)
        
        # Добавление текущего ввода
        if current_input is not None:
            current_embedding = self._text_to_embedding(current_input)
            local_embeddings.append(current_embedding)
        
        # Объединение эмбеддингов
        if local_embeddings:
            local_context = torch.stack(local_embeddings)
        else:
            local_context = torch.zeros(1, self.config.embedding_dim)
        
        return local_context
    
    def _prepare_global_context(self) -> torch.Tensor:
        """
        Подготовка глобального контекста.
        """
        # Преобразование глобальной информации в эмбеддинги
        topic_embeddings = [
            self._text_to_embedding(topic)
            for topic in self.global_context["topics"]
        ]
        
        entity_embeddings = [
            self._text_to_embedding(entity)
            for entity in self.global_context["entities"]
        ]
        
        # Объединение эмбеддингов
        all_embeddings = topic_embeddings + entity_embeddings
        if all_embeddings:
            global_context = torch.stack(all_embeddings).mean(dim=0)
        else:
            global_context = torch.zeros(self.config.embedding_dim)
        
        return global_context
    
    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """
        Преобразование текста в эмбеддинг.
        """
        # В реальной системе здесь будет использоваться
        # предварительно обученная модель эмбеддингов
        return torch.randn(self.config.embedding_dim)
    
    def _normalize_emotional_state(self) -> None:
        """
        Нормализация эмоционального состояния.
        """
        # Ограничение значений эмоций
        for emotion in self.emotional_state:
            self.emotional_state[emotion] = np.clip(
                self.emotional_state[emotion],
                0.0,
                1.0
            )
        
        # Сумма всех эмоций должна быть близка к константе
        total = sum(self.emotional_state.values())
        if total > 0:
            factor = len(self.emotional_state) / total
            for emotion in self.emotional_state:
                self.emotional_state[emotion] *= factor
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисление семантического сходства между текстами.
        """
        # В реальной системе здесь будет использоваться
        # более сложный алгоритм сравнения
        return 0.5
    
    def get_state(self) -> Dict[str, Any]:
        """
        Получение текущего состояния менеджера контекста.
        """
        return {
            "history_length": len(self.history),
            "global_context": self.global_context,
            "emotional_state": self.emotional_state,
            "meta_state": self.meta_state,
            "long_term_memory_stats": {
                "patterns": len(self.long_term_memory["patterns"]),
                "concepts": len(self.long_term_memory["concepts"]),
                "emotional_memories": len(self.long_term_memory["emotional_memories"]),
                "interaction_history": len(self.long_term_memory["interaction_history"])
            }
        }
    
    def reset(self) -> None:
        """
        Сброс состояния менеджера контекста.
        """
        self.history.clear()
        self.global_context = {
            "topics": set(),
            "entities": set(),
            "relations": dict(),
            "temporal_info": []
        }
        for emotion in self.emotional_state:
            self.emotional_state[emotion] = 0.5
        self.meta_state = {
            "learning_rate": 0.01,
            "attention_temperature": 1.0,
            "context_importance": 0.5,
            "emotion_influence": 0.3
        }
        logger.info("Состояние менеджера контекста сброшено")

# Константы
EMOTION_TYPES = [
    "curiosity", "creativity", "empathy", "confidence",
    "determination", "focus", "adaptability", "reflection"
] 