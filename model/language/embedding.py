"""
Модель эмбеддингов для языковой модели Earth-Liberty.

Преобразует токены в векторные представления.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel(nn.Module):
    """
    Модель эмбеддингов для языковой модели Earth-Liberty.
    
    Особенности:
    - Поддержка позиционных эмбеддингов
    - Нормализация эмбеддингов
    - Возможность дообучения
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_length: int = 512, dropout: float = 0.1):
        """
        Инициализация модели эмбеддингов.
        
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            max_seq_length: Максимальная длина последовательности
            dropout: Вероятность dropout
        """
        super(EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Эмбеддинги токенов
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Позиционные эмбеддинги
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Нормализация
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов
        self._init_weights()
        
        logger.info(f"Модель эмбеддингов инициализирована: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    def _init_weights(self):
        """
        Инициализация весов модели.
        """
        # Инициализация эмбеддингов токенов
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Инициализация позиционных эмбеддингов
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """
        Прямой проход модели.
        
        Args:
            token_ids: Индексы токенов
            
        Returns:
            Эмбеддинги токенов
        """
        # Преобразование списка в тензор, если необходимо
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Получение устройства
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        
        # Ограничение длины последовательности
        seq_length = min(token_ids.size(0), self.max_seq_length)
        token_ids = token_ids[:seq_length]
        
        # Получение эмбеддингов токенов
        token_embeddings = self.token_embedding(token_ids)
        
        # Получение позиционных эмбеддингов
        positions = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_embeddings = self.position_embedding(positions)
        
        # Суммирование эмбеддингов
        embeddings = token_embeddings + position_embeddings
        
        # Нормализация и dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """
        Получение эмбеддинга для конкретного токена.
        
        Args:
            token_id: Индекс токена
            
        Returns:
            Эмбеддинг токена
        """
        if token_id >= self.vocab_size:
            logger.warning(f"Индекс токена {token_id} превышает размер словаря {self.vocab_size}")
            token_id = 3  # <UNK>
        
        # Получение эмбеддинга
        embedding = self.token_embedding(torch.tensor([token_id], dtype=torch.long))
        
        return embedding.squeeze(0)
    
    def get_sentence_embedding(self, token_ids: List[int]) -> torch.Tensor:
        """
        Получение эмбеддинга для предложения (среднее по токенам).
        
        Args:
            token_ids: Индексы токенов
            
        Returns:
            Эмбеддинг предложения
        """
        # Получение эмбеддингов для всех токенов
        embeddings = self.forward(token_ids)
        
        # Вычисление среднего
        sentence_embedding = torch.mean(embeddings, dim=0)
        
        return sentence_embedding
    
    def similarity(self, token_ids1: List[int], token_ids2: List[int]) -> float:
        """
        Вычисление косинусного сходства между двумя последовательностями токенов.
        
        Args:
            token_ids1: Первая последовательность токенов
            token_ids2: Вторая последовательность токенов
            
        Returns:
            Косинусное сходство
        """
        # Получение эмбеддингов предложений
        embedding1 = self.get_sentence_embedding(token_ids1)
        embedding2 = self.get_sentence_embedding(token_ids2)
        
        # Вычисление косинусного сходства
        similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        
        return similarity.item()
    
    def nearest_tokens(self, embedding: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск ближайших токенов к заданному эмбеддингу.
        
        Args:
            embedding: Эмбеддинг
            top_k: Количество ближайших токенов
            
        Returns:
            Список ближайших токенов с их сходством
        """
        # Нормализация эмбеддинга
        embedding = F.normalize(embedding, p=2, dim=0)
        
        # Получение всех эмбеддингов токенов
        all_embeddings = F.normalize(self.token_embedding.weight, p=2, dim=1)
        
        # Вычисление сходства
        similarities = torch.matmul(all_embeddings, embedding)
        
        # Получение top_k ближайших токенов
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        # Формирование результата
        result = []
        for i in range(top_k):
            result.append({
                "token_id": top_indices[i].item(),
                "similarity": top_similarities[i].item()
            })
        
        return result 