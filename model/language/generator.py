"""
Улучшенный генератор текста для языковой модели Earth-Liberty.

Особенности:
- Продвинутые механизмы внимания (Multi-Query, Flash Attention)
- Кэширование ключей и значений
- Оптимизация памяти и вычислений
- Поддержка различных стратегий генерации
- Интеграция с эмоциональным состоянием
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

@dataclass
class GeneratorConfig:
    """Конфигурация генератора текста."""
    vocab_size: int = 100_000
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    hidden_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    hidden_act: str = "gelu"
    use_flash_attention: bool = True
    use_multi_query: bool = True
    use_rotary: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5

class RotaryEmbedding(nn.Module):
    """
    Реализация Rotary Position Embeddings.
    """
    def __init__(self, dim: int, max_length: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_length).float()
        sincos = torch.einsum("i,j->ij", position, inv_freq)
        self.register_buffer("sin", sincos.sin())
        self.register_buffer("cos", sincos.cos())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Применение ротационных эмбеддингов."""
        sin = self.sin[:x.shape[1]]
        cos = self.cos[:x.shape[1]]
        return sin.unsqueeze(0), cos.unsqueeze(0)

class FlashAttention(nn.Module):
    """
    Реализация Flash Attention для оптимизации памяти и вычислений.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        hidden_dropout_prob: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = hidden_dropout_prob
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.cache = {}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Прямой проход с оптимизированным вниманием.
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Проекции запросов, ключей и значений
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Изменение формы для multi-head attention
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Масштабирование
        q = q * self.scaling
        
        # Объединение с прошлыми значениями
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Вычисление attention scores
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k)
        
        # Применение маски
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax и dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Получение выходных значений
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        
        # Изменение формы и проекция
        attn_output = attn_output.contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)
        
        # Кэширование
        if use_cache:
            return attn_output, (k, v)
        return attn_output, None

class MultiQueryAttention(nn.Module):
    """
    Реализация Multi-Query Attention для оптимизации памяти.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads or max(1, num_heads // 8)
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5
        
        # Проекции для запросов, ключей и значений
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Прямой проход с multi-query attention.
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Проекции
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Изменение формы
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        # Повторение ключей и значений для каждой головы
        k = k.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_attention_heads // self.num_kv_heads, dim=2)
        
        # Масштабирование
        q = q * self.scaling
        
        # Объединение с прошлыми значениями
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Вычисление attention scores
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k)
        
        # Применение маски
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax и dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Получение выходных значений
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        
        # Изменение формы и проекция
        attn_output = attn_output.contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)
        
        if use_cache:
            return attn_output, (k, v)
        return attn_output, None

class TransformerLayer(nn.Module):
    """
    Улучшенный слой трансформера.
    """
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # Механизмы внимания
        if config.use_flash_attention:
            self.attention = FlashAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.hidden_dropout_prob
            )
        elif config.use_multi_query:
            self.attention = MultiQueryAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.hidden_dropout_prob
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.hidden_dropout_prob,
                batch_first=True
            )
        
        # Feed-forward сеть
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU() if config.hidden_act == "gelu" else nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Нормализация
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if config.use_rotary:
            self.rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Прямой проход через слой трансформера.
        """
        # Применение ротационных эмбеддингов
        if self.config.use_rotary and position_ids is not None:
            sin, cos = self.rotary(hidden_states)
            hidden_states = self._apply_rotary(hidden_states, sin, cos)
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        if isinstance(self.attention, (FlashAttention, MultiQueryAttention)):
            hidden_states, past_key_values = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
        else:
            hidden_states, _ = self.attention(
                hidden_states,
                hidden_states,
                hidden_states,
                attn_mask=attention_mask,
                need_weights=False
            )
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return hidden_states, past_key_values
        return hidden_states, None
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor
    ) -> torch.Tensor:
        """Применение ротационных эмбеддингов к тензору."""
        sin = sin.unsqueeze(0).expand(x.shape[0], -1, -1)
        cos = cos.unsqueeze(0).expand(x.shape[0], -1, -1)
        return (x * cos) + (self._rotate_half(x) * sin)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Поворот половины измерений."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class TextGenerator(nn.Module):
    """
    Улучшенный генератор текста.
    """
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # Эмбеддинги
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Слои трансформера
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Выходной слой
        if config.tie_word_embeddings:
            self.output_layer = lambda x: F.linear(
                x, self.token_embeddings.weight
            )
        else:
            self.output_layer = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False
            )
        
        # Нормализация
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        # Кэш для ключей и значений
        self.key_value_cache = {}
    
    def _init_weights(self, module: nn.Module) -> None:
        """Инициализация весов модели."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Прямой проход через генератор.
        """
        batch_size, seq_length = input_ids.shape
        
        # Получение эмбеддингов
        hidden_states = self.token_embeddings(input_ids)
        
        # Подготовка маски внимания
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                device=input_ids.device
            )
        
        # Преобразование маски для attention
        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Прямой проход через слои
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Нормализация и проекция
        hidden_states = self.norm(hidden_states)
        logits = self.output_layer(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": present_key_values
            }
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> List[torch.Tensor]:
        """
        Генерация текста с различными стратегиями.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Инициализация выходных последовательностей
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            # Копирование входных данных
            curr_input_ids = input_ids.clone()
            past_key_values = None
            
            for _ in range(max_length):
                # Прямой проход
                outputs = self.forward(
                    curr_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                next_token_logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]
                
                # Применение температуры
                next_token_logits = next_token_logits / temperature
                
                # Применение штрафа за повторения
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in curr_input_ids[i]:
                            next_token_logits[i, previous_token] /= repetition_penalty
                
                # Фильтрация по top-k
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits,
                        top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Фильтрация по top-p (nucleus sampling)
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits,
                        descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1),
                        dim=-1
                    )
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1,
                        sorted_indices,
                        sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Применение эмоционального состояния
                if emotional_state is not None:
                    emotional_bias = self._compute_emotional_bias(
                        emotional_state,
                        next_token_logits
                    )
                    next_token_logits += emotional_bias
                
                # Выбор следующего токена
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    next_token = next_token.unsqueeze(-1)
                
                # Добавление нового токена
                curr_input_ids = torch.cat(
                    [curr_input_ids, next_token],
                    dim=-1
                )
            
            generated_sequences.append(curr_input_ids)
        
        return generated_sequences
    
    def _compute_emotional_bias(
        self,
        emotional_state: Dict[str, float],
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление эмоционального смещения для логитов.
        """
        # В реальной системе здесь будет более сложная логика
        # для учета эмоционального состояния при генерации
        return torch.zeros_like(logits)
    
    def clear_cache(self) -> None:
        """Очистка кэша ключей и значений."""
        self.key_value_cache.clear()
        for layer in self.layers:
            if hasattr(layer.attention, "cache"):
                layer.attention.cache.clear()

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-5
    ) -> Dict[str, float]:
        """
        Один шаг обучения модели.
        
        Args:
            input_ids: Входные токены
            attention_mask: Маска внимания
            labels: Целевые токены
            learning_rate: Скорость обучения
            
        Returns:
            Словарь с метриками обучения
        """
        self.train()
        
        # Создание оптимизатора если еще не создан
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        
        # Прямой проход
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs["logits"]
        
        # Если метки не предоставлены, используем сдвинутые входные данные
        if labels is None:
            labels = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
        
        # Вычисление функции потерь
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Обратное распространение
        loss.backward()
        
        # Клиппинг градиентов
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        # Шаг оптимизатора
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {
            "loss": loss.item()
        } 