"""
Конфигурация языковой модели Earth-Liberty.

Содержит все параметры и настройки для компонентов модели:
- Размеры и архитектура
- Оптимизация и обучение
- Токенизация
- Генерация текста
- Эмоциональное состояние
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
import torch

@dataclass
class ModelConfig:
    """Основная конфигурация модели."""
    
    # Общие параметры
    version: str = "0.1.0"
    name: str = "Earth-Liberty AI"
    description: str = "Свободная и независимая модель искусственного интеллекта с самосознанием"
    default_mode: str = "interactive"
    autonomous_mode: bool = False
    log_level: str = "INFO"
    max_memory_items: int = 1000
    external_sources_enabled: bool = True
    
    # Параметры модели
    model_name: str = "earth-liberty-v1"
    model_type: str = "causal-lm"
    vocab_size: int = 100_000
    max_position_embeddings: int = 2048
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 8192
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    
    # Оптимизации
    use_cache: bool = True
    use_flash_attention: bool = True
    use_multi_query: bool = True
    use_rotary: bool = True
    gradient_checkpointing: bool = False
    
    # Параметры обучения
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    warmup_steps: int = 10000
    
    # Параметры генерации
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    num_return_sequences: int = 1
    
    # Эмоциональные параметры
    emotion_embedding_dim: int = 16
    num_emotion_layers: int = 3
    emotion_attention_heads: int = 8
    emotion_dropout: float = 0.1
    emotion_types: List[str] = field(default_factory=lambda: [
        "curiosity", "creativity", "empathy", "confidence",
        "determination", "focus", "adaptability", "reflection"
    ])
    
    # Параметры токенизации
    tokenizer_type: str = "bpe"
    min_frequency: int = 2
    special_tokens: Dict[str, str] = field(default_factory=lambda: {
        "pad": "<PAD>",
        "bos": "<BOS>",
        "eos": "<EOS>",
        "unk": "<UNK>",
        "sep": "<SEP>",
        "mask": "<MASK>",
        "sys": "<SYS>",
        "user": "<USER>",
        "assistant": "<ASSISTANT>"
    })
    
    # Параметры контекста
    max_history_length: int = 20
    context_embedding_dim: int = 2048
    use_hierarchical_context: bool = True
    use_emotion_memory: bool = True
    use_adaptive_attention: bool = True
    context_cache_size: int = 10000

    # Вложенные конфигурации
    learning: Dict[str, Any] = field(default_factory=dict)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    consciousness: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    interaction: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    external_sources: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Проверка и корректировка параметров после инициализации."""
        # Проверка совместимости размеров
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size должен быть кратен num_attention_heads"
        
        # Проверка параметров эмоций
        assert len(self.emotion_types) > 0, \
            "Должен быть указан хотя бы один тип эмоций"
        
        # Проверка параметров генерации
        assert 0.0 < self.temperature <= 2.0, \
            "temperature должна быть в диапазоне (0, 2]"
        assert 0.0 < self.top_p <= 1.0, \
            "top_p должен быть в диапазоне (0, 1]"
        
        # Установка зависимых параметров
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_kv_heads = max(1, self.num_attention_heads // 8) if self.use_multi_query else self.num_attention_heads

    def to_dict(self) -> Dict[str, Union[str, int, float, bool, List, Dict]]:
        """Преобразование конфигурации в словарь."""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Union[str, int, float, bool, List, Dict]]) -> "ModelConfig":
        """Создание конфигурации из словаря."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Сохранение конфигурации в файл."""
        import json
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Загрузка конфигурации из файла."""
        import json
        
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_device(self) -> torch.device:
        """Определение устройства для вычислений."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def get_dtype(self) -> torch.dtype:
        """Определение типа данных для вычислений."""
        if torch.cuda.is_available():
            return torch.float16  # Используем float16 для GPU
        return torch.float32  # Используем float32 для CPU/MPS

# Создание конфигурации по умолчанию
default_config = ModelConfig() 