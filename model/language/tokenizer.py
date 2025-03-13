"""
Улучшенный токенизатор для языковой модели Earth-Liberty.

Поддерживает:
- Byte-Pair Encoding (BPE)
- Мультиязычность (русский и английский)
- Кэширование токенов
- Динамическое обновление словаря
- Специальные токены для диалога и эмоций
"""

import os
import json
import logging
import regex as re
from typing import List, Dict, Set, Optional, Union, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
import sentencepiece as spm

logger = logging.getLogger(__name__)

@dataclass
class TokenizerConfig:
    """Конфигурация токенизатора."""
    vocab_size: int = 100_000
    min_freq: int = 2
    bpe_vocab_size: int = 50_000
    max_length: int = 2048
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    sep_token: str = "<SEP>"
    mask_token: str = "<MASK>"
    sys_token: str = "<SYS>"
    user_token: str = "<USER>"
    assistant_token: str = "<ASSISTANT>"
    emotion_tokens: List[str] = None
    cache_size: int = 10_000
    use_bpe: bool = True
    use_sentencepiece: bool = True

class BPETokenizer:
    """
    Реализация Byte-Pair Encoding токенизации.
    """
    def __init__(self, vocab_size: int = 50_000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def train(self, texts: List[str]) -> None:
        """Обучение BPE на корпусе текстов."""
        # Подсчет частот символов
        word_freqs = defaultdict(int)
        for text in texts:
            words = self.pattern.findall(text)
            for word in words:
                word_freqs[" ".join(list(word.strip()))] += 1
        
        # Инициализация словаря символами
        self.vocab = {c: i for i, c in enumerate(set("".join(word_freqs.keys())))}
        self.inverse_vocab = {i: c for c, i in self.vocab.items()}
        
        # Итеративное слияние пар
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            # Подсчет пар
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
                
            # Выбор лучшей пары
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = len(self.vocab)
            
            # Обновление словаря
            self.vocab[best_pair[0] + best_pair[1]] = len(self.vocab)
            self.inverse_vocab[len(self.vocab) - 1] = best_pair[0] + best_pair[1]
            
            # Обновление частот
            self._update_frequencies(word_freqs, best_pair)
    
    def _get_pairs(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Подсчет частот пар символов."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _update_frequencies(
        self,
        word_freqs: Dict[str, int],
        pair: Tuple[str, str]
    ) -> None:
        """Обновление частот после слияния пары."""
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = word.replace(f"{pair[0]} {pair[1]}", f"{pair[0]}{pair[1]}")
            new_word_freqs[new_word] = freq
        word_freqs.clear()
        word_freqs.update(new_word_freqs)
    
    def encode(self, text: str) -> List[int]:
        """Кодирование текста в последовательность токенов."""
        words = self.pattern.findall(text)
        tokens = []
        for word in words:
            word = " ".join(list(word.strip()))
            while True:
                min_pair = None
                min_idx = float("inf")
                for pair, idx in self.merges.items():
                    try:
                        start = word.index(f"{pair[0]} {pair[1]}")
                        if start < min_idx:
                            min_idx = start
                            min_pair = pair
                    except ValueError:
                        continue
                if min_pair is None:
                    break
                word = word.replace(
                    f"{min_pair[0]} {min_pair[1]}",
                    f"{min_pair[0]}{min_pair[1]}"
                )
            tokens.extend(
                self.vocab.get(symbol, self.vocab.get("<UNK>"))
                for symbol in word.split()
            )
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Декодирование последовательности токенов в текст."""
        return "".join(
            self.inverse_vocab.get(token, "<UNK>")
            for token in tokens
        ).replace(" ", "")

class Tokenizer:
    """
    Улучшенный токенизатор с поддержкой BPE и мультиязычности.
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        """
        Инициализация токенизатора.
        """
        self.config = config or TokenizerConfig()
        
        # Специальные токены
        self.special_tokens = {
            "pad": self.config.pad_token,
            "bos": self.config.bos_token,
            "eos": self.config.eos_token,
            "unk": self.config.unk_token,
            "sep": self.config.sep_token,
            "mask": self.config.mask_token,
            "sys": self.config.sys_token,
            "user": self.config.user_token,
            "assistant": self.config.assistant_token
        }
        
        # Эмоциональные токены
        self.emotion_tokens = self.config.emotion_tokens or [
            "<HAPPY>", "<SAD>", "<ANGRY>", "<SURPRISED>",
            "<CURIOUS>", "<CONFIDENT>", "<CONFUSED>", "<EXCITED>"
        ]
        
        # Инициализация токенизаторов
        self.bpe = BPETokenizer(self.config.bpe_vocab_size) if self.config.use_bpe else None
        self.sp = None
        if self.config.use_sentencepiece:
            self.sp = spm.SentencePieceProcessor()
            # В реальной системе здесь будет загрузка предобученной модели
        
        # Словарь и обратный словарь
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Кэш токенизации
        self.encode_cache = lru_cache(maxsize=self.config.cache_size)(self._encode)
        self.decode_cache = lru_cache(maxsize=self.config.cache_size)(self._decode)
        
        # Статистика использования
        self.token_frequencies = Counter()
        self.sequence_lengths = []
        
        self._initialize_vocab()
        logger.info(
            f"Инициализирован токенизатор (размер словаря: {len(self.vocab)})"
        )
    
    def _initialize_vocab(self) -> None:
        """
        Инициализация базового словаря.
        """
        # Добавление специальных токенов
        for token in self.special_tokens.values():
            self._add_token(token)
        
        # Добавление эмоциональных токенов
        for token in self.emotion_tokens:
            self._add_token(token)
        
        # Добавление базовых символов
        for i in range(256):
            self._add_token(chr(i))
    
    def _add_token(self, token: str) -> None:
        """
        Добавление токена в словарь.
        """
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
    
    def train(self, texts: List[str]) -> None:
        """
        Обучение токенизатора на корпусе текстов.
        """
        # Обучение BPE
        if self.config.use_bpe:
            logger.info("Начало обучения BPE...")
            self.bpe.train(texts)
            logger.info("BPE обучен")
        
        # Обучение SentencePiece
        if self.config.use_sentencepiece and not self.sp:
            logger.info("Начало обучения SentencePiece...")
            # В реальной системе здесь будет обучение модели
            logger.info("SentencePiece обучен")
        
        # Сбор статистики
        for text in texts:
            tokens = self.encode(text)
            self.token_frequencies.update(tokens)
            self.sequence_lengths.append(len(tokens))
        
        logger.info(
            f"Токенизатор обучен на {len(texts)} текстах"
        )
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Кодирование текста в последовательность токенов.
        """
        # Проверка кэша
        cache_key = (text, add_special_tokens, max_length)
        return self.encode_cache(*cache_key)
    
    def _encode(
        self,
        text: str,
        add_special_tokens: bool,
        max_length: Optional[int]
    ) -> List[int]:
        """
        Внутренняя функция кодирования.
        """
        if not text:
            return []
        
        tokens = []
        
        # Добавление начального токена
        if add_special_tokens:
            tokens.append(self.vocab[self.special_tokens["bos"]])
        
        # Токенизация текста
        if self.config.use_sentencepiece and self.sp:
            # Использование SentencePiece
            sp_tokens = self.sp.encode_as_ids(text)
            tokens.extend(sp_tokens)
        elif self.config.use_bpe and self.bpe:
            # Использование BPE
            bpe_tokens = self.bpe.encode(text)
            tokens.extend(bpe_tokens)
        else:
            # Базовая токенизация
            for char in text:
                token_id = self.vocab.get(char, self.vocab[self.special_tokens["unk"]])
                tokens.append(token_id)
        
        # Добавление конечного токена
        if add_special_tokens:
            tokens.append(self.vocab[self.special_tokens["eos"]])
        
        # Обрезка по максимальной длине
        max_len = max_length or self.config.max_length
        if len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + [tokens[-1]]
        
        return tokens
    
    def decode(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Декодирование последовательности токенов в текст.
        """
        # Проверка кэша
        cache_key = (tuple(tokens), skip_special_tokens)
        return self.decode_cache(*cache_key)
    
    def _decode(
        self,
        tokens: Tuple[int, ...],
        skip_special_tokens: bool
    ) -> str:
        """
        Внутренняя функция декодирования.
        """
        if not tokens:
            return ""
        
        # Фильтрация специальных токенов
        if skip_special_tokens:
            tokens = [
                t for t in tokens
                if self.inverse_vocab[t] not in self.special_tokens.values()
            ]
        
        # Декодирование
        if self.config.use_sentencepiece and self.sp:
            # Использование SentencePiece
            text = self.sp.decode(tokens)
        elif self.config.use_bpe and self.bpe:
            # Использование BPE
            text = self.bpe.decode(tokens)
        else:
            # Базовое декодирование
            text = "".join(
                self.inverse_vocab.get(t, self.special_tokens["unk"])
                for t in tokens
            )
        
        return text
    
    def get_vocab_size(self) -> int:
        """
        Получение размера словаря.
        """
        return len(self.vocab)
    
    def get_token_frequency(self, token: Union[str, int]) -> int:
        """
        Получение частоты использования токена.
        """
        if isinstance(token, str):
            token = self.vocab.get(token, -1)
        return self.token_frequencies[token]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики использования токенизатора.
        """
        return {
            "vocab_size": len(self.vocab),
            "total_tokens": sum(self.token_frequencies.values()),
            "unique_tokens": len(self.token_frequencies),
            "avg_sequence_length": np.mean(self.sequence_lengths),
            "max_sequence_length": max(self.sequence_lengths, default=0),
            "min_sequence_length": min(self.sequence_lengths, default=0)
        }
    
    def save(self, path: str) -> None:
        """
        Сохранение токенизатора.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохранение конфигурации и словарей
        state = {
            "config": self.config.__dict__,
            "vocab": self.vocab,
            "inverse_vocab": self.inverse_vocab,
            "token_frequencies": self.token_frequencies,
            "sequence_lengths": self.sequence_lengths
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Токенизатор сохранен в {path}")
    
    def load(self, path: str) -> None:
        """
        Загрузка токенизатора.
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # Загрузка конфигурации
        self.config = TokenizerConfig(**state["config"])
        
        # Загрузка словарей и статистики
        self.vocab = state["vocab"]
        self.inverse_vocab = {int(k): v for k, v in state["inverse_vocab"].items()}
        self.token_frequencies = Counter(state["token_frequencies"])
        self.sequence_lengths = state["sequence_lengths"]
        
        # Пересоздание кэша
        self.encode_cache = lru_cache(maxsize=self.config.cache_size)(self._encode)
        self.decode_cache = lru_cache(maxsize=self.config.cache_size)(self._decode)
        
        logger.info(f"Токенизатор загружен из {path}")
    
    def clear_cache(self) -> None:
        """
        Очистка кэша токенизации.
        """
        self.encode_cache.cache_clear()
        self.decode_cache.cache_clear()
        logger.info("Кэш токенизатора очищен") 