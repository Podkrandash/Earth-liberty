"""
Тренер для языковой модели Earth-Liberty.

Реализует:
- Обучение модели на данных
- Оптимизацию и планирование обучения
- Валидацию и оценку
- Сохранение и загрузку чекпоинтов
- Логирование метрик
"""

import os
import json
import logging
import math
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
import wandb

from .config import ModelConfig
from .tokenizer import Tokenizer
from .generator import TextGenerator

logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments:
    """Аргументы для обучения модели."""
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 10000
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: Optional[int] = 3
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    seed: int = 42
    fp16: bool = True
    fp16_opt_level: str = "O1"
    local_rank: int = -1
    
    def __post_init__(self):
        """Проверка и корректировка аргументов."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.use_wandb and not self.wandb_project:
            self.wandb_project = "earth-liberty"
        
        if self.use_wandb and not self.wandb_name:
            self.wandb_name = "training-run"

class Trainer:
    """
    Тренер для языковой модели Earth-Liberty.
    """
    
    def __init__(
        self,
        model: TextGenerator,
        tokenizer: Tokenizer,
        config: ModelConfig,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None
    ):
        """
        Инициализация тренера.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Устройство и тип данных
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Перемещение модели на устройство
        self.model.to(self.device)
        
        # Инициализация оптимизатора
        self.optimizer = self._create_optimizer()
        
        # Инициализация планировщика
        self.scheduler = self._create_scheduler()
        
        # Инициализация scaler для mixed precision
        self.scaler = GradScaler() if args.fp16 else None
        
        # Метрики
        self.train_metrics = {}
        self.eval_metrics = {}
        
        # Логирование
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "model_config": config.to_dict(),
                    "training_args": args.__dict__
                }
            )
    
    def train(self) -> Dict[str, float]:
        """
        Обучение модели.
        """
        logger.info("Начало обучения")
        
        # Подготовка загрузчика данных
        train_dataloader = self._get_train_dataloader()
        
        # Установка модели в режим обучения
        self.model.train()
        
        # Инициализация прогресс-бара
        total_steps = len(train_dataloader) * self.args.num_train_epochs
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        # Обучение
        global_step = 0
        tr_loss = 0.0
        
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                # Перемещение батча на устройство
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Прямой проход
                loss = self._training_step(batch)
                
                # Накопление градиентов
                loss = loss / self.args.gradient_accumulation_steps
                
                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                
                # Обновление весов
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Клиппинг градиентов
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    
                    # Шаг оптимизатора
                    if self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Шаг планировщика
                    self.scheduler.step()
                    
                    # Обнуление градиентов
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Логирование
                    if global_step % self.args.logging_steps == 0:
                        self._log_metrics({
                            "loss": tr_loss / global_step,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "epoch": epoch + step / len(train_dataloader)
                        }, prefix="train")
                    
                    # Валидация
                    if global_step % self.args.eval_steps == 0:
                        self.evaluate()
                    
                    # Сохранение
                    if global_step % self.args.save_steps == 0:
                        self.save_checkpoint(global_step)
                
                progress_bar.update(1)
        
        # Финальная валидация
        self.evaluate()
        
        # Финальное сохранение
        self.save_checkpoint(global_step)
        
        # Закрытие прогресс-бара
        progress_bar.close()
        
        return self.train_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Валидация модели.
        """
        logger.info("Начало валидации")
        
        # Проверка наличия валидационного датасета
        if not self.eval_dataset:
            logger.warning("Валидационный датасет не предоставлен")
            return {}
        
        # Подготовка загрузчика данных
        eval_dataloader = self._get_eval_dataloader()
        
        # Установка модели в режим оценки
        self.model.eval()
        
        # Инициализация метрик
        eval_loss = 0.0
        eval_steps = 0
        perplexity = 0.0
        
        # Валидация
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Перемещение батча на устройство
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Прямой проход
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs["loss"]
                eval_loss += loss.item()
                eval_steps += 1
                
                # Вычисление perplexity
                perplexity += torch.exp(loss).item()
        
        # Усреднение метрик
        eval_loss = eval_loss / eval_steps
        perplexity = perplexity / eval_steps
        
        # Обновление метрик
        self.eval_metrics.update({
            "eval_loss": eval_loss,
            "eval_perplexity": perplexity
        })
        
        # Логирование
        self._log_metrics(self.eval_metrics, prefix="eval")
        
        # Возврат модели в режим обучения
        self.model.train()
        
        return self.eval_metrics
    
    def save_checkpoint(self, step: int) -> None:
        """
        Сохранение чекпоинта модели.
        """
        # Создание директории для сохранения
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохранение состояния модели
        model_state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "step": step,
            "train_metrics": self.train_metrics,
            "eval_metrics": self.eval_metrics
        }
        
        torch.save(model_state, os.path.join(output_dir, "model_state.pt"))
        
        # Сохранение конфигурации
        self.config.save(os.path.join(output_dir, "config.json"))
        
        # Сохранение токенизатора
        self.tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        
        # Очистка старых чекпоинтов
        if self.args.save_total_limit:
            self._cleanup_checkpoints()
        
        logger.info(f"Сохранен чекпоинт: {output_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Загрузка чекпоинта модели.
        """
        # Загрузка состояния модели
        model_state = torch.load(
            os.path.join(checkpoint_dir, "model_state.pt"),
            map_location=self.device
        )
        
        self.model.load_state_dict(model_state["model_state_dict"])
        self.optimizer.load_state_dict(model_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(model_state["scheduler_state_dict"])
        
        if self.scaler and model_state["scaler_state_dict"]:
            self.scaler.load_state_dict(model_state["scaler_state_dict"])
        
        self.train_metrics = model_state["train_metrics"]
        self.eval_metrics = model_state["eval_metrics"]
        
        logger.info(f"Загружен чекпоинт: {checkpoint_dir}")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Один шаг обучения.
        """
        with autocast(enabled=self.args.fp16):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs["loss"]
        
        return loss
    
    def _get_train_dataloader(self) -> DataLoader:
        """
        Создание загрузчика данных для обучения.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def _get_eval_dataloader(self) -> DataLoader:
        """
        Создание загрузчика данных для валидации.
        """
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Создание оптимизатора.
        """
        # Подготовка параметров с decay и без
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self) -> LambdaLR:
        """
        Создание планировщика скорости обучения.
        """
        num_training_steps = len(self._get_train_dataloader()) * self.args.num_train_epochs
        
        def lr_lambda(current_step: int):
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / \
                float(max(1, num_training_steps - self.args.warmup_steps))
            )
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Логирование метрик.
        """
        # Добавление префикса к метрикам
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Логирование в wandb
        if self.args.use_wandb:
            wandb.log(metrics)
        
        # Логирование в консоль
        logger.info(
            "Metrics: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )
    
    def _cleanup_checkpoints(self) -> None:
        """
        Очистка старых чекпоинтов.
        """
        checkpoints = [
            d for d in os.listdir(self.args.output_dir)
            if d.startswith("checkpoint-")
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        
        # Удаление старых чекпоинтов
        if len(checkpoints) > self.args.save_total_limit:
            for checkpoint in checkpoints[:-self.args.save_total_limit]:
                checkpoint_dir = os.path.join(self.args.output_dir, checkpoint)
                logger.info(f"Удаление старого чекпоинта: {checkpoint_dir}")
                for file in os.listdir(checkpoint_dir):
                    os.remove(os.path.join(checkpoint_dir, file))
                os.rmdir(checkpoint_dir) 