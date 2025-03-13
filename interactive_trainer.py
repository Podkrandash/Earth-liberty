"""
Интерактивное обучение модели через диалоги.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import torch
from model.language.generator import TextGenerator
from model.language.config import ModelConfig
from model.data.processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveTrainer:
    def __init__(
        self,
        model: Optional[TextGenerator] = None,
        config_path: str = "config/model_config.json",
        dialogues_dir: str = "data/dialogues",
        save_dir: str = "output/models"
    ):
        self.dialogues_dir = dialogues_dir
        self.save_dir = save_dir
        os.makedirs(dialogues_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Загрузка или создание модели
        if model is None:
            config = ModelConfig.load(config_path)
            self.model = TextGenerator(config)
        else:
            self.model = model

        self.current_dialogue = {
            "system": "Вы - Earth-Liberty AI, дружелюбный и умный ассистент. Вы всегда стараетесь помочь пользователю и отвечаете честно.",
            "turns": []
        }

    def chat(self):
        """Интерактивный режим чата с сохранением диалога."""
        print("Начинаем диалог (для выхода введите 'exit' или 'quit')")
        print("Для сохранения и обучения введите 'save'")
        
        while True:
            # Ввод пользователя
            user_input = input("\nВы: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                if len(self.current_dialogue["turns"]) > 0:
                    self.save_dialogue()
                break
            
            if user_input.lower() == 'save':
                self.save_dialogue()
                self.train_on_saved_dialogues()
                continue

            # Генерация ответа
            response = self.generate_response(user_input)
            print(f"\nEarth-Liberty AI: {response}")

            # Сохранение обмена репликами
            self.current_dialogue["turns"].append({
                "user": user_input,
                "assistant": response
            })

    def generate_response(self, user_input: str) -> str:
        """Генерация ответа модели."""
        try:
            # Подготовка контекста из текущего диалога
            context = self.current_dialogue["system"] + "\n"
            for turn in self.current_dialogue["turns"]:
                context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            context += f"User: {user_input}\nAssistant:"

            # Генерация ответа
            with torch.no_grad():
                response = self.model.generate(
                    input_ids=torch.tensor([self.model.tokenizer.encode(context)]),
                    max_length=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )[0]
                
            decoded_response = self.model.tokenizer.decode(response)
            # Извлекаем только ответ ассистента
            response_text = decoded_response.split("Assistant:")[-1].strip()
            return response_text

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return "Извините, произошла ошибка при генерации ответа. Я все еще учусь."

    def save_dialogue(self):
        """Сохранение текущего диалога."""
        if len(self.current_dialogue["turns"]) == 0:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.dialogues_dir, f"dialogue_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_dialogue, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Диалог сохранен в {filename}")
        
        # Очистка текущего диалога
        self.current_dialogue["turns"] = []

    def train_on_saved_dialogues(self):
        """Обучение на сохраненных диалогах."""
        try:
            # Подготовка данных
            processor = DataProcessor(
                data_dir=self.dialogues_dir,
                output_dir="data/processed"
            )
            
            train_files, val_files = processor.process_dialogue_corpus(
                os.path.join(self.dialogues_dir, "*.json")
            )

            if not train_files:
                logger.info("Нет новых диалогов для обучения")
                return

            # Загрузка диалогов
            dialogues = []
            for file in train_files:
                with open(file, 'r', encoding='utf-8') as f:
                    dialogues.extend(json.load(f))

            # Подготовка данных для обучения
            training_data = []
            for dialogue in dialogues:
                context = dialogue["system"] + "\n"
                for turn in dialogue["turns"]:
                    # Добавляем контекст с вопросом
                    context += f"User: {turn['user']}\n"
                    training_data.append(context + "Assistant: " + turn['assistant'])
                    # Обновляем контекст ответом
                    context += f"Assistant: {turn['assistant']}\n"

            # Обучение модели
            logger.info("Начало дообучения модели...")
            self.model.train()
            
            for epoch in range(3):  # 3 эпохи обучения
                total_loss = 0
                for text in training_data:
                    # Токенизация текста
                    inputs = self.model.tokenizer.encode(text)
                    input_ids = torch.tensor([inputs]).to(self.model.device)
                    
                    # Обучение на одном примере
                    metrics = self.model.train_step(input_ids)
                    total_loss += metrics["loss"]
                
                avg_loss = total_loss / len(training_data)
                logger.info(f"Эпоха {epoch + 1}, средняя потеря: {avg_loss:.4f}")
            
            # Сохранение модели
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"model_{timestamp}.pt")
            torch.save(self.model.state_dict(), save_path)
            
            logger.info(f"Модель сохранена в {save_path}")

        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            logger.exception(e)

def main():
    trainer = InteractiveTrainer()
    trainer.chat()

if __name__ == "__main__":
    main() 