from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from sklearn.metrics.pairwise import cosine_similarity
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
import pandas as pd
import datasets
import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import os
from typing import List, Dict, Optional
import time
from tqdm import tqdm


class MistralQA:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Инициализация модели Mistral-7B с RAG поддержкой
        :param model_name: Название модели Hugging Face
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Конфигурация для оптимизации памяти
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ) if self.device == "cuda" else None

        # Загрузка модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).eval()

        # Системный промпт
        self.system_prompt = """Ты - помощник, отвечающий на вопросы. Используй предоставленный контекст для точных ответов.
        Если ответа нет в контексте, скажи об этом. Будь кратким и информативным."""

#        convert(framework="pt", model=self.model, output="mistral-7b.onnx", opset=12)
        # Инициализация RAG
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = defaultdict(list)
        self.documents = []
        self.document_embeddings = None

    def load_qa_pairs_from_txt(self, file_path: str):
        """
        Загрузка пар вопрос-ответ из файла
        Формат:
        === Question
        текст вопроса
        === Answer
        текст ответа
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        qa_pairs = re.findall(r'=== Question\n(.*?)\n=== Answer\n(.*?)(?=\n=== Question|\Z)', content, re.DOTALL)

        for question, answer in qa_pairs:
            self.knowledge_base[question.strip()].append(answer.strip())
            self._add_to_rag_context(f"Q: {question.strip()}\nA: {answer.strip()}")

    def load_qa_from_df(self, df: pd.Dataset):
        """Загрузка пар вопрос-ответ из pandas Dataset"""
        for index, row in tqdm(df.iterrows(), desc="Loading Dataset", total=len(df)):
            try:
                self.knowledge_base[row['name'].strip() + '. ' + row['message'].strip()].append(row['answer'].strip())
                self._add_to_rag_context(
                    f"Q: {row['name'].strip() + row['message'].strip()}\nA: {row['answer'].strip()}")
            except:
                pass

    def load_qa_from_csv(self, file_path: str):
        """Загрузка пар вопрос-ответ из csv файла"""
        df = pd.read_csv(file_path)
        self.load_qa_from_df(df)

    def load_qa_from_parquet(self, file_path: str):
        df = pd.read_parquet(file_path)
        for index, row in tqdm(df.iterrows(), desc="Loading parquet", total=len(df)):
            try:
                self.knowledge_base[row['context'].strip() + '. ' + row['question'].strip()].append(row['answers'].strip())
                self._add_to_rag_context(
                    f"Q: {row['context'].strip() + row['question'].strip()}\nA: {row['answers'].strip()}")
            except:
                pass

    def add_document(self, text: str):
        """Добавление документа в RAG контекст"""
        self._add_to_rag_context(text)

    def add_website(self, url: str):
        """Добавление контента с веб-сайта"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            self._add_to_rag_context(f"Контент с сайта {url}:\n{text}")
        except Exception as e:
            print(f"Ошибка при загрузке сайта: {e}")

    def _add_to_rag_context(self, text: str):
        """Внутренний метод для добавления текста в RAG"""
        self.documents.append(text)
        new_embedding = self.embedding_model.encode([text])
        self.document_embeddings = (
            new_embedding if self.document_embeddings is None
            else np.vstack([self.document_embeddings, new_embedding])
        )

    def _find_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Поиск релевантного контекста для RAG"""
        if not self.documents:
            return ""

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return "\n\n".join([self.documents[i] for i in top_indices])

    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Генерация ответа с учетом контекста
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {prompt}"} if context
            else {"role": "user", "content": prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Параметры генерации для CPU
        generate_kwargs = {
            "input_ids": inputs,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,

            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": 2000,
            "early_stopping": True,
            "num_beams": 2,
        }

        if self.device == "cpu":
            generate_kwargs.update({
                "max_new_tokens": 256,  # Меньше токенов для CPU
#                "max_new_tokens": 128,  # Меньше токенов для CPU
            })

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        return self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    def answer(self, question: str) -> str:
        """Основной метод для получения ответа"""
        # Сначала проверяем базу знаний
        if question in self.knowledge_base:
            return self.knowledge_base[question][0]

        # Ищем похожие вопросы
        for q in self.knowledge_base:
            if question.lower() in q.lower() or q.lower() in question.lower():
                return self.knowledge_base[q][0]

        # Получаем релевантный контекст для RAG
        context = self._find_relevant_context(question)

        # Генерируем ответ
        return self.generate_response(question, context)

    def save(self, dir_path='./my_mistral_model'):
        """Сохранение модели и знаний"""
        os.makedirs(dir_path, exist_ok=True)

        # Сохраняем модель
        self.model.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)

        # Сохраняем базу знаний
        with open(os.path.join(dir_path, 'knowledge_base.txt'), 'w', encoding='utf-8') as f:
            for q, answers in self.knowledge_base.items():
                for a in answers:
                    f.write(f"=== Question\n{q}\n=== Answer\n{a}\n\n")

        # Сохраняем документы RAG
        with open(os.path.join(dir_path, 'rag_documents.txt'), 'w', encoding='utf-8') as f:
            f.write("\n===DOCUMENT===\n".join(self.documents))

    def load(self, dir_path='./my_mistral_model'):
        """Загрузка сохраненной модели"""
        # Загружаем модель
        self.model = AutoModelForCausalLM.from_pretrained(
            dir_path,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(dir_path)

        # Загружаем базу знаний
        kb_file = os.path.join(dir_path, 'knowledge_base.txt')
        if os.path.exists(kb_file):
            self.load_qa_pairs(kb_file)

        # Загружаем документы RAG
        rag_file = os.path.join(dir_path, 'rag_documents.txt')
        if os.path.exists(rag_file):
            with open(rag_file, 'r', encoding='utf-8') as f:
                docs = f.read().split("\n===DOCUMENT===\n")
                for doc in docs:
                    if doc.strip():
                        self._add_to_rag_context(doc.strip())

    def LoRA_train(
        self,
        train_data: pd.DataFrame,  # DataFrame с колонками "instruction", "input", "output"
        output_dir: str = "./lora_results",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 2e-5,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_seq_length: int = 1024,
        save_steps: int = 500,
        logging_steps: int = 100,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        fp16: bool = True,
    ):
        """
        Дообучение модели с использованием LoRA (Low-Rank Adaptation)

        :param train_data: DataFrame с обучающими данными (колонки: "instruction", "input", "output")
        :param output_dir: Директория для сохранения результатов
        :param num_train_epochs: Количество эпох обучения
        :param per_device_train_batch_size: Размер батча на устройство
        :param learning_rate: Скорость обучения
        :param lora_rank: Ранг матриц LoRA
        :param lora_alpha: Альфа параметр для LoRA
        :param lora_dropout: Dropout для LoRA
        :param max_seq_length: Максимальная длина последовательности
        :param save_steps: Сохранять модель каждые N шагов
        :param logging_steps: Логировать каждые N шагов
        :param warmup_steps: Количество шагов разогрева
        :param gradient_accumulation_steps: Шагов накопления градиента
        :param fp16: Использовать mixed precision (FP16)
        """
        # Подготовка модели для LoRA обучения
        self.model = prepare_model_for_kbit_training(self.model)

        # Конфигурация LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Преобразование DataFrame в формат datasets
        def format_instruction(row):
            if pd.isna(row['input']) or str(row['input']).strip() == "":
                return f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['output']}"
            else:
                return f"### Instruction:\n{row['instruction']}\n\n### Input:\n{row['input']}\n\n### Response:\n{row['output']}"

        formatted_data = train_data.apply(format_instruction, axis=1).tolist()
        dataset = datasets.Dataset.from_dict({"text": formatted_data})

        # Токенизация данных
        def tokenize_function(examples):
            # Сначала создаем полные промпты
            prompts = []
            for text in examples["text"]:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text.split("### Response:")[0].strip()},
                    {"role": "assistant", "content": text.split("### Response:")[1].strip()}
                ]
                prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

            # Затем токенизируем
            return self.tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        # Аргументы обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            save_steps=save_steps,
            logging_steps=logging_steps,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            report_to="none",
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            evaluation_strategy="no",
            save_strategy="steps",
        )

        # Тренировка
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=lambda data: {
                "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
                "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
                "labels": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            },
        )

        # Запуск обучения
        trainer.train()

        # Сохранение LoRA адаптеров
        self.model.save_pretrained(output_dir)

        # Объединение модели с адаптерами для последующего использования
        self.model = self.model.merge_and_unload()

        print(f"LoRA обучение завершено. Результаты сохранены в {output_dir}")
