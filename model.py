import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os
from typing import List, Dict, Optional
import warnings

# Suppress beta warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning)

class MedicalChatbotRAG:
    def __init__(
        self,
        base_model: str = "google/flan-t5-large",
        embedding_model: str = "all-MiniLM-L6-v2",
        checkpoint_path: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        top_k_retrieval: int = 3
    ):
        """
        Initialize MedicalChatbotRAG with Retrieval-Augmented Generation capabilities.
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load embedding model for document retrieval
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize knowledge base and embeddings
        self.knowledge_base: List[str] = []
        self.knowledge_embeddings = None
        self.faiss_index = None

        # Retrieval parameters
        self.top_k_retrieval = top_k_retrieval

        # Checkpoint logic for model loading
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(base_model, checkpoint_path)
        else:
            self._initialize_model(base_model, lora_r, lora_alpha, lora_dropout)

    def _initialize_model(self, base_model: str, lora_r: int, lora_alpha: int, lora_dropout: float):
        """
        Initialize the model and apply LoRA configuration.
        """
        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # Load base model with quantization
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=quantization_config
        )

        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v", "k", "o"],
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

    def _load_from_checkpoint(self, base_model: str, checkpoint_path: str):
        """
        Load model from a checkpoint with LoRA.
        """
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            )
            self.model = prepare_model_for_kbit_training(self.model)
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            self.model = get_peft_model(self.model, peft_config)
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            print(f"Model loaded from checkpoint: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from checkpoint: {e}")

    def add_to_knowledge_base(self, documents: List[str]):
        """
        Add documents to the knowledge base and compute embeddings.
        """
        self.knowledge_base.extend(documents)
        self.knowledge_embeddings = self.embedding_model.encode(self.knowledge_base)
        dimension = self.knowledge_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.knowledge_embeddings)

    def retrieve_relevant_context(self, query: str) -> List[str]:
        """
        Retrieve relevant documents from the knowledge base for a query.
        """
        if not self.faiss_index:
            return []

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding, self.top_k_retrieval)
        return [self.knowledge_base[idx] for idx in indices[0]]

    def generate_response(self, prompt: str) -> str:
        """
        Generate a detailed response using the RAG model.
        This method combines retrieved context from the knowledge base with an augmented prompt
        to guide the model toward generating comprehensive, detailed, and accurate responses.

        Args:
            prompt (str): The user's query or question.

        Returns:
            str: The generated response from the model.
        """
        # Step 1: Retrieve relevant context from the knowledge base
        retrieved_contexts = self.retrieve_relevant_context(prompt)

        # Step 2: Construct an augmented prompt with detailed instructions
        augmented_prompt = (
            f"You are a highly knowledgeable medical assistant. "
            f"Your goal is to provide accurate, detailed, and structured responses "
            f"to medical queries. Use the retrieved context below to formulate your answer.\n\n"
            f"Context: {' '.join(retrieved_contexts)}\n\n"
            f"Query: {prompt}\n\n"
            f"Response: "
        )

        # Step 3: Tokenize the augmented prompt
        inputs = self.tokenizer(
            augmented_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        for key in inputs:
          inputs[key] = inputs[key].to(self.model.device)

        # Step 4: Generate the response using the model
        try:
            outputs = self.model.generate(
                **inputs,
                max_length=256,                # Limit the response length
                num_return_sequences=1,        # Generate a single response
                temperature=0.7,               # Lower temperature for focused and deterministic answers
                top_p=0.9,                     # Use nucleus sampling for diversity
                repetition_penalty=1.2,        # Penalize repetitive outputs
                early_stopping=True            # Stop generation once criteria are met
            )
            # Decode the generated response to a human-readable string
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            # Handle exceptions gracefully
            response = f"An error occurred while generating the response: {str(e)}"

        return response

    def prepare_dataset(self, df: pd.DataFrame):
        """
        Prepare dataset for training.
        """
        dataset = Dataset.from_pandas(df[['Prompt', 'Completion']])
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples['Prompt'], max_length=512, truncation=True, padding='max_length'
            )
            targets = self.tokenizer(
                examples['Completion'], max_length=512, truncation=True, padding='max_length'
            )
            inputs['labels'] = targets['input_ids']
            return inputs
        return dataset.map(tokenize_function, remove_columns=['Prompt', 'Completion'])


    def train(self, df: pd.DataFrame, output_dir: str = "./output", epochs: int = 5, batch_size: int = 4):
        """
        Train the model with LoRA.
        """
        train_dataset = self.prepare_dataset(df)
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            predict_with_generate=True,
            fp16=True,
            save_strategy="epoch",
            evaluation_strategy="no",
            logging_dir="./logs",
            logging_steps=10
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        trainer.train()
        trainer.save_model(output_dir)

def main():
    # Initialize chatbot with the checkpoint
    chatbot = MedicalChatbotRAG(
        base_model="google/flan-t5-large",
        embedding_model="all-MiniLM-L6-v2",
        checkpoint_path="./output"
    )

    # Add some documents to knowledge base (optional)
    # Create a list of medical knowledge bases
    medical_knowledge_base = [
        # Cardiovascular Diseases
        "Hypertension (high blood pressure) is a common condition where the force of the blood against the walls of your arteries is too high. It can lead to serious complications like heart disease and stroke.",
        "Coronary artery disease occurs when blood vessels become narrowed or blocked by plaque, leading to reduced blood flow to the heart muscle. It can result in chest pain (angina) or a heart attack.",

        # Diabetes and Metabolic Disorders
        "Type 2 diabetes is a condition where the body becomes resistant to insulin or does not produce enough insulin. It often results in increased thirst, frequent urination, and fatigue.",
        "Insulin resistance can lead to metabolic syndrome, characterized by obesity, high blood pressure, and high blood sugar, which increase the risk of cardiovascular diseases.",

        # Respiratory Diseases
        "Asthma is a chronic condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, and coughing.",
        "COPD is a group of lung diseases that cause chronic airflow limitation. It is mainly caused by long-term smoking and includes conditions like emphysema and chronic bronchitis.",

        # Neurological Conditions
        "Alzheimer's disease is a neurodegenerative condition that causes memory loss, confusion, and behavioral changes, commonly affecting older adults.",
        "Parkinson's disease is a neurodegenerative disorder that leads to tremors, stiffness, and difficulty with movement. It is caused by the loss of dopamine-producing neurons.",

        # Cancer Types and Treatments
        "Breast cancer is a common type of cancer that can present as a lump in the breast. Treatment may include surgery, chemotherapy, and radiation therapy.",
        "Lung cancer is often diagnosed through imaging and biopsy. Treatment options include surgery, chemotherapy, immunotherapy, and targeted therapies.",

        # Mental Health Disorders
        "Depression is a mood disorder marked by persistent feelings of sadness, loss of interest, and fatigue. It is typically treated with therapy and antidepressant medications.",
        "Anxiety disorders are characterized by excessive worry and fear. Treatment includes therapy (e.g., CBT) and medications (e.g., SSRIs, benzodiazepines).",

        # Infectious Diseases
        "Tuberculosis (TB) primarily affects the lungs and is characterized by a chronic cough, fever, and weight loss. It is treated with a combination of antibiotics.",
        "COVID-19 is caused by the SARS-CoV-2 virus and can present with fever, cough, difficulty breathing, and fatigue. Vaccines and antiviral treatments are available.",

        # Orthopedic Conditions
        "Osteoarthritis is a degenerative joint disease that causes pain, stiffness, and swelling, particularly in weight-bearing joints such as the knees and hips.",
        "Fractures are breaks in the bone caused by trauma. Treatment may include immobilization with a cast or surgery, depending on the type and location of the fracture.",

        # Pediatric Conditions
        "Pediatric asthma is common in children and is characterized by wheezing, coughing, and shortness of breath. It is often triggered by allergens or respiratory infections.",
        "Viral infections like RSV (Respiratory Syncytial Virus) can cause symptoms such as wheezing and difficulty breathing, especially in infants.",

        # Gastrointestinal Diseases
        "Irritable Bowel Syndrome (IBS) is a functional gastrointestinal disorder causing symptoms like abdominal pain, bloating, and changes in bowel habits.",
        "Gastroesophageal reflux disease (GERD) occurs when stomach acid frequently leaks into the esophagus, leading to symptoms like heartburn and regurgitation.",

        # Optional dataset related knowledge bases
        "Osteoarthritis is a degenerative joint disease that commonly affects older adults, leading to pain, stiffness, and reduced function.",
        "Management of osteoarthritis typically includes lifestyle modifications, physical therapy, pain relief with NSAIDs, and weight management.",
        "In cases of worsened symptoms, corticosteroid injections, hyaluronic acid injections, or surgical interventions may be considered.",

        "A fall onto an outstretched hand may result in a wrist sprain, fracture, or other injuries, presenting with pain, swelling, and difficulty moving the wrist.",
        "Common injuries include scaphoid fractures, distal radius fractures, or wrist sprains. These injuries can be diagnosed with X-rays or other imaging techniques.",
        "Treatment may involve immobilization with a cast or splint, pain management, and potentially surgery, depending on the severity of the fracture.",

        "Attention-Deficit/Hyperactivity Disorder (ADHD) is characterized by symptoms like restlessness, impulsivity, and difficulty concentrating, often beginning in childhood.",
        "ADHD can significantly affect personal and professional life and is diagnosed through clinical evaluation, including symptom assessment and history.",
        "Treatment options for ADHD include stimulant medications such as methylphenidate or non-stimulants like atomoxetine, along with behavioral therapy."
    ]

    # Add knowledge to the RAG model's knowledge base
    chatbot.add_to_knowledge_base(medical_knowledge_base)

    # Interact with the chatbot
    prompt = input("Enter your query: ")
    response = chatbot.generate_response(f"Please provide a detailed medical explanation for the following case: {prompt}")
    print(f"Chatbot: {response}")

# Call the main function to run the chatbot
if __name__ == "__main__":
    main()
