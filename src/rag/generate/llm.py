import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rag.config import LLM_MODEL, MAX_TOKENS


class LLM:
    def __init__(self):
        self.quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=self.quant,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    def generate(
        self, question_prompt: str, system_prompt: str = "You are a helpful assistant."
    ):
        message = self.create_message(
            query=question_prompt, system_prompt=system_prompt
        )
        text = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=MAX_TOKENS)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response

    def create_message(self, query: str, system_prompt: str):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
