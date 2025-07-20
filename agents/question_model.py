# Starting with Qwen3-4B
import time
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
import torch
from peft import LoraConfig, PeftModel

class QAgent(object):
    def __init__(self, adapter_type='sft', **kwargs):
        # self.model_type = input("Available models: Qwen3-1.7B and Qwen3-4B. Please enter 1.7B or 4B: ").strip()
        self.model_type = kwargs.get('model_type', '4B').strip()
        # model_name = "Qwen/Qwen3-4B"
        model_name = "/jupyter-tutorial/hf_models/Qwen3-4B"
        
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.adapter_type = adapter_type
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._setup_model_with_adapter(self.model)        
        self.model = self.model.eval()
        
    def _setup_model_with_adapter(self, base_model):
        """Setup model with or without adapter based on configuration."""
        if self.adapter_type is None:
            print("No adapter specified - using base model")
            return base_model

        self.adapter_type = self.adapter_type.lower()

        if self.adapter_type == 'sft' or self.adapter_type == 'grpo':
            print("Loading required adapter")
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                print(f"No trained checkpoints found for {self.adapter_type}")
                inference_model = base_model
            else:
                print(f"Loading LoRA adapters from: {checkpoint_path}")
                inference_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            return inference_model
        else:
            print(f"Unknown adapter type: {self.adapter_type}. Using base model.")
            return base_model

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the output directory."""

        output_dir = "./ckpt/"
        
        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                match = checkpoint_pattern.match(item)
                if match:
                    step_num = int(match.group(1))
                    checkpoints.append((step_num, item_path))
        
        if not checkpoints:
            print(f"No checkpoints found in {output_dir}")
            return None
        
        checkpoints.sort(key=lambda x: int(x[0]))
        latest_checkpoint = checkpoints[-1][1]
        
        print(f"Found {len(checkpoints)} checkpoints. Using latest: {latest_checkpoint}")
        return latest_checkpoint
        
    def generate_response(self, message: str|List[str], system_prompt: Optional[str] = None, **kwargs)->str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
        # Prepare all messages for batch processing
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg}
            ]
            all_messages.append(messages)
        
        # convert all messages to text format
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            texts.append(text)

        # tokenize all texts together with padding
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        tgps_show_var = kwargs.get('tgps_show', False)
        # conduct batch text completion
        if tgps_show_var: start_time = time.time()   
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 1024),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if tgps_show_var: generation_time = time.time() - start_time

        # decode the batch
        batch_outs = []
        if tgps_show_var: token_len = 0
        for i, (input_ids, generated_sequence) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            # extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids):].tolist()
            
            # compute total tokens generated
            if tgps_show_var: token_len += len(output_ids)

            # remove thinking content using regex
            # result = re.sub(r'<think>[\s\S]*?</think>', '', full_result, flags=re.DOTALL).strip()
            index = len(output_ids) - output_ids[::-1].index(151668) if 151668 in output_ids else 0
            
            # decode the full result
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            batch_outs.append(content)
        if tgps_show_var:
            return batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None

if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    import random
    topics=['Blood Relations and Family Trees','Truth-teller and Liar Problems','Seating Arrangements (Linear and Circular)']
    t=random.choice(topics)
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, {t}.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": [A/B/C/D],
        }}
    """
            # "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"

    
    response, tl, tm = model.generate_response(prompt, tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    print("Single example response:")
    print("Response: ", response)
    print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
    print("+-------------------------------------------------\n\n")

    # # Multi example generation
    # prompts = [
    #     "What is the capital of France?",
    #     "Explain the theory of relativity.",
    #     "What are the main differences between Python and Java?",
    #     "What is the significance of the Turing Test in AI?",
    #     "What is the capital of Japan?",
    # ]
    # responses, tl, tm = model.generate_response(prompts, tgps_show=True, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
    # print("\nMulti example responses:")
    # for i, resp in enumerate(responses):
    #     print(f"Response {i+1}: {resp}")
    # print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
