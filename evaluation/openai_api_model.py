import time
import random

import torch
from transformers import PreTrainedTokenizer, BatchEncoding
import openai


class GPT3API:

    device: str = "cpu"

    def __init__(self, api_key: str, openai_model_id: str, tokenizer: PreTrainedTokenizer, test: bool = True):
        self.tokenizer = tokenizer
        self.name_or_path = openai_model_id
        self.input_tokens_counter = 0
        self.test = test
        openai.api_key = api_key

    def generate(self, input_ids: BatchEncoding, **other_inputs) -> torch.LongTensor:
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        assert len(input_texts) == 1, "OpenAI model supports only batch size==1 inputs"
        self.input_tokens_counter += len(input_ids[0])
        input_text = input_texts[0]
        if not self.test:
            response = openai.Completion.create(model=self.name_or_path,
                                                prompt=input_text,
                                                temperature=0)
            response_text = response["choices"][0]["text"]
            if self.name_or_path == "davinci" and "Input" in response_text:
                # GPT3 struggles with terminating generation in few-shot settings, but we do not want to penalize that
                response_text = response_text.split("Input")[0]
            time.sleep(random.randint(300, 600) / 100)
        else:
            response_text = "mock text"
        response_encoding = self.tokenizer(response_text, return_tensors="pt").input_ids
        print("Input tokens: %s, price: %s USD" % (self.input_tokens_counter,
                                                   (self.input_tokens_counter / 1000) * 0.02))
        return response_encoding
