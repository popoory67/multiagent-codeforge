from openai import OpenAI

class LLMClient:
    def __init__(self, base_url, api_key, model, temperature=0.2):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages, temperature=None):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
        )
        return resp.choices[0].message.content.strip()