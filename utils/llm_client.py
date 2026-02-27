import time
from openai import OpenAI

class LLMClient:
    def __init__(
        self,
        base_url,
        api_key,
        model,
        temperature=0.2,
        logger=None,
        agent_id=None,
        max_tokens=1500
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.logger = logger
        self.agent_id = agent_id
        self.max_tokens = max_tokens

    def chat(
        self,
        messages,
        temperature=None,
        stream=True,
        stream_log=False,
        log_lines=True,
        batch_size=10,
    ):
        temp = temperature if temperature is not None else self.temperature

        # Non-streaming mode
        if not stream:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                timeout=60,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()

        # Streaming
        full = []
        buffer = ""

        try:
            stream_resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                stream=True,
                timeout=60,
                max_tokens=self.max_tokens
            )

            # Logging
            start = time.time()
            
            for chunk in stream_resp:
                if time.time() - start > 60:
                    raise TimeoutError("Streaming timeout")
                
                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue

                token = delta.content
                full.append(token)
                
                # Logging is not activated
                if not stream_log or not self.logger:
                    continue

                # Per line
                if log_lines:
                    buffer += token
                    if "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        self.logger.info(f"[Agent {self.agent_id}] {line}")

                # Per batch
                else:
                    buffer += token
                    if len(buffer) >= batch_size:
                        self.logger.info(f"[Agent {self.agent_id}] {buffer}")
                        buffer = ""

            # If there's remaining buffer after streaming ends
            if stream_log and buffer and self.logger:
                self.logger.info(f"[Agent {self.agent_id}] {buffer}")

            return "".join(full).strip()

        except Exception as e:
            if self.logger:
                self.logger.error(f"[Agent {self.agent_id}] LLMClient.chat error: {e}")
            raise