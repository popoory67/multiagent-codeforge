# agents/base_agent.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any, List

class BaseAgent:
    """
    - llm: LLMClient (streaming/logging)
    - prompts: dict
    - project_ctx/config/logger/agent_id: optional but useful for logging and advanced prompting
    """
    def __init__(
        self,
        llm,
        prompts: Dict[str, Any],
        project_ctx: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ):
        self.llm = llm
        self.prompts = prompts
        self.project_ctx = project_ctx
        self.config = config or {}
        self.agent_id = agent_id

    def chat(self, 
             system: str, 
             user: str, 
             temperature: Optional[float] = None, 
             stream: bool = True, 
             stream_log: bool = False, 
             log_lines: bool = True,
             batch_size: int = 10) -> str:
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.llm.chat(messages, 
                             temperature=temperature, 
                             stream=stream,
                             stream_log=stream_log,
                             log_lines=log_lines,
                             batch_size=batch_size
                        ).strip()