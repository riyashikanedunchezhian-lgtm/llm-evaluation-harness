"""Model interface for LLM API calls."""

import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelResponse:
    """Response from a model."""
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str

class ModelClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def call_model(
        self,
        model_id: str,
        provider: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """Call a model and return response with metrics."""
        start_time = time.time()
        
        if provider == "anthropic":
            response = self._call_anthropic(
                model_id, prompt, max_tokens, temperature, system_prompt
            )
        elif provider == "openai":
            response = self._call_openai(
                model_id, prompt, max_tokens, temperature, system_prompt
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ModelResponse(
            content=response["content"],
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            latency_ms=latency_ms,
            model_id=model_id
        )
    
    def _call_anthropic(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.anthropic_client.messages.create(**kwargs)
        
        return {
            "content": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    
    def _call_openai(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
