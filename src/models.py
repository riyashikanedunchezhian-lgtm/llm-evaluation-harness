"""Model interface for LLM API calls."""

import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import anthropic
import openai
import cohere
from dotenv import load_dotenv
import logging
from functools import wraps

# Try to import Google AI (prefer new package)
try:
    import google.genai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai
        GOOGLE_GENAI_AVAILABLE = True
        import warnings
        warnings.warn("Using deprecated google.generativeai package. Install google-genai instead: pip install google-genai")
    except ImportError:
        GOOGLE_GENAI_AVAILABLE = False
        genai = None

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_error(max_retries=3, backoff_factor=2, exceptions=(Exception,)):
    """Decorator for retrying function calls on specific exceptions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
            raise last_exception
        return wrapper
    return decorator

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
        
        # Initialize Google AI
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key and GOOGLE_GENAI_AVAILABLE:
            try:
                genai.configure(api_key=google_api_key)
                logger.info("Google AI initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google AI: {str(e)}")
        
        # Initialize Cohere
        self.cohere_client = None
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key:
            self.cohere_client = cohere.Client(api_key=cohere_api_key)
        
        # Local model client (uses OpenAI-compatible API)
        self.local_client = None
        local_api_base = os.getenv("LOCAL_API_BASE", "http://localhost:8000/v1")
        if local_api_base:
            self.local_client = openai.OpenAI(
                base_url=local_api_base,
                api_key="dummy-key"  # Local models often don't require real API keys
            )
    
    def call_model(
        self,
        model_id: str,
        provider: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        api_base: Optional[str] = None
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
        elif provider == "google":
            response = self._call_google(
                model_id, prompt, max_tokens, temperature, system_prompt
            )
        elif provider == "cohere":
            response = self._call_cohere(
                model_id, prompt, max_tokens, temperature, system_prompt
            )
        elif provider == "local":
            response = self._call_local(
                model_id, prompt, max_tokens, temperature, system_prompt, api_base
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
    
    @retry_on_error(max_retries=3, backoff_factor=2, exceptions=(Exception,))
    def _call_anthropic(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            # Anthropic's newer API doesn't accept temperature in messages.create
            # It's set at the client level or not supported in this endpoint
            response = self.anthropic_client.messages.create(**kwargs)
            
            return {
                "content": response.content[0].text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
    @retry_on_error(max_retries=3, backoff_factor=2, exceptions=(Exception,))
    def _call_openai(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        try:
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
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    @retry_on_error(max_retries=3, backoff_factor=2, exceptions=(Exception,))
    def _call_google(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call Google AI API."""
        try:
            if not GOOGLE_GENAI_AVAILABLE:
                raise ValueError("Google AI library not available. Install with: pip install google-genai")
            
            model = genai.GenerativeModel(model_id)
            
            if system_prompt:
                # For Google, we prepend system prompt to the user message
                prompt = f"{system_prompt}\n\n{prompt}"
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            # Token counting for Google is approximate
            input_tokens = len(prompt.split())  # Rough estimate
            output_tokens = len(response.text.split())  # Rough estimate
            
            return {
                "content": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        except Exception as e:
            logger.error(f"Google AI API call failed: {str(e)}")
            raise
    
    @retry_on_error(max_retries=3, backoff_factor=2, exceptions=(Exception,))
    def _call_cohere(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Call Cohere API."""
        try:
            if not self.cohere_client:
                raise ValueError("Cohere API key not configured")
            
            message = prompt
            if system_prompt:
                message = f"{system_prompt}\n\n{prompt}"
            
            response = self.cohere_client.chat(
                message=message,
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "content": response.text,
                "input_tokens": response.meta.billed_units.input_tokens,
                "output_tokens": response.meta.billed_units.output_tokens
            }
        except Exception as e:
            logger.error(f"Cohere API call failed: {str(e)}")
            raise
    
    @retry_on_error(max_retries=3, backoff_factor=2, exceptions=(Exception,))
    def _call_local(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        api_base: Optional[str]
    ) -> Dict[str, Any]:
        """Call local model via OpenAI-compatible API."""
        try:
            if not self.local_client:
                # Initialize with custom API base if provided
                base_url = api_base or os.getenv("LOCAL_API_BASE", "http://localhost:8000/v1")
                self.local_client = openai.OpenAI(
                    base_url=base_url,
                    api_key="dummy-key"
                )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.local_client.chat.completions.create(
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
        except Exception as e:
            logger.error(f"Local model API call failed: {str(e)}")
            raise
