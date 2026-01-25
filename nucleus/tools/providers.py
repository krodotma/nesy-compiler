
import abc
import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# Constants
AUTH_FILE = Path.home() / ".pluribus" / "auth.json"

def load_auth_token(provider: str) -> Optional[str]:
    """Load token from ~/.pluribus/auth.json for the given provider."""
    try:
        if not AUTH_FILE.exists():
            return None
        with open(AUTH_FILE, 'r') as f:
            data = json.load(f)
        
        provider_data = data.get("providers", {}).get(provider, {})
        return provider_data.get("access_token")
    except Exception:
        return None

# Abstract Base Class
class BaseLLMProvider(abc.ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any] = {}) -> str:
        """Generate text from the LLM provider."""
        pass

    @property
    @abc.abstractmethod
    def provider_id(self) -> str:
        """Return the unique provider ID."""
        pass

# Claude Provider (Anthropic) - Uses LASER Token
class ClaudeProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        super().__init__(model_name)
        self.token = load_auth_token("claude")
        # Fallback to env var if explicit bypass is set (legacy)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @property
    def provider_id(self) -> str:
        return "claude-opus"

    async def generate(self, prompt: str, context: Dict[str, Any] = {}) -> str:
        if self.token:
            auth_method = "LASER Token"
            credential = self.token[:10] + "..."
        elif self.api_key:
            auth_method = "Legacy API Key"
            credential = "masked"
        else:
            return "[Error: No Claude Token found in ~/.pluribus/auth.json. Run 'laser_auth_tool.py auth claude']"
        
        # Simulate Network Latency
        await asyncio.sleep(0.5)
        
        return f"[Claude Opus] (Auth: {auth_method}) Processed: {prompt[:50]}..."

# Gemini Provider (Google) - Uses LASER Token
class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gemini-2.0-flash-thinking-exp"):
        super().__init__(model_name)
        self.token = load_auth_token("gemini")
        self.api_key = os.getenv("GOOGLE_API_KEY")

    @property
    def provider_id(self) -> str:
        return "gemini-2"

    async def generate(self, prompt: str, context: Dict[str, Any] = {}) -> str:
        if self.token:
            auth_method = "LASER Token"
        elif self.api_key:
            auth_method = "Legacy API Key"
        else:
            return "[Error: No Gemini Token found. Run 'laser_auth_tool.py auth gemini']"
            
        await asyncio.sleep(0.3)
        return f"[Gemini 2.0] (Auth: {auth_method}) Processed: {prompt[:50]}..."

# Codex Provider (GitHub Copilot) - Uses LASER Token
class CodexProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "copilot-chat"):
        super().__init__(model_name)
        self.token = load_auth_token("codex")
        self.api_key = os.getenv("GITHUB_TOKEN")

    @property
    def provider_id(self) -> str:
        return "codex"

    async def generate(self, prompt: str, context: Dict[str, Any] = {}) -> str:
        if self.token:
            auth_method = "LASER Token"
        elif self.api_key:
            auth_method = "Legacy API Key"
        else:
            return "[Error: No Codex Token found. Run 'laser_auth_tool.py auth codex']"
            
        await asyncio.sleep(0.2)
        return f"[Codex] (Auth: {auth_method}) Processed: {prompt[:50]}..."

class ProviderFactory:
    _instances = {}
    
    @classmethod
    def get_provider(cls, provider_id: str) -> Optional[BaseLLMProvider]:
        if provider_id not in cls._instances:
            if provider_id.startswith("claude"):
                cls._instances[provider_id] = ClaudeProvider()
            elif provider_id.startswith("gemini"):
                cls._instances[provider_id] = GeminiProvider()
            elif provider_id.startswith("codex"):
                cls._instances[provider_id] = CodexProvider()
            elif provider_id.startswith("qwen"):
                # Map Qwen to Codex or new Qwen as needed. 
                # For now, leaving as None or implementing strict Qwen separately if requested.
                # Returning None to enforce explicitly supported providers.
                return None
            else:
                return None
        return cls._instances[provider_id]
