from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests
from .config import Config, LLMConfig, load_config

class OllamaUnavailableError(RuntimeError):
    pass

@dataclass
class OllamaClient:
    cfg: LLMConfig
    _availability_checked: bool = False
    _availability_cache: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[Config] = None) -> "OllamaClient":
        cfg = cfg or load_config()
        return cls(cfg=cfg.llm)

    def _chat_url(self) -> str:
        return self.cfg.host.rstrip("/") + "/api/chat"

    def is_available(self, force_recheck: bool = False) -> bool:
        if self._availability_checked and not force_recheck:
            return self._availability_cache
        try:
            resp = requests.get(self.cfg.host.rstrip("/") + "/api/tags", timeout=5)
            self._availability_cache = resp.status_code == 200
        except requests.RequestException:
            self._availability_cache = False
        self._availability_checked = True
        return self._availability_cache

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise OllamaUnavailableError(
                f"Ollama server at {self.cfg.host} is not reachable (cached availability check)."
            )

        retries = max_retries if max_retries is not None else self.cfg.max_retries
        last_err: Optional[Exception] = None

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }

        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    self._chat_url(),
                    json=payload,
                    timeout=self.cfg.request_timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return json.loads(content)
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                last_err = e
                if isinstance(e, requests.RequestException):
                    self._availability_cache = False
                    self._availability_checked = True
                    break
                time.sleep(min(2 ** attempt, 8))
                continue

        raise OllamaUnavailableError(
            f"Could not get a valid JSON response from Ollama at {self.cfg.host} "
            f"(model={self.cfg.model}) after {retries} attempts: {last_err}"
        )

    def chat_text(self, system_prompt: str, user_prompt: str, max_retries: Optional[int] = None) -> str:
        """Call the model for free-form text (no forced JSON format)."""
        if not self.is_available():
            raise OllamaUnavailableError(
                f"Ollama server at {self.cfg.host} is not reachable (cached availability check)."
            )

        retries = max_retries if max_retries is not None else self.cfg.max_retries
        last_err: Optional[Exception] = None

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }

        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    self._chat_url(),
                    json=payload,
                    timeout=self.cfg.request_timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
            except requests.RequestException as e:
                last_err = e
                self._availability_cache = False
                self._availability_checked = True
                break

        raise OllamaUnavailableError(
            f"Could not get a response from Ollama at {self.cfg.host} "
            f"(model={self.cfg.model}) after {retries} attempts: {last_err}"
        )

if __name__ == "__main__":
    client = OllamaClient.from_config()
    print(f"Ollama host: {client.cfg.host}")
    print(f"Ollama model: {client.cfg.model}")
    print(f"Available: {client.is_available()}")
    if client.is_available():
        try:
            out = client.chat_json(
                system_prompt="Respond only with JSON of the form {\"ok\": true}.",
                user_prompt="ping",
            )
            print("Test response:", out)
        except OllamaUnavailableError as e:
            print("Test call failed:", e)