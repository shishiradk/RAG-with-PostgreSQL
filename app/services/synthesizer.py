from typing import Any, Dict, List, Type, Optional
import time
import random
import logging

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from config.settings import get_settings


class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()
        # defaults for new features (can be overridden in settings)
        self.max_retries = getattr(self.settings, "max_retries", 3)
        self.backoff_base = getattr(self.settings, "backoff_base", 0.5)  # seconds
        self.backoff_max = getattr(self.settings, "backoff_max", 8.0)    # seconds
        # cost routing map: provider -> cost per 1k tokens (USD or arbitrary units)
        self.cost_map = getattr(self.settings, "cost_per_1k_tokens", {
            "openai": 0.03,
            "openrouter": 0.02,
            "groq": 0.015,
            "llama": 0.0,
            "anthropic": 0.05,
        })

    def _initialize_client(self) -> Any:
        client_initializers = {
            "openai": lambda s: instructor.from_openai(
                OpenAI(api_key=s.api_key),
                mode=instructor.Mode.JSON,
            ),
            "anthropic": lambda s: instructor.from_anthropic(
                Anthropic(api_key=s.api_key)
            ),
            "llama": lambda s: instructor.from_openai(
                OpenAI(base_url=s.base_url, api_key=s.api_key),
                mode=instructor.Mode.JSON,
            ),
            "openrouter": lambda s: instructor.from_openai(
                OpenAI(base_url=s.base_url, api_key=s.api_key),
                mode=instructor.Mode.JSON,
            ),
            "groq": lambda s: instructor.from_openai(
                OpenAI(base_url=s.base_url, api_key=s.api_key),
                mode=instructor.Mode.JSON,
            ),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            logging.debug("Initializing client for provider: %s", self.provider)
            return initializer(self.settings)

        raise ValueError(f"Unsupported provider: {self.provider}")

    # ---------- Utility: token estimation ----------
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Cheap heuristic to estimate tokens from messages.
        Counts words and scales to approximate tokens.
        """
        total_words = 0
        for m in messages:
            total_words += len(m.get("content", "").split())
        # heuristic: 1 token ≈ 0.75 words -> tokens = ceil(words / 0.75)
        tokens = int((total_words * 4 + 2) // 3)  # integer math approx for ceil(words/0.75)
        return max(tokens, 1)

    # ---------- Utility: auto model selection ----------
    def _select_model_by_tokens(self, tokens: int) -> str:
        """
        Select model based on estimated token count and settings.
        Uses settings.model_priority (list of tuples) if present:
            model_priority = [
                {"max_tokens": 2048, "model": "gpt-4o"},
                {"max_tokens": 8192, "model": "gpt-4o-mini"},
                ...
            ]
        Falls back to settings.default_model or settings.model.
        """
        priority = getattr(self.settings, "model_priority", None)
        if priority:
            for entry in priority:
                if tokens <= int(entry.get("max_tokens", 2**30)):
                    return entry.get("model")
        return getattr(self.settings, "default_model", getattr(self.settings, "model", None))

    # ---------- Utility: provider selection by cost ----------
    def _select_provider_by_cost(self, estimated_tokens: int, candidates: Optional[List[str]] = None) -> str:
        """
        Choose the cheapest provider per estimated cost for the job.
        estimated_tokens is count of tokens. cost_map uses cost per 1k tokens.
        """
        if not candidates:
            candidates = list(self.cost_map.keys())

        best = None
        best_cost = float("inf")
        for p in candidates:
            price_per_1k = float(self.cost_map.get(p, 0.0))
            # cost = price_per_1k * (tokens / 1000)
            cost = price_per_1k * (estimated_tokens / 1000.0)
            if cost < best_cost:
                best_cost = cost
                best = p
        return best or self.provider

    # ---------- Utility: retry with exponential backoff ----------
    def _call_with_retries(self, call_fn, *args, **kwargs):
        attempts = 0
        while True:
            try:
                attempts += 1
                start = time.time()
                result = call_fn(*args, **kwargs)
                elapsed = time.time() - start
                logging.info("Provider call succeeded on attempt %d in %.3fs", attempts, elapsed)
                return result
            except Exception as e:
                logging.warning("Provider call failed on attempt %d: %s", attempts, str(e))
                if attempts >= self.max_retries:
                    logging.error("Max retries reached (%d). Raising.", self.max_retries)
                    raise
                # exponential backoff with jitter
                backoff = min(self.backoff_max, self.backoff_base * (2 ** (attempts - 1)))
                jitter = random.uniform(0, backoff * 0.2)
                sleep_time = backoff + jitter
                logging.info("Sleeping %.3fs before retry %d", sleep_time, attempts + 1)
                time.sleep(sleep_time)

    # ---------- Embedding API with retries and logging ----------
    def create_embedding(self, text: str) -> List[float]:
        embed_model = getattr(self.settings, "embedding_model", None)
        if not embed_model:
            raise ValueError(f"No embedding_model defined for provider {self.provider}")

        # Use provider-specific base_url if available
        embed_client = OpenAI(
            api_key=self.settings.api_key,
            base_url=getattr(self.settings, "base_url", None),
        )

        def _call(text_inner):
            return embed_client.embeddings.create(
                model=embed_model,
                input=text_inner.replace("\n", " "),
            )

        logging.debug("Generating embedding using provider '%s' model '%s'.", self.provider, embed_model)
        response = self._call_with_retries(_call, text)
        embedding = response.data[0].embedding
        logging.debug("Embedding length: %d", len(embedding))
        return embedding

    # ---------- Completion with auto-fallback, model selection, cost routing, retries ----------
    def create_completion(
        self,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        # estimate tokens
        estimated_tokens = self._estimate_tokens(messages)
        logging.debug("Estimated tokens for request: %d", estimated_tokens)

        # auto model selection
        selected_model = kwargs.get("model") or self._select_model_by_tokens(estimated_tokens)
        logging.debug("Selected model: %s", selected_model)

        # cost-based routing (if caller allows alternative providers)
        allow_fallback_routing = kwargs.get("enable_cost_routing", True)
        provider_to_use = self.provider
        if allow_fallback_routing:
            provider_to_use = self._select_provider_by_cost(estimated_tokens)
            if provider_to_use != self.provider:
                logging.info("Cost routing selected provider '%s' over '%s'", provider_to_use, self.provider)

        # build completion parameters
        completion_params = {
            "model": selected_model,
            "temperature": kwargs.get("temperature", getattr(self.settings, "temperature", 0.0)),
            "max_tokens": kwargs.get("max_tokens", getattr(self.settings, "max_tokens", None)),
            "max_retries": kwargs.get("max_retries", getattr(self.settings, "max_retries", self.max_retries)),
            "response_model": response_model,
            "messages": messages,
        }

        # if provider_to_use differs, construct a temporary client
        if provider_to_use != self.provider:
            temp_settings = getattr(get_settings(), provider_to_use)
            temp_client = instructor.from_openai(
                OpenAI(base_url=getattr(temp_settings, "base_url", None), api_key=temp_settings.api_key),
                mode=instructor.Mode.JSON,
            )
            call_target = temp_client.completions.create
            log_provider = provider_to_use
        else:
            call_target = self.client.completions.create
            log_provider = self.provider

        logging.info("Calling provider '%s' model '%s' with estimated %d tokens", log_provider, selected_model, estimated_tokens)
        try:
            return self._call_with_retries(call_target, **completion_params)
        except Exception:
            logging.warning("Primary provider failed; attempting fallback chain.")
            return self._fallback(response_model, messages, **kwargs)

    # ---------- Fallback chain ----------
    def _fallback(
        self,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        **kwargs,
    ):
        fallback_order = ["groq", "openrouter", "openai", "llama", "anthropic"]
        # remove original provider if in list
        if self.provider in fallback_order:
            fallback_order.remove(self.provider)

        for p in fallback_order:
            try:
                logging.info("Trying fallback provider: %s", p)
                fallback_llm = LLMFactory(p)
                return fallback_llm.create_completion(
                    response_model=response_model,
                    messages=messages,
                    **kwargs,
                )
            except Exception as e:
                logging.warning("Fallback provider %s failed: %s", p, str(e))
                continue

        logging.error("All providers in fallback chain failed.")
        raise RuntimeError("All providers failed.")
