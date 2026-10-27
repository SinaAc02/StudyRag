import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from collections.abc import Iterator


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class StreamingTextGenerator(Protocol):
    def stream(self, prompt: str) -> Iterator[str]: ...


@dataclass(frozen=True, slots=True)
class OllamaGenerator:
    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 300.0

    def unload(self) -> None:
        """Release this configured model only; do not unload unrelated models."""
        request = Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps({"model": self.model, "keep_alive": 0, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"Could not unload {self.model}: {error}") from error

    def generate(self, prompt: str) -> str:
        answer = "".join(self.stream(prompt)).strip()
        if not answer:
            raise RuntimeError("Ollama returned no answer text")
        return answer

    def stream(self, prompt: str) -> Iterator[str]:
        yield from self._stream_request(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.2},
            },
            "generate",
        )

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield from self._stream_request(
            {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.2, "num_ctx": 16384},
            },
            "chat",
        )

    def _stream_request(self, payload: dict, endpoint: str) -> Iterator[str]:
        request = Request(
            f"{self.base_url.rstrip('/')}/api/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                completed = False
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise RuntimeError("Ollama returned an invalid streaming response") from error
                    if not isinstance(result, dict):
                        raise RuntimeError("Ollama returned an invalid streaming response")
                    if result.get("error"):
                        raise RuntimeError(f"Ollama generation failed: {result['error']}")
                    if endpoint == "chat":
                        message = result.get("message", {})
                        if not isinstance(message, dict):
                            raise RuntimeError("Ollama returned an invalid chat message")
                        fragment = message.get("content", "")
                    else:
                        fragment = result.get("response", "")
                    if not isinstance(fragment, str):
                        raise RuntimeError("Ollama returned invalid answer text")
                    if fragment:
                        yield fragment
                    if result.get("done") is True:
                        completed = True
                        break
                if not completed:
                    raise RuntimeError("Ollama stream ended before generation completed")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. Start Ollama and confirm the model is installed."
            ) from error
        except TimeoutError as error:
            raise RuntimeError(f"Ollama did not finish within {self.timeout_seconds:g} seconds.") from error
