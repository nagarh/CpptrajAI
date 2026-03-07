"""
Unified LLM backend — Claude, OpenAI, Gemini.
All three support reliable function calling / tool use.
"""

from __future__ import annotations
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any


def _extract_text_tool_calls(text: str) -> list[dict]:
    """Fallback: parse tool calls that a model printed as JSON text instead of using native calling."""
    calls = []
    # Match {"name": "...", "arguments": {...}} or {"name": "...", "input": {...}}
    pattern = r'\{[\s\S]*?"name"\s*:\s*"([^"]+)"[\s\S]*?\}'
    for m in re.finditer(pattern, text):
        try:
            obj = json.loads(m.group(0))
            name = obj.get("name")
            inp  = obj.get("arguments") or obj.get("input") or obj.get("parameters") or {}
            if name and isinstance(inp, dict):
                calls.append({"id": f"text_{len(calls)}", "name": name, "input": inp})
        except Exception:
            continue
    return calls

PROVIDER_DEFAULTS = {
    "claude": {
        "default_model": "claude-haiku-4-5-20251001",
        "label": "Anthropic Claude",
        "models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "label": "Google Gemini",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    },
}


class LLMBackend(ABC):
    @abstractmethod
    def chat(self, messages, tools, system) -> tuple[str, list[dict], bool]: ...

    @abstractmethod
    def stream_chat(self, messages, tools, system):
        """Yields ('text', chunk), then ('tool_calls', list), then ('stop_reason', str)."""
        ...

    @abstractmethod
    def make_assistant_message(self, text: str, tool_calls: list[dict]) -> dict: ...

    @abstractmethod
    def make_tool_result_message(self, tool_calls: list[dict], results: list[str]) -> dict: ...

    @property
    @abstractmethod
    def provider(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...


class ClaudeBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._model = model
        # Pass api_key explicitly; use a dummy if empty to prevent SDK env-var fallback
        self._client = anthropic.Anthropic(api_key=api_key or "no-key")

    @property
    def provider(self): return "claude"
    @property
    def model(self): return self._model

    def chat(self, messages, tools, system):
        response = self._client.messages.create(
            model=self._model, max_tokens=4096,
            system=system, tools=self._claude_tools(tools), messages=messages,
        )
        text_parts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        return "\n".join(text_parts), tool_calls, response.stop_reason == "tool_use"

    def _claude_tools(self, tools):
        out = []
        for t in tools:
            if "input_schema" in t:
                out.append(t)
            else:
                fn = t.get("function", t)
                out.append({"name": fn["name"], "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {"type": "object", "properties": {}})})
        return out

    def stream_chat(self, messages, tools, system):
        claude_tools = self._claude_tools(tools)
        with self._client.messages.stream(
            model=self._model, max_tokens=4096,
            system=system, tools=claude_tools, messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield ("text", text)
            final = stream.get_final_message()
            tool_calls = [
                {"id": b.id, "name": b.name, "input": b.input}
                for b in final.content if b.type == "tool_use"
            ]
            yield ("tool_calls", tool_calls)
            yield ("stop_reason", final.stop_reason)

    def make_assistant_message(self, text, tool_calls):
        content = []
        if text: content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
        return {"role": "assistant", "content": content}

    def make_tool_result_message(self, tool_calls, results):
        return {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tc["id"], "content": r}
            for tc, r in zip(tool_calls, results)
        ]}


class OpenAICompatBackend(LLMBackend):
    def __init__(self, api_key: str, model: str, base_url: str, provider_name: str):
        from openai import OpenAI
        self._provider = provider_name
        self._model = model
        self._client = OpenAI(api_key=api_key or "no-key", base_url=base_url)

    @property
    def provider(self): return self._provider
    @property
    def model(self): return self._model

    def _oai_tools(self, tools):
        out = []
        for t in tools:
            if "function" in t:
                out.append(t)
            else:
                out.append({"type": "function", "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                }})
        return out

    def stream_chat(self, messages, tools, system):
        oai_tools = self._oai_tools(tools)
        full_messages = [{"role": "system", "content": system}] + messages
        kwargs: dict[str, Any] = dict(model=self._model, messages=full_messages, max_tokens=4096, stream=True)
        if oai_tools:
            kwargs["tools"] = oai_tools
        response = self._client.chat.completions.create(**kwargs)
        tc_acc: dict[int, dict] = {}
        text_chunks: list[str] = []
        finish_reason = "stop"
        for chunk in response:
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            if choice.delta.content:
                text_chunks.append(choice.delta.content)
                yield ("text", choice.delta.content)
            if choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id: tc_acc[idx]["id"] = tc.id
                    if tc.function and tc.function.name: tc_acc[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments: tc_acc[idx]["arguments"] += tc.function.arguments
        tool_calls = []
        for idx in sorted(tc_acc):
            tc = tc_acc[idx]
            try: inp = json.loads(tc["arguments"])
            except Exception: inp = {}
            tool_calls.append({"id": tc["id"], "name": tc["name"], "input": inp})

        yield ("tool_calls", tool_calls)
        yield ("stop_reason", "tool_calls" if finish_reason == "tool_calls" else "end_turn")

    def chat(self, messages, tools, system):
        oai_tools = self._oai_tools(tools)
        full_messages = [{"role": "system", "content": system}] + messages
        kwargs: dict[str, Any] = dict(model=self._model, messages=full_messages, max_tokens=4096)
        if oai_tools:
            kwargs["tools"] = oai_tools
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    inp = json.loads(tc.function.arguments)
                except Exception:
                    inp = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "input": inp})

        # Fallback: model printed tool calls as text instead of using native calling
        if not tool_calls and text:
            tool_calls = _extract_text_tool_calls(text)

        return text, tool_calls, choice.finish_reason == "tool_calls" or bool(tool_calls)

    def make_assistant_message(self, text, tool_calls):
        msg: dict[str, Any] = {"role": "assistant", "content": text or ""}
        if tool_calls:
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}}
                for tc in tool_calls
            ]
        return msg

    def make_tool_result_message(self, tool_calls, results):
        return {"_multi": [
            {"role": "tool", "tool_call_id": tc["id"], "content": r}
            for tc, r in zip(tool_calls, results)
        ]}


REACT_SYSTEM_SUFFIX = """

## HOW TO CALL TOOLS
You do NOT have access to native function calling. Instead, when you need to use a tool,
output a line in this EXACT format — nothing before or after the ACTION line:

ACTION: {"name": "TOOL_NAME", "arguments": {KEY: VALUE}}

Available tools:
- run_cpptraj_script  → arguments: {"script": "<full cpptraj script>", "description": "<what it does>"}
- run_python_script   → arguments: {"script": "<full python script>", "description": "<what it does>"}
- read_output_file    → arguments: {"filename": "<filename.dat>"}
- list_output_files   → arguments: {}

RULES:
- Output ONLY the ACTION line when calling a tool — no other text on that line
- Wait for the RESULT before continuing
- After seeing RESULT, interpret it and respond to the user

EXAMPLE:
User: how many frames in my trajectory?
ACTION: {"name": "run_cpptraj_script", "arguments": {"script": "parm protein.prmtop\\ntrajin traj.nc\\ngo", "description": "Count frames"}}
RESULT: Success: True\\nSTDOUT: ... 500 frames ...
Answer: Your trajectory has 500 frames.
"""


class OllamaReActBackend(LLMBackend):
    """ReAct-style backend for Ollama — no native tool calling, parses ACTION: lines instead."""

    def __init__(self, model: str, base_url: str):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key="ollama", base_url=base_url)

    @property
    def provider(self): return "ollama"
    @property
    def model(self): return self._model

    def _parse_action(self, text: str) -> list[dict]:
        """Extract ACTION: {...} lines from model output."""
        calls = []
        for line in text.splitlines():
            line = line.strip()
            if not line.upper().startswith("ACTION:"):
                continue
            raw = line[line.index(":")+1:].strip()
            try:
                obj  = json.loads(raw)
                name = obj.get("name", "")
                inp  = obj.get("arguments") or obj.get("input") or {}
                if name:
                    calls.append({"id": f"react_{len(calls)}", "name": name, "input": inp})
            except Exception:
                # Try fallback generic JSON extraction
                extracted = _extract_text_tool_calls(raw)
                calls.extend(extracted)
        return calls

    def stream_chat(self, messages, tools, system):
        full_system = system + REACT_SYSTEM_SUFFIX
        full_messages = [{"role": "system", "content": full_system}] + messages
        response = self._client.chat.completions.create(
            model=self._model, messages=full_messages, max_tokens=4096, stream=True
        )
        text_chunks = []
        for chunk in response:
            choice = chunk.choices[0]
            if choice.delta.content:
                text_chunks.append(choice.delta.content)
                yield ("text", choice.delta.content)

        full_text = "".join(text_chunks)
        tool_calls = self._parse_action(full_text)
        yield ("tool_calls", tool_calls)
        yield ("stop_reason", "tool_calls" if tool_calls else "end_turn")

    def chat(self, messages, tools, system):
        full_system = system + REACT_SYSTEM_SUFFIX
        full_messages = [{"role": "system", "content": full_system}] + messages
        response = self._client.chat.completions.create(
            model=self._model, messages=full_messages, max_tokens=4096
        )
        text = response.choices[0].message.content or ""
        tool_calls = self._parse_action(text)
        return text, tool_calls, bool(tool_calls)

    def make_assistant_message(self, text, tool_calls):
        # Inject RESULT markers so the model sees tool outputs in conversation
        if tool_calls:
            action_lines = "\n".join(
                f'ACTION: {json.dumps({"name": tc["name"], "arguments": tc["input"]})}'
                for tc in tool_calls
            )
            return {"role": "assistant", "content": action_lines}
        return {"role": "assistant", "content": text}

    def make_tool_result_message(self, tool_calls, results):
        result_text = "\n".join(
            f"RESULT for {tc['name']}:\n{r}" for tc, r in zip(tool_calls, results)
        )
        return {"role": "user", "content": result_text}


def create_backend(provider: str, api_key: str = "", model: str = "", base_url: str = "") -> LLMBackend:
    provider  = provider.lower().strip()
    defaults  = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    model     = model    or defaults["default_model"]
    base_url  = base_url or defaults.get("base_url", "")
    api_key   = api_key

    if provider == "claude":
        return ClaudeBackend(api_key=api_key, model=model)
    if provider == "ollama":
        return OllamaReActBackend(model=model, base_url=base_url)
    return OpenAICompatBackend(api_key=api_key, model=model, base_url=base_url, provider_name=provider)
