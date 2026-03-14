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
        "default_model": "gpt-4o-mini",
        "label": "OpenAI",
        "models": ["gpt-4o-mini"],
    },
    "gemini": {
        "default_model": "gemini-2.5-flash",
        "label": "Google Gemini",
        "models": ["gemini-2.5-flash"],
    },
    "ollama": {
        "default_model": "deepseek-v3",
        "label": "Ollama (local)",
        "base_url": "http://localhost:11434/v1",
        "models": ["deepseek-v3"],
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
        kwargs: dict[str, Any] = dict(model=self._model, messages=full_messages, stream=True)
        if self._provider != "ollama":
            kwargs["max_tokens"] = 4096
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

        # Fallback: model printed tool calls as text instead of using native calling
        if not tool_calls and text_chunks:
            full_text = "".join(text_chunks)
            tool_calls = _extract_text_tool_calls(full_text)
            if tool_calls:
                # Signal that the streamed text was actually a tool call, not display text
                yield ("retract_text", None)

        yield ("tool_calls", tool_calls)
        yield ("stop_reason", "tool_calls" if (finish_reason == "tool_calls" or tool_calls) else "end_turn")

    def chat(self, messages, tools, system):
        oai_tools = self._oai_tools(tools)
        full_messages = [{"role": "system", "content": system}] + messages
        kwargs: dict[str, Any] = dict(model=self._model, messages=full_messages)
        if self._provider != "ollama":
            kwargs["max_tokens"] = 4096
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


class OpenAIResponsesBackend(LLMBackend):
    """
    Uses the OpenAI Responses API (client.responses.create).
    Required for accounts that don't have Chat Completions access for newer models.
    """

    def __init__(self, api_key: str, model: str, provider_name: str = "openai"):
        from openai import OpenAI
        self._provider = provider_name
        self._model = model
        self._client = OpenAI(api_key=api_key or "no-key")

    @property
    def provider(self): return self._provider
    @property
    def model(self): return self._model

    def _resp_tools(self, tools):
        out = []
        for t in tools:
            if "function" in t:
                fn = t["function"]
                out.append({"type": "function", "name": fn["name"],
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters", {"type": "object", "properties": {}})})
            else:
                out.append({"type": "function", "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema", {"type": "object", "properties": {}})})
        return out

    def _to_input(self, messages: list) -> list:
        """Convert internal chat history to Responses API input items."""
        items = []
        for msg in messages:
            role = msg.get("role", "")

            # Tool results stored as _multi
            if "_multi" in msg:
                for tm in msg["_multi"]:
                    items.append({"type": "function_call_output",
                                  "call_id": tm["tool_call_id"],
                                  "output": tm["content"]})
                continue

            # Assistant message (may have tool_calls)
            if role == "assistant":
                content = msg.get("content") or ""
                if content:
                    items.append({"role": "assistant", "content": content})
                for tc in msg.get("tool_calls", []):
                    items.append({"type": "function_call",
                                  "call_id": tc["id"],
                                  "name": tc["function"]["name"],
                                  "arguments": tc["function"]["arguments"]})
                continue

            # Plain user/tool messages
            if role == "user":
                items.append({"role": "user", "content": msg.get("content") or ""})
            elif role == "tool":
                items.append({"type": "function_call_output",
                              "call_id": msg.get("tool_call_id", ""),
                              "output": msg.get("content") or ""})
        return items

    def chat(self, messages, tools, system):
        resp_tools = self._resp_tools(tools)
        input_items = self._to_input(messages)
        kwargs: dict[str, Any] = dict(model=self._model, input=input_items, instructions=system)
        if resp_tools:
            kwargs["tools"] = resp_tools

        response = self._client.responses.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for item in response.output:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", "") == "output_text":
                        text_parts.append(block.text)
            elif item_type == "function_call":
                try:
                    inp = json.loads(item.arguments)
                except Exception:
                    inp = {}
                tool_calls.append({"id": item.call_id, "name": item.name, "input": inp})

        return "\n".join(text_parts), tool_calls, bool(tool_calls)

    def stream_chat(self, messages, tools, system):
        # Use non-streaming chat for reliable tool call extraction.
        # Complex streaming accumulation of function call arguments is error-prone.
        text, tool_calls, _ = self.chat(messages, tools, system)
        if text:
            yield ("text", text)
        yield ("tool_calls", tool_calls)
        yield ("stop_reason", "tool_calls" if tool_calls else "end_turn")

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


class GeminiNativeBackend(LLMBackend):
    """Native Google Generative AI backend — works with any AI Studio key."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai
        self._genai = genai
        self._model_name = model
        genai.configure(api_key=api_key or "no-key")

    @property
    def provider(self): return "gemini"
    @property
    def model(self): return self._model_name

    def _gemini_tools(self, tools):
        protos = self._genai.protos
        declarations = []
        for t in tools:
            if "function" in t:
                fn, params = t["function"], t["function"].get("parameters", {})
            else:
                fn, params = t, t.get("input_schema", {})
            props = {}
            for pname, pschema in params.get("properties", {}).items():
                ptype = pschema.get("type", "string").upper()
                gemini_type = getattr(protos.Type, ptype, protos.Type.STRING)
                props[pname] = protos.Schema(type=gemini_type,
                                             description=pschema.get("description", ""))
            declarations.append(protos.FunctionDeclaration(
                name=fn["name"] if "function" in t else t["name"],
                description=fn.get("description", ""),
                parameters=protos.Schema(type=protos.Type.OBJECT,
                                         properties=props,
                                         required=params.get("required", [])),
            ))
        return [protos.Tool(function_declarations=declarations)]

    def _to_contents(self, messages):
        protos = self._genai.protos
        contents = []
        for msg in messages:
            role = msg.get("role", "")
            if "_fn_responses" in msg:
                parts = [protos.Part(function_response=protos.FunctionResponse(
                             name=fr["name"], response={"result": fr["response"]}))
                         for fr in msg["_fn_responses"]]
                contents.append(protos.Content(role="user", parts=parts))
            elif role == "user":
                contents.append(protos.Content(role="user",
                    parts=[protos.Part(text=msg.get("content") or "")]))
            elif role in ("assistant", "model"):
                parts = []
                if msg.get("content"):
                    parts.append(protos.Part(text=msg["content"]))
                for fc in msg.get("_fn_calls", []):
                    parts.append(protos.Part(function_call=protos.FunctionCall(
                        name=fc["name"], args=fc["args"])))
                if parts:
                    contents.append(protos.Content(role="model", parts=parts))
        return contents

    def chat(self, messages, tools, system):
        model = self._genai.GenerativeModel(
            self._model_name,
            tools=self._gemini_tools(tools) if tools else None,
            system_instruction=system,
        )
        contents = self._to_contents(messages)
        response = model.generate_content(contents)
        text_parts, tool_calls = [], []
        for part in response.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            elif hasattr(part, "function_call") and part.function_call.name:
                fc = part.function_call
                tool_calls.append({"id": fc.name, "name": fc.name, "input": dict(fc.args)})
        return "\n".join(text_parts), tool_calls, bool(tool_calls)

    def stream_chat(self, messages, tools, system):
        text, tool_calls, _ = self.chat(messages, tools, system)
        if text:
            yield ("text", text)
        yield ("tool_calls", tool_calls)
        yield ("stop_reason", "tool_calls" if tool_calls else "end_turn")

    def make_assistant_message(self, text, tool_calls):
        msg: dict[str, Any] = {"role": "model", "content": text or ""}
        if tool_calls:
            msg["_fn_calls"] = [{"name": tc["name"], "args": tc["input"]} for tc in tool_calls]
        return msg

    def make_tool_result_message(self, tool_calls, results):
        return {"role": "user", "_fn_responses": [
            {"name": tc["name"], "response": r}
            for tc, r in zip(tool_calls, results)
        ]}


def create_backend(provider: str, api_key: str = "", model: str = "", base_url: str = "") -> LLMBackend:
    provider = provider.lower().strip()
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    model    = model    or defaults["default_model"]
    base_url = base_url or defaults.get("base_url", "")

    if provider == "claude":
        return ClaudeBackend(api_key=api_key, model=model)
    if provider == "openai":
        return OpenAIResponsesBackend(api_key=api_key, model=model)
    if provider == "gemini":
        return GeminiNativeBackend(api_key=api_key, model=model)
    if provider == "ollama":
        resolved_url = base_url or PROVIDER_DEFAULTS["ollama"]["base_url"]
        return OpenAICompatBackend(api_key="ollama", model=model, base_url=resolved_url, provider_name="ollama")
    return OpenAICompatBackend(api_key=api_key, model=model, base_url=base_url, provider_name=provider)
