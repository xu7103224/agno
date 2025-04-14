from collections.abc import AsyncIterator
from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx
from pydantic import BaseModel

from agno.exceptions import ModelProviderError
from agno.media import AudioResponse
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.utils.log import log_error, log_warning
from agno.utils.openai import audio_to_message, images_to_message

try:
    from openai import APIConnectionError, APIStatusError, RateLimitError
    from openai import AsyncOpenAI as AsyncOpenAIClient
    from openai import OpenAI as OpenAIClient
    from openai.types.chat import ChatCompletionAudio
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_chunk import (
        ChatCompletionChunk,
        ChoiceDelta,
        ChoiceDeltaToolCall,
    )
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
except (ImportError, ModuleNotFoundError):
    raise ImportError("`openai` not installed. Please install using `pip install openai`")


@dataclass
class OpenAIChat(Model):
    """
    A class for interacting with OpenAI models using the Chat completions API.

    For more information, see: https://platform.openai.com/docs/api-reference/chat/create
    """

    id: str = "gpt-4o"
    name: str = "OpenAIChat"
    provider: str = "OpenAI"
    supports_native_structured_outputs: bool = True

    # Request parameters
    store: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    modalities: Optional[List[str]] = None  # "text" and/or "audio"
    audio: Optional[Dict[str, Any]] = (
        None  # E.g. {"voice": "alloy", "format": "wav"}. `format` must be one of `wav`, `mp3`, `flac`, `opus`, or `pcm16`. `voice` must be one of `ash`, `ballad`, `coral`, `sage`, `verse`, `alloy`, `echo`, and `shimmer`.
    )
    presence_penalty: Optional[float] = None
    response_format: Optional[Any] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    user: Optional[str] = None
    top_p: Optional[float] = None
    extra_headers: Optional[Any] = None
    extra_query: Optional[Any] = None
    request_params: Optional[Dict[str, Any]] = None

    # Client parameters
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    default_headers: Optional[Any] = None
    default_query: Optional[Any] = None
    http_client: Optional[httpx.Client] = None
    client_params: Optional[Dict[str, Any]] = None

    # OpenAI clients
    client: Optional[OpenAIClient] = None
    async_client: Optional[AsyncOpenAIClient] = None

    # Internal parameters. Not used for API requests
    # Whether to use the structured outputs with this Model.
    structured_outputs: bool = False

    # The role to map the message role to.
    role_map = {
        "system": "developer",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
        "model": "assistant",
    }

    def _get_client_params(self) -> Dict[str, Any]:
        # Fetch API key from env if not already set
        if not self.api_key:
            self.api_key = getenv("OPENAI_API_KEY")
            if not self.api_key:
                log_error("OPENAI_API_KEY not set. Please set the OPENAI_API_KEY environment variable.")

        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # Add additional client params if provided
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    def get_client(self) -> OpenAIClient:
        """
        Returns an OpenAI client.

        Returns:
            OpenAIClient: An instance of the OpenAI client.
        """
        if self.client:
            return self.client

        client_params: Dict[str, Any] = self._get_client_params()
        if self.http_client is not None:
            client_params["http_client"] = self.http_client
        self.client = OpenAIClient(**client_params)
        return self.client

    def get_async_client(self) -> AsyncOpenAIClient:
        """
        Returns an asynchronous OpenAI client.

        Returns:
            AsyncOpenAIClient: An instance of the asynchronous OpenAI client.
        """
        if self.async_client:
            return self.async_client

        client_params: Dict[str, Any] = self._get_client_params()
        if self.http_client:
            client_params["http_client"] = self.http_client
        else:
            # Create a new async HTTP client with custom limits
            client_params["http_client"] = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            )
        return AsyncOpenAIClient(**client_params)

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        # Define base request parameters
        base_params = {
            "store": self.store,
            "reasoning_effort": self.reasoning_effort,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "modalities": self.modalities,
            "audio": self.audio,
            "presence_penalty": self.presence_penalty,
            "response_format": self.response_format,
            "seed": self.seed,
            "stop": self.stop,
            "temperature": self.temperature,
            "user": self.user,
            "top_p": self.top_p,
            "extra_headers": self.extra_headers,
            "extra_query": self.extra_query,
            "metadata": self.metadata,
        }

        # Filter out None values
        request_params = {k: v for k, v in base_params.items() if v is not None}

        # Add tools
        if self._tools is not None and len(self._tools) > 0:
            request_params["tools"] = self._tools

            if self.tool_choice is not None:
                request_params["tool_choice"] = self.tool_choice

        # Add additional request params if provided
        if self.request_params:
            request_params.update(self.request_params)
        return request_params

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.

        Returns:
            Dict[str, Any]: The dictionary representation of the model.
        """
        model_dict = super().to_dict()
        model_dict.update(
            {
                "store": self.store,
                "frequency_penalty": self.frequency_penalty,
                "logit_bias": self.logit_bias,
                "logprobs": self.logprobs,
                "top_logprobs": self.top_logprobs,
                "max_tokens": self.max_tokens,
                "max_completion_tokens": self.max_completion_tokens,
                "modalities": self.modalities,
                "audio": self.audio,
                "presence_penalty": self.presence_penalty,
                "response_format": self.response_format
                if isinstance(self.response_format, dict)
                else str(self.response_format),
                "seed": self.seed,
                "stop": self.stop,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "user": self.user,
                "extra_headers": self.extra_headers,
                "extra_query": self.extra_query,
            }
        )
        if self._tools is not None:
            model_dict["tools"] = self._tools
            if self.tool_choice is not None:
                model_dict["tool_choice"] = self.tool_choice
            else:
                model_dict["tool_choice"] = "auto"
        cleaned_dict = {k: v for k, v in model_dict.items() if v is not None}
        return cleaned_dict

    def _format_message(self, message: Message) -> Dict[str, Any]:
        """
        Format a message into the format expected by OpenAI.

        Args:
            message (Message): The message to format.

        Returns:
            Dict[str, Any]: The formatted message.
        """
        message_dict: Dict[str, Any] = {
            "role": self.role_map[message.role],
            "content": message.content,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "tool_calls": message.tool_calls,
        }
        message_dict = {k: v for k, v in message_dict.items() if v is not None}

        # Ignore non-string message content
        # because we assume that the images/audio are already added to the message
        if (message.images is not None and len(message.images) > 0) or (
            message.audio is not None and len(message.audio) > 0
        ):
            # Ignore non-string message content
            # because we assume that the images/audio are already added to the message
            if isinstance(message.content, str):
                message_dict["content"] = [{"type": "text", "text": message.content}]
                if message.images is not None:
                    message_dict["content"].extend(images_to_message(images=message.images))

                if message.audio is not None:
                    message_dict["content"].extend(audio_to_message(audio=message.audio))

        if message.audio_output is not None:
            message_dict["content"] = None
            message_dict["audio"] = {"id": message.audio_output.id}

        if message.videos is not None and len(message.videos) > 0:
            log_warning("Video input is currently unsupported.")

        # OpenAI expects the tool_calls to be None if empty, not an empty list
        if message.tool_calls is not None and len(message.tool_calls) == 0:
            message_dict["tool_calls"] = None

        # Manually add the content field even if it is None
        if message.content is None:
            message_dict["content"] = None

        return message_dict

    def invoke(self, messages: List[Message]) -> Union[ChatCompletion, ParsedChatCompletion]:
        """
        Send a chat completion request to the OpenAI API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            ChatCompletion: The chat completion response from the API.
        """

        try:
            if self.response_format is not None and self.structured_outputs:
                if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                    return self.get_client().beta.chat.completions.parse(
                        model=self.id,
                        messages=[self._format_message(m) for m in messages],  # type: ignore
                        **self.request_kwargs,
                    )
                else:
                    raise ValueError("response_format must be a subclass of BaseModel if structured_outputs=True")

            return self.get_client().chat.completions.create(
                model=self.id,
                messages=[self._format_message(m) for m in messages],  # type: ignore
                **self.request_kwargs,
            )
        except RateLimitError as e:
            log_error(f"Rate limit error from OpenAI API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except APIConnectionError as e:
            log_error(f"API connection error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except APIStatusError as e:
            log_error(f"API status error from OpenAI API: {e}")
            try:
                error_message = e.response.json().get("error", {})
            except Exception:
                error_message = e.response.text
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except Exception as e:
            log_error(f"Error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

    async def ainvoke(self, messages: List[Message]) -> Union[ChatCompletion, ParsedChatCompletion]:
        """
        Sends an asynchronous chat completion request to the OpenAI API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            ChatCompletion: The chat completion response from the API.
        """

        try:
            if self.response_format is not None and self.structured_outputs:
                if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                    return await self.get_async_client().beta.chat.completions.parse(
                        model=self.id,
                        messages=[self._format_message(m) for m in messages],  # type: ignore
                        **self.request_kwargs,
                    )
                else:
                    raise ValueError("response_format must be a subclass of BaseModel if structured_outputs=True")
            return await self.get_async_client().chat.completions.create(
                model=self.id,
                messages=[self._format_message(m) for m in messages],  # type: ignore
                **self.request_kwargs,
            )
        except RateLimitError as e:
            log_error(f"Rate limit error from OpenAI API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except APIConnectionError as e:
            log_error(f"API connection error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except APIStatusError as e:
            log_error(f"API status error from OpenAI API: {e}")
            try:
                error_message = e.response.json().get("error", {})
            except Exception:
                error_message = e.response.text
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except Exception as e:
            log_error(f"Error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

    def invoke_stream(self, messages: List[Message]) -> Iterator[ChatCompletionChunk]:
        """
        Send a streaming chat completion request to the OpenAI API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Iterator[ChatCompletionChunk]: An iterator of chat completion chunks.
        """

        try:
            yield from self.get_client().chat.completions.create(
                model=self.id,
                messages=[self._format_message(m) for m in messages],  # type: ignore
                stream=True,
                stream_options={"include_usage": True},
                **self.request_kwargs,
            )  # type: ignore
        except RateLimitError as e:
            log_error(f"Rate limit error from OpenAI API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except APIConnectionError as e:
            log_error(f"API connection error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except APIStatusError as e:
            log_error(f"API status error from OpenAI API: {e}")
            try:
                error_message = e.response.json().get("error", {})
            except Exception:
                error_message = e.response.text
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except Exception as e:
            log_error(f"Error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

    async def ainvoke_stream(self, messages: List[Message]) -> AsyncIterator[ChatCompletionChunk]:
        """
        Sends an asynchronous streaming chat completion request to the OpenAI API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Any: An asynchronous iterator of chat completion chunks.
        """

        try:
            async_stream = await self.get_async_client().chat.completions.create(
                model=self.id,
                messages=[self._format_message(m) for m in messages],  # type: ignore
                stream=True,
                stream_options={"include_usage": True},
                **self.request_kwargs,
            )
            async for chunk in async_stream:
                yield chunk
        except RateLimitError as e:
            log_error(f"Rate limit error from OpenAI API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except APIConnectionError as e:
            log_error(f"API connection error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except APIStatusError as e:
            log_error(f"API status error from OpenAI API: {e}")
            try:
                error_message = e.response.json().get("error", {})
            except Exception:
                error_message = e.response.text
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from e
        except Exception as e:
            log_error(f"Error from OpenAI API: {e}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

    # Override base method
    @staticmethod
    def parse_tool_calls(tool_calls_data: List[ChoiceDeltaToolCall]) -> List[Dict[str, Any]]:
        """
        Build tool calls from streamed tool call data.

        Args:
            tool_calls_data (List[ChoiceDeltaToolCall]): The tool call data to build from.

        Returns:
            List[Dict[str, Any]]: The built tool calls.
        """
        tool_calls: List[Dict[str, Any]] = []
        for _tool_call in tool_calls_data:
            _index = _tool_call.index or 0
            _tool_call_id = _tool_call.id
            _tool_call_type = _tool_call.type
            _function_name = _tool_call.function.name if _tool_call.function else None
            _function_arguments = _tool_call.function.arguments if _tool_call.function else None

            if len(tool_calls) <= _index:
                tool_calls.extend([{}] * (_index - len(tool_calls) + 1))
            tool_call_entry = tool_calls[_index]
            if not tool_call_entry:
                tool_call_entry["id"] = _tool_call_id
                tool_call_entry["type"] = _tool_call_type
                tool_call_entry["function"] = {
                    "name": _function_name or "",
                    "arguments": _function_arguments or "",
                }
            else:
                if _function_name:
                    tool_call_entry["function"]["name"] += _function_name
                if _function_arguments:
                    tool_call_entry["function"]["arguments"] += _function_arguments
                if _tool_call_id:
                    tool_call_entry["id"] = _tool_call_id
                if _tool_call_type:
                    tool_call_entry["type"] = _tool_call_type
        return tool_calls

    def parse_provider_response(self, response: Union[ChatCompletion, ParsedChatCompletion]) -> ModelResponse:
        """
        Parse the OpenAI response into a ModelResponse.

        Args:
            response: Response from invoke() method

        Returns:
            ModelResponse: Parsed response data
        """
        model_response = ModelResponse()

        if hasattr(response, "error") and response.error:
            raise ModelProviderError(
                message=response.error.get("message", "Unknown model error"),
                model_name=self.name,
                model_id=self.id,
            )

        # Get response message
        response_message = response.choices[0].message

        # Parse structured outputs if enabled
        try:
            if (
                self.response_format is not None
                and self.structured_outputs
                and issubclass(self.response_format, BaseModel)
            ):
                parsed_object = response_message.parsed  # type: ignore
                if parsed_object is not None:
                    model_response.parsed = parsed_object
        except Exception as e:
            log_warning(f"Error retrieving structured outputs: {e}")

        # Add role
        if response_message.role is not None:
            model_response.role = response_message.role

        # Add content
        if response_message.content is not None:
            model_response.content = response_message.content

        # Add tool calls
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            try:
                model_response.tool_calls = [t.model_dump() for t in response_message.tool_calls]
            except Exception as e:
                log_warning(f"Error processing tool calls: {e}")

        # Add audio transcript to content if available
        response_audio: Optional[ChatCompletionAudio] = response_message.audio
        if response_audio and response_audio.transcript and not model_response.content:
            model_response.content = response_audio.transcript

        # Add audio if present
        if hasattr(response_message, "audio") and response_message.audio is not None:
            # If the audio output modality is requested, we can extract an audio response
            try:
                if isinstance(response_message.audio, dict):
                    model_response.audio = AudioResponse(
                        id=response_message.audio.get("id"),
                        content=response_message.audio.get("data"),
                        expires_at=response_message.audio.get("expires_at"),
                        transcript=response_message.audio.get("transcript"),
                    )
                else:
                    model_response.audio = AudioResponse(
                        id=response_message.audio.id,
                        content=response_message.audio.data,
                        expires_at=response_message.audio.expires_at,
                        transcript=response_message.audio.transcript,
                    )
            except Exception as e:
                log_warning(f"Error processing audio: {e}")

        if hasattr(response_message, "reasoning_content") and response_message.reasoning_content is not None:
            model_response.reasoning_content = response_message.reasoning_content

        if response.usage is not None:
            model_response.response_usage = response.usage

        return model_response

    def parse_provider_response_delta(self, response_delta: ChatCompletionChunk) -> ModelResponse:
        """
        Parse the OpenAI streaming response into a ModelResponse.

        Args:
            response_delta: Raw response chunk from OpenAI

        Returns:
            ModelResponse: Parsed response data
        """
        model_response = ModelResponse()
        if response_delta.choices and len(response_delta.choices) > 0:
            delta: ChoiceDelta = response_delta.choices[0].delta

            # Add content
            if delta.content is not None:
                model_response.content = delta.content

            # Add tool calls
            if delta.tool_calls is not None:
                model_response.tool_calls = delta.tool_calls  # type: ignore

            # Add audio if present
            if hasattr(delta, "audio") and delta.audio is not None:
                try:
                    if isinstance(delta.audio, dict):
                        model_response.audio = AudioResponse(
                            id=delta.audio.get("id"),
                            content=delta.audio.get("data"),
                            expires_at=delta.audio.get("expires_at"),
                            transcript=delta.audio.get("transcript"),
                            sample_rate=24000,
                            mime_type="pcm16",
                        )
                    else:
                        model_response.audio = AudioResponse(
                            id=delta.audio.id,
                            content=delta.audio.data,
                            expires_at=delta.audio.expires_at,
                            transcript=delta.audio.transcript,
                            sample_rate=24000,
                            mime_type="pcm16",
                        )
                except Exception as e:
                    log_warning(f"Error processing audio: {e}")

        # Add usage metrics if present
        if response_delta.usage is not None:
            model_response.response_usage = response_delta.usage

        return model_response

@dataclass
class DashScopeCompatChat(OpenAIChat):
    """通义千问系列模型的基类，处理兼容API的特殊要求"""
    
    def _get_client_params(self) -> Dict[str, Any]:
            """确保使用子类设置的API密钥，不依赖环境变量"""
            # 直接使用已设置的api_key，不检查环境变量
            client_params = {
                "api_key": self.api_key,
                "base_url": self.base_url,
            }
            
            # 添加其他非空参数
            for param in ["organization", "timeout", "max_retries", "default_headers", "default_query"]:
                if getattr(self, param) is not None:
                    client_params[param] = getattr(self, param)
                    
            # 添加其他客户端参数
            if self.client_params:
                client_params.update(self.client_params)
                
            return client_params
    
    def invoke(self, messages: List[Message]) -> Union[ChatCompletion, ParsedChatCompletion]:
        """强制通过流式API处理请求，因为通义千问模型只支持流式输出"""
        try:
            # 准备请求参数
            kwargs = {k: v for k, v in self.request_kwargs.items() if k != "stream"}
            
            # 对于结构化输出的特殊处理
            if self.response_format is not None and self.structured_outputs:
                # 添加JSON关键词要求到消息中
                messages = self._add_json_keyword_to_messages(messages)
                
                # 确保使用正确的response_format
                if "response_format" in kwargs and isinstance(kwargs["response_format"], type):
                    kwargs["response_format"] = {"type": "json_object"}
            
            # 使用流式API获取完整响应
            full_response = ""
            for chunk in self.get_client().chat.completions.create(
                model=self.id,
                messages=[self._format_message(m) for m in messages],
                stream=True,  # 强制使用流式输出
                **kwargs
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            # 处理结构化输出
            if self.response_format is not None and self.structured_outputs:
                return self._parse_structured_response(full_response)
            
            # 构建普通响应
            return self._create_mock_completion(full_response)
            
        except Exception as e:
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
    
    def _add_json_keyword_to_messages(self, messages: List[Message]) -> List[Message]:
        """确保消息中包含'json'关键词，以满足通义千问结构化输出的要求"""
        # 创建消息副本以避免修改原始消息
        messages_copy = []
        has_json_keyword = False
        has_system_message = False
        
        for msg in messages:
            msg_copy = Message(
                role=msg.role,
                content=msg.content,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
                tool_calls=msg.tool_calls
            )
            
            if msg.role == "system":
                has_system_message = True
                # 检查是否已包含json关键词
                if msg.content and "json" in msg.content.lower():
                    has_json_keyword = True
                else:
                    # 添加json关键词
                    msg_copy.content = (msg.content or "") + "\n请以JSON格式返回响应。"
                    has_json_keyword = True
            
            messages_copy.append(msg_copy)
        
        # 如果没有系统消息，添加一个包含json关键词的系统消息
        if not has_system_message:
            messages_copy.insert(0, Message(role="system", content="请以JSON格式返回响应。"))
            has_json_keyword = True
        
        # 如果仍然没有json关键词，将其添加到最后一条用户消息中
        if not has_json_keyword:
            for i in range(len(messages_copy) - 1, -1, -1):
                if messages_copy[i].role == "user":
                    messages_copy[i].content = (messages_copy[i].content or "") + "\n请以JSON格式返回。"
                    break
        
        return messages_copy
    
    def _parse_structured_response(self, response_text: str) -> ParsedChatCompletion:
        """从文本响应中解析结构化输出"""
        import json
        import re
        import time
        import uuid
        
        # 尝试提取JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有代码块，尝试直接解析整个响应
            json_str = response_text
        
        # 清理并解析JSON
        try:
            json_data = json.loads(json_str)
        except Exception:
            # 尝试修复常见的JSON格式问题
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_data = json.loads(json_str)
        
        # 简化复杂响应
        json_data = self._simplify_response(json_data)
        
        # 创建模型实例
        model_instance = self.response_format(**json_data)
        
        # 构建模拟的ParsedChatCompletion
        return ParsedChatCompletion(
            id=f"mock-{uuid.uuid4()}",
            created=int(time.time()),
            model=self.id,
            object="chat.completion",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(json_data),
                    "parsed": model_instance
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
    
    def _simplify_response(self, data: Dict) -> Dict:
        """将复杂的嵌套JSON转换为简单格式"""
        # 现有实现或简化版本...
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                # 尝试从嵌套字典中提取有用信息
                for subkey in ["value", "text", "name", "description", "primary"]:
                    if subkey in value:
                        result[key] = value[subkey]
                        break
                else:
                    result[key] = str(value)
            elif isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    result[key] = value
                else:
                    result[key] = [str(item) for item in value]
            else:
                result[key] = value
                
        return result
    
    def _create_mock_completion(self, content: str) -> ChatCompletion:
        """创建模拟的非结构化ChatCompletion"""
        import time
        import uuid
        
        return ChatCompletion(
            id=f"mock-{uuid.uuid4()}",
            created=int(time.time()),
            model=self.id,
            object="chat.completion",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

# add by xuy 20250411
@dataclass
class QWQChat(DashScopeCompatChat):
    """
    A pre-configured OpenAIChat class for using the QWQ-32B model through DashScope compatible API.
    
    This model is pre-configured to use the QWQ-32B model via the Alibaba Cloud DashScope
    compatible API endpoint.
    """
    
    def __init__(
        self,
        id: str = "qwq-32b",
        name: str = "QWQChat",
        provider: str = "Alibaba DashScope",
        **kwargs
    ):
        super().__init__(
            id=id,
            name=name,
            provider=provider,
            **kwargs
        )
        # Set up pre-configured settings
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = 'sk-df66e1c0892143e7a610c5c20db08d0e'
        # 修正角色映射
        self.role_map = {
            "system": "system",  # 改为标准角色
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        }
        

@dataclass
class QwenMaxChat20250125(OpenAIChat):
    """
    A pre-configured OpenAIChat class for using the QwenMax model through DashScope compatible API.
    """
    
    def __init__(
        self,
        id: str = "qwen-max-2025-01-25",
        name: str = "QwenMaxChat20250125",
        provider: str = "Alibaba DashScope",
        **kwargs
    ):
        super().__init__(
            id=id,
            name=name,
            provider=provider,
            **kwargs
        )
        # Set up pre-configured settings
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = 'sk-df66e1c0892143e7a610c5c20db08d0e'
        # 修正角色映射
        self.role_map = {
            "system": "system",  # 改为标准角色
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        }

@dataclass
class QwenV1MaxChat(OpenAIChat):
    """
    A pre-configured OpenAIChat class for using the QwenMax model through DashScope compatible API.
    """
    
    def __init__(
        self,
        id: str = "qwen-vl-max",
        name: str = "QwenV1MaxChat",
        provider: str = "Alibaba DashScope",
        **kwargs
    ):
        super().__init__(
            id=id,
            name=name,
            provider=provider,
            **kwargs
        )
        # Set up pre-configured settings
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = 'sk-df66e1c0892143e7a610c5c20db08d0e'
        # 修正角色映射
        self.role_map = {
            "system": "system",  # 改为标准角色
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        }


@dataclass
class Qwen2Dot5Omni7bChat(OpenAIChat):
    """
    A mixed model for using the QwenOmniTurbo model through DashScope compatible API.
    """
    
    def __init__(
        self,
        id: str = "qwen2.5-omni-7b",
        name: str = "Qwen2Dot5Omni7bChat",
        provider: str = "Alibaba DashScope",
        **kwargs
    ):
        super().__init__(
            id=id,
            name=name,
            provider=provider,
            **kwargs
        )
        # Set up pre-configured settings
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = 'sk-df66e1c0892143e7a610c5c20db08d0e'
        # 修正角色映射
        self.role_map = {
            "system": "system",  # 改为标准角色
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        }

@dataclass
class QwenOmniTurboChat(OpenAIChat):
    """
    A mixed model for using the QwenOmniTurbo model through DashScope compatible API.
    """
    
    def __init__(
        self,
        id: str = "qwen-omni-turbo",
        name: str = "QwenOmniTurboChat",
        provider: str = "Alibaba DashScope",
        **kwargs
    ):
        super().__init__(
            id=id,
            name=name,
            provider=provider,
            **kwargs
        )
        # Set up pre-configured settings
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = 'sk-df66e1c0892143e7a610c5c20db08d0e'
        # 修正角色映射
        self.role_map = {
            "system": "system",  # 改为标准角色
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        }

import json
import time
import uuid
from typing import Any, Type

from pydantic import ValidationError

@dataclass
class DeepSeekV3Chat(OpenAIChat):
    """
    A pre-configured OpenAIChat class for using the QWQ-32B model through DashScope compatible API.
    
    This model is pre-configured to use the QWQ-32B model via the Alibaba Cloud DashScope
    compatible API endpoint.
    """
    
    def __init__(
        self,
        id: str = "deepseek-v3",
        name: str = "DeepSeekChat",
        provider: str = "Alibaba DashScope",
        **kwargs
    ):
        super().__init__(
            id=id,
            name=name,
            provider=provider,
            **kwargs
        )
        # Set up pre-configured settings
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = 'sk-df66e1c0892143e7a610c5c20db08d0e'
        # 修正角色映射
        self.role_map = {
            "system": "system",  # 改为标准角色
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "model": "assistant",
        }

    # def invoke(self, messages: List[Message]) -> Union[ChatCompletion, ParsedChatCompletion]:
    #     # 如果需要结构化输出，注入明确的JSON格式指导
    #     if self.response_format is not None and self.structured_outputs:
    #         # 备份原请求参数并移除可能的 stream 参数
    #         kwargs = {k: v for k, v in self.request_kwargs.items() if k != "stream"}
            
    #         # 将结构化描述添加到系统消息中
    #         has_system_message = False
    #         for i, msg in enumerate(messages):
    #             if msg.role == "system":
    #                 has_system_message = True
    #                 # 添加明确格式要求
    #                 format_instructions = """
    #                 请严格按照以下规则格式化您的响应：
    #                 1. 所有字段必须是简单的字符串，不要使用嵌套对象
    #                 2. 必须提供所有必需字段(name, storyline, ending)
    #                 3. characters数组中的每一项必须是简单字符串
    #                 4. 以纯文本格式返回，不要使用Markdown
    #                 """
    #                 messages[i].content = (msg.content or "") + format_instructions
    #                 break
            
    #         if not has_system_message:
    #             # 如果没有系统消息，添加一个
    #             messages.insert(0, Message(role="system", content="您需要以JSON格式回复，且所有字段必须是简单字符串，不要使用嵌套对象。"))
            
    #         # 使用流式API获取完整响应
    #         try:
    #             full_response = ""
    #             for chunk in self.get_client().chat.completions.create(
    #                 model=self.id,
    #                 messages=[self._format_message(m) for m in messages],
    #                 stream=True,  # 只在这里设置流参数
    #                 **kwargs
    #             ):
    #                 if chunk.choices and chunk.choices[0].delta.content:
    #                     full_response += chunk.choices[0].delta.content
                
    #             # 从响应文本中提取JSON
    #             json_text = self._extract_json(full_response)
    #             json_data = json.loads(json_text)
                
    #             # 转换格式为标准格式
    #             json_data = self._simplify_response(json_data)
                
    #             # 创建模拟的完整响应
    #             return self._create_mock_completion(json_data, self.response_format)
    #         except Exception as e:
    #             raise ModelProviderError(f"解析结构化输出失败: {str(e)}", model_name=self.name, model_id=self.id)
        
    #     # 非结构化输出走标准流程
    #     return super().invoke(messages)
    
    # def _extract_json(self, text: str) -> str:
    #     """提取JSON文本，处理可能的非JSON前缀/后缀"""
    #     # 寻找JSON边界
    #     json_start = text.find("{")
    #     json_end = text.rfind("}")
        
    #     if json_start == -1 or json_end == -1:
    #         raise ValueError("无法找到有效的JSON对象")
            
    #     return text[json_start:json_end+1]
    
    # def _simplify_response(self, data: Dict) -> Dict:
    #     """将复杂的嵌套JSON转换为简单格式"""
    #     result = {}
        
    #     # 处理缺失的必填字段
    #     if "title" in data and "name" not in data:
    #         result["name"] = data["title"]
        
    #     if "storyline" not in data and "plot" in data:
    #         result["storyline"] = data["plot"]
            
    #     if "ending" not in data:
    #         result["ending"] = "结局待定"
            
    #     # 转换设置
    #     if "setting" in data:
    #         if isinstance(data["setting"], dict):
    #             setting = data["setting"]
    #             if "location" in setting:
    #                 result["setting"] = f"{setting['location']}"
    #                 if "description" in setting:
    #                     result["setting"] += f": {setting['description']}"
    #         else:
    #             result["setting"] = data["setting"]
                
    #     # 转换类型
    #     if "genre" in data:
    #         if isinstance(data["genre"], dict):
    #             genre = data["genre"]
    #             result["genre"] = genre.get("primary", "")
    #         else:
    #             result["genre"] = data["genre"]
                
    #     # 处理角色
    #     if "characters" in data:
    #         characters = []
    #         for char in data["characters"]:
    #             if isinstance(char, dict):
    #                 name = char.get("name", "")
    #                 role = char.get("role", "")
    #                 chars = f"{name}"
    #                 if role:
    #                     chars += f" ({role})"
    #                 characters.append(chars)
    #             else:
    #                 characters.append(char)
    #         result["characters"] = characters
            
    #     # 拷贝其他简单字段
    #     for key, value in data.items():
    #         if key not in result and isinstance(value, (str, int, float, bool)):
    #             result[key] = value
        
    #     return result
    
    # def _create_mock_completion(self, data: Dict, model_class: Type[BaseModel]) -> ParsedChatCompletion:
    #     """创建模拟的ParsedChatCompletion对象"""
    #     # 创建Pydantic模型实例
    #     try:
    #         model_instance = model_class(**data)
    #     except ValidationError as e:
    #         raise ModelProviderError(f"创建结构化输出失败: {str(e)}", model_name=self.name, model_id=self.id)
            
    #     # 模拟ParsedChatCompletion的结构
    #     completion = ParsedChatCompletion(
    #         id=f"mock-{uuid.uuid4()}",
    #         object="chat.completion",
    #         created=int(time.time()),
    #         model=self.id,
    #         choices=[{
    #             "index": 0,
    #             "message": {
    #                 "role": "assistant",
    #                 "content": json.dumps(data),
    #                 "function_call": None,
    #                 "tool_calls": None,
    #                 "parsed": model_instance
    #             },
    #             "finish_reason": "stop"
    #         }],
    #         usage={
    #             "prompt_tokens": 0,
    #             "completion_tokens": 0,
    #             "total_tokens": 0
    #         }
    #     )
        
    #     return completion