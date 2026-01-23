from typing import Optional, Type
from pydantic import BaseModel as PydanticModel
import anthropic

from .base_model import BaseModel


class AnthropicModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

    @staticmethod
    def extract_system_prompt(messages: list[dict]) -> str:
        system_prompts = [
            m["content"] for m in messages if m.get("role") == "system"
        ]

        return "\n".join(system_prompts)
    @staticmethod
    def extract_user_messages(messages: list[dict]) -> list[dict]:
        return [
            m for m in messages if m.get("role") == "user"
        ]


    def generate(
        self,
        messages: list[dict],
        response_model: Optional[Type[PydanticModel]] = None,
    ):
        """
        If response_model is provided, enforce structured output.
        Otherwise return raw text.
        """

        system_prompts = self.extract_system_prompt(messages)
        user_message = self.extract_user_messages(messages)


        if response_model is None:
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompts,
                messages=user_message,
                max_tokens=1048
            )

            return response.content

        response = self.client.beta.messages.parse(
            model=self.model_name,
            betas=["structured-outputs-2025-11-13"],
            system=system_prompts,
            messages=user_message,
            output_format=response_model,
            max_tokens=1048
        )


        return response.parsed_output

    async def agenerate(
        self,
        messages: list[dict],
        response_model: Optional[Type[PydanticModel]] = None,
    ):
        """
        If response_model is provided, enforce structured output.
        Otherwise return raw text.
        """

        system_prompts = self.extract_system_prompt(messages)
        user_message = self.extract_user_messages(messages)


        if response_model is None:
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_prompts,
                messages=user_message,
                max_tokens=10000
            )

            return response.content

        response = await self.async_client.beta.messages.parse(
            model=self.model_name,
            betas=["structured-outputs-2025-11-13"],
            system=system_prompts,
            messages=user_message,
            output_format=response_model,
            max_tokens=10000
        )


        return response.parsed_output

