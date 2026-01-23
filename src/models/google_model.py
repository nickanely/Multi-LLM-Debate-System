import asyncio
from typing import Optional, Type
from pydantic import BaseModel as PydanticModel
from google import genai
from google.genai import types

from .base_model import BaseModel


class GoogleModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)


    @staticmethod
    def extract_system_prompt(messages: list[dict]) -> str:
        system_prompts = [
            m["content"] for m in messages if m.get("role") == "system"
        ]

        return "\n".join(system_prompts)

    @staticmethod
    def extract_user_messages(messages: list[dict]) -> str:
        user_messages  = [
            m["content"] for m in messages if m.get("role") == "user"
        ]
        return "\n".join(user_messages)


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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompts,
                )
            )
            return response.text


        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompts,
                response_mime_type="application/json",
                response_json_schema=response_model.model_json_schema()
            ),
        )

        return response_model.model_validate_json(response.text)

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
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=user_message,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompts,
                )
            )
            return response.text

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompts,
                response_mime_type="application/json",
                response_json_schema=response_model.model_json_schema()
            ),
        )

        return response_model.model_validate_json(response.text)
