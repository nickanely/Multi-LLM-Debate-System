import asyncio
from typing import Optional, Type
from pydantic import BaseModel as PydanticModel

from xai_sdk import Client
from xai_sdk.chat import user, system

from .base_model import BaseModel


class XAIModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.client = Client(
            api_key=api_key,
        )
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
            chat = self.client.chat.create(model=self.model_name)

            chat.append(user(user_message))

            response = chat.sample()

            return response.content

        chat = self.client.chat.create(model=self.model_name)

        chat.append(system(system_prompts))
        chat.append(user(user_message))

        response, model = chat.parse(response_model)

        return model

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
            chat = self.client.chat.create(model=self.model_name)

            chat.append(user(user_message))

            response =  await asyncio.to_thread(chat.sample)

            return response.content

        chat = self.client.chat.create(model=self.model_name)

        chat.append(system(system_prompts))
        chat.append(user(user_message))

        response, model = await asyncio.to_thread(chat.parse,response_model)

        return model
