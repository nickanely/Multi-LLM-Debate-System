from typing import Optional, Type
from pydantic import BaseModel as PydanticModel
from openai import OpenAI, AsyncOpenAI

from .base_model import BaseModel

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

    def generate(
        self,
        messages: list[dict],
        response_model: Optional[Type[PydanticModel]] = None,
    ):
        if response_model is None:
            response = self.client.responses.parse(
                model=self.model_name,
                input=messages,
            )
            return response.output_text

        response = self.client.responses.parse(
            model=self.model_name,
            input=messages,
            text_format=response_model,
        )

        return response.output_parsed


    async def agenerate(
        self,
        messages: list[dict],
        response_model: Optional[Type[PydanticModel]] = None,
    ):
        """Asynchronous generation."""
        if response_model is None:
            response = await self.async_client.responses.parse(
                model=self.model_name,
                input=messages,
            )
            return response.output_text

        response = await self.async_client.responses.parse(
            model=self.model_name,
            input=messages,
            text_format=response_model,
        )
        return response.output_parsed