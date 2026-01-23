from abc import ABC, abstractmethod
from typing import Optional, Type
from pydantic import BaseModel as PydanticModel

class BaseModel(ABC):
    model_name: str

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        response_model: Optional[Type[PydanticModel]] = None,
    ):
        """
        If response_model is provided, return a validated Pydantic object.
        Otherwise, return raw text.
        """
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict],
        response_model: Optional[Type[PydanticModel]] = None,
    ):
        """
        Asynchronous generation.
        If response_model is provided, return a validated Pydantic object.
        Otherwise, return raw text.
        """
        pass