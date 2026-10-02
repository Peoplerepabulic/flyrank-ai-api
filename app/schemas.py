"""Strict Pydantic schemas for requests and responses."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical_issue", "account", "shipping", "general"]


class ClassifyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


class ClassifyResponse(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=500)


class KillSwitchRequest(BaseModel):
    disabled: bool


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    llm_disabled: bool
