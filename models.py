from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document: str
    page: int
    text: str


class Source(BaseModel):
    document: str
    page: int
    excerpt: str


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class Answer(BaseModel):
    text: str
    sources: list[Source]


class IndexStatus(BaseModel):
    state: str
    documents: list[str]
    chunks: int
    message: str = ""
