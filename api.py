# app.py
import os
import hashlib
from datetime import datetime
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from sqlalchemy import text, Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import update as sql_update, delete as sql_delete

# pgvector support - только для создания колонки с размерностью
from pgvector.sqlalchemy import Vector


# -------------------- ML SETUP --------------------

print("Загрузка модели sentence-transformers all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Модель успешно загружена. Размерность эмбеддинга: {len(embedding_model.encode('test'))}")

def generate_embedding(text: str) -> list[float]:
    return embedding_model.encode(text).tolist()

def format_vector(embedding: list[float]) -> str:
    # PostgreSQL ожидает вектор в формате '[0.1,0.2,0.3]'
    return '[' + ','.join(map(str, embedding)) + ']'

def hash_author(author: str) -> str:
    return hashlib.sha256(author.encode("utf-8")).hexdigest()


# -------------------- DATABASE SETUP --------------------

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

engine = create_async_engine(
    DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


# -------------------- MODELS --------------------

class Idea(Base):
    __tablename__ = "ideas"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    author_hash = Column(String(64), nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class IdeaVector(Base):
    __tablename__ = "idea_vectors"

    id = Column(
        Integer,
        ForeignKey("ideas.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(384), nullable=False)  # Для правильного CREATE TABLE ... VECTOR(384)


# -------------------- FASTAPI --------------------

app = FastAPI(title="Ideas Archive API")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# Автоматическое создание таблиц + активация pgvector при старте
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    print("pgvector активирован, таблицы проверены/созданы")


# -------------------- SCHEMAS --------------------

class IdeaCreate(BaseModel):
    title: str
    author: str
    body: str


class IdeaUpdate(BaseModel):
    title: Optional[str] = None
    author: str
    body: Optional[str] = None


class IdeaResponse(BaseModel):
    id: int
    title: str
    author_hash: str
    body: str
    created_at: datetime
    updated_at: datetime


class SearchRequest(BaseModel):
    query: str
    top_n: int = 5


# -------------------- ENDPOINTS --------------------

@app.post("/api/v1/ideas", response_model=IdeaResponse)
async def create_idea(
    idea: IdeaCreate,
    db: AsyncSession = Depends(get_db),
):
    author_hash = hash_author(idea.author)
    embedding_list = generate_embedding(f"{idea.title} {idea.body}")
    embedding_str = format_vector(embedding_list)

    async with db.begin():
        res = await db.execute(
            text("""
                INSERT INTO ideas (title, author_hash, body)
                VALUES (:title, :author_hash, :body)
                RETURNING id, title, author_hash, body, created_at, updated_at
            """),
            {
                "title": idea.title,
                "author_hash": author_hash,
                "body": idea.body,
            },
        )
        row = res.fetchone()

        await db.execute(
            text("""
                INSERT INTO idea_vectors (id, embedding)
                VALUES (:id, :embedding)
            """),
            {"id": row.id, "embedding": embedding_str},
        )

    return IdeaResponse(**row._mapping)


@app.post("/api/v1/search", response_model=List[IdeaResponse])
async def search_ideas(
    search: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    embedding_list = generate_embedding(search.query)
    embedding_str = format_vector(embedding_list)

    res = await db.execute(
        text("""
            SELECT i.id, i.title, i.author_hash, i.body, i.created_at, i.updated_at
            FROM idea_vectors v
            JOIN ideas i ON i.id = v.id
            ORDER BY v.embedding <=> :embedding
            LIMIT :top_n
        """),
        {"embedding": embedding_str, "top_n": search.top_n},
    )

    return [IdeaResponse(**r._mapping) for r in res.fetchall()]


@app.get("/api/v1/ideas", response_model=List[IdeaResponse])
async def list_ideas(
    page: int = 1,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        text("""
            SELECT id, title, author_hash, body, created_at, updated_at
            FROM ideas
            ORDER BY created_at DESC
            OFFSET :offset LIMIT :limit
        """),
        {"offset": (page - 1) * limit, "limit": limit},
    )

    return [IdeaResponse(**r._mapping) for r in res.fetchall()]


@app.put("/api/v1/ideas/{id}", response_model=IdeaResponse)
async def update_idea(
    id: int,
    data: IdeaUpdate,
    db: AsyncSession = Depends(get_db),
):
    stored = await db.execute(
        text("SELECT author_hash FROM ideas WHERE id = :id"),
        {"id": id},
    )
    if stored.scalar() != hash_author(data.author):
        raise HTTPException(403, "author mismatch")

    async with db.begin():
        res = await db.execute(
            sql_update(Idea)
            .where(Idea.id == id)
            .values(
                **{k: v for k, v in data.dict().items() if v is not None and k != "author"}
            )
            .returning(*Idea.__table__.columns)
        )
        row = res.fetchone()

        embedding_list = generate_embedding(f"{row.title} {row.body}")
        embedding_str = format_vector(embedding_list)

        await db.execute(
            text("""
                INSERT INTO idea_vectors (id, embedding)
                VALUES (:id, :embedding)
                ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding
            """),
            {"id": id, "embedding": embedding_str},
        )

    return IdeaResponse(**row._mapping)


@app.delete("/api/v1/ideas/{id}")
async def delete_idea(
    id: int,
    author: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    stored_hash = await db.scalar(
        text("SELECT author_hash FROM ideas WHERE id = :id"),
        {"id": id}
    )
    if stored_hash != hash_author(author):
        raise HTTPException(403, "author mismatch")

    await db.execute(sql_delete(Idea).where(Idea.id == id))
    await db.commit()
    return {"detail": "deleted"}


# -------------------- RUN --------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)