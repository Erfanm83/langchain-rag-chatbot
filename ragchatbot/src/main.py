from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, Header, status
from fastapi.requests import Request
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from src.security import is_rate_limited

# One-time valid tokens
one_time_tokens = set()

app = FastAPI(
    title="SalesChatBot API",
    summary="This API receives question as a json and returns an answer based on its knowledge base trained before.",
    version="1.1.2",
    contact={
        "name": "Erfan Mahmoudi",
        "email": "e.mahmoudi@greenweb.ir",
    },
)

# Simple in-memory rate limiter
rate_limit_data: Dict[str, list] = {}

# Request Models
class GetTokenRequest(BaseModel):
    secret: str = Field(
        ...,
        title="secret key for test in dev mode",
        description="کلید جهت استفاده از API",
        example="tir-ligufhwli-ughfwieuy-lguewiyg"
    )

# Response Models
class TokenResponse(BaseModel):
    token: str = Field(
        ...,
        title="Generated token",
        description="توکن یکبار مصرف تولیدشده",
        example="550e8400-e29b-41d4-a716-446655440000"
    )

class BaseErrorResponse(BaseModel):
    detail: str = Field(..., example="خطایی غیر منتظره رخ داده است")

class UnauthorizedResponse(BaseErrorResponse):
    detail: str = Field(..., example=".توکن وارد نشده است")

class ForbiddenResponse(BaseErrorResponse):
    detail: str = Field(..., example=".توکن وارد شده معتبر نیست")

class ValidationResponse(BaseErrorResponse):
    detail: str = Field(..., example="فرمت ورودی نامعتبر است")

class RateLimitResponse(BaseErrorResponse):
    detail: str = Field(..., example="تعداد درخواست‌ها بیش از حد مجاز است.")

class InternalServerErrorResponse(BaseErrorResponse):
    detail: str = Field(..., example="خطایی سمت سرور رخ داده است")

@app.post(
    "/get_token/",
    summary="Generate a one-time-use token",
    responses={
        401: {"model": UnauthorizedResponse, "description": "When token is empty"},
        403: {"model": ForbiddenResponse, "description": "When token is invalid"},
        422: {"model": ValidationResponse, "description": "input format validation error"},
        429: {"model": RateLimitResponse, "description": "When rate limit is exceeded"},
        500: {"model": InternalServerErrorResponse, "description": "internal server error"},
    },
    response_model=TokenResponse
)
async def get_token(
    request: GetTokenRequest,
    client_request: Request
):  
    if not request.secret or request.secret == "":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"msg": "فیلد کلید نمیتواند خالی باشد"}
        )
    
    client_ip = client_request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"msg": "تعداد درخواست‌ها بیش از حد مجاز است"}
        )

    if request.secret != TOKEN_RETRIEVAL_SECRET:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"msg": "توکن نامعتبر یا منقضی شده است"}
        )

    new_token = str(uuid.uuid4())
    one_time_tokens.add(new_token)
    return {"token": new_token}

# @app.post("/documents/index")
# async def index(token: str = Depends(oauth2_scheme)):
#     conn = sqlite3.connect("db.sqlite")
#     docs = conn.execute("SELECT id, text FROM raw_documents WHERE indexed=0").fetchall()
#     texts = [d[1] for d in docs]
#     db = Chroma.from_texts(texts, OpenAIEmbeddings(), persist_directory="chroma_db")
#     db.persist()
#     conn.executemany("UPDATE raw_documents SET indexed=1 WHERE id=?", [(d[0],) for d in docs])
#     conn.commit()
#     return {"status": "indexed", "count": len(docs)}

# --- Query ---
# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = 5

# @app.post("/query")
# async def query(req: QueryRequest, token: str = Depends(oauth2_scheme)):
#     db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
#     qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": req.top_k}))
#     answer = qa.run(req.question)
#     # Call Goftino
#     async with httpx.AsyncClient() as client:
#         await client.post("https://api.goftino.com/track", json={"q": req.question, "a": answer}, headers={"Authorization": f"Bearer {GOFTINO_API_KEY}"})
#     return {"answer": answer}
