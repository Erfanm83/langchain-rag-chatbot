from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.requests import Request
from src.security import is_rate_limited
from config.config import TOKEN_RETRIEVAL_SECRET
from src.rag import generate_answer
import uuid

# One-time valid tokens
one_time_tokens = set()

app = FastAPI(
    title="SalesChatBot API",
    summary="This API receives question as a json and returns an answer based on its knowledge base trained before.",
    version="1.0.0",
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

class ChatRequest(BaseModel):
    question: str = Field(..., example="اسمت چیه ؟?")
    token: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000")

class ChatResponse(BaseModel):
    answer: str = Field(..., example="من سروریار هستم دستیار فروش ایران سرور")

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

@app.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            403: {"model": ForbiddenResponse, "description": "When token is invalid"},
            422: {"model": ValidationResponse, "description": "Input format validation error"},
            500: {"model": InternalServerErrorResponse, "description": "Internal server error"},
        }
        )
async def chat(request: ChatRequest):
    if request.token not in one_time_tokens:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="توکن نامعتبر یا منقضی شده است"
        )
    # Remove token after use
    one_time_tokens.remove(request.token)

    try:
        # Call the generate_answer function from rag.py
        answer = generate_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطایی در پردازش سوال رخ داده است: {str(e)}"
        )