from fastapi import APIRouter, Depends

from app.auth import get_auth_context
from app.principals import AuthContext
from app.rag.schemas import RAGRequest, RAGResponse
from app.rag.service import rag_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse)
async def rag_query(
    request: RAGRequest,
    ctx: AuthContext = Depends(get_auth_context),
) -> RAGResponse:
    """Execute secure RAG query.

    Grounded generation over authorized document chunks with source citations and auditing.
    """
    return await rag_service.query(request=request, ctx=ctx)
