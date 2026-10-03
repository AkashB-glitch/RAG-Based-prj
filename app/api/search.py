from fastapi import APIRouter, Depends

from app.auth import get_auth_context
from app.principals import AuthContext
from app.retrieval.scoring import SearchRequest, SearchResponse
from app.retrieval.service import retrieval_service

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    ctx: AuthContext = Depends(get_auth_context),
) -> SearchResponse:
    """Execute permission-aware hybrid search over authorized document chunks.

    Returns only chunks visible to the authenticated user under Postgres RLS.
    """
    return await retrieval_service.search(request=request, ctx=ctx)
