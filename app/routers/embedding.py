# app/routers/embedding.py

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from app.services.embedding_service import (
    TextEmbeddingService,
    EnhancedSemanticSearchService,
    KeywordClusteringService,
    DocumentClusteringService,
    TextSummarizationService,
)
from app.config import settings
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    texts: List[str]


class SemanticSearchRequest(BaseModel):
    query: str
    corpus: List[str]
    num_results: Optional[int] = 10


class ClusterKeywordsRequest(BaseModel):
    corpus: List[str]
    num_keywords: Optional[int] = 20
    n_clusters: Optional[int] = 5


class SummarizeTextRequest(BaseModel):
    document: str
    num_sentences: Optional[int] = 3


class SummarizeBatchRequest(BaseModel):
    documents: List[str]
    num_sentences: Optional[int] = 3


router = APIRouter()
# Initialize your embedding services
text_embedding_service = TextEmbeddingService(api_key=settings.OPENAI_API_KEY)
semantic_search_service = EnhancedSemanticSearchService(api_key=settings.OPENAI_API_KEY)
keyword_clustering_service = KeywordClusteringService(api_key=settings.OPENAI_API_KEY)
document_clustering_service = DocumentClusteringService(api_key=settings.OPENAI_API_KEY)
text_summarization_service = TextSummarizationService(api_key=settings.OPENAI_API_KEY)


@router.post("/generate_embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        embeddings = text_embedding_service.embed(request.texts)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/semantic_search")
async def semantic_search(request: SemanticSearchRequest):
    try:
        results = semantic_search_service.enhanced_search(
            request.query, request.corpus, request.num_results
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cluster_keywords")
async def cluster_keywords(request: ClusterKeywordsRequest):
    try:
        clusters = keyword_clustering_service.extract_and_cluster_keywords(
            request.corpus, request.num_keywords, request.n_clusters
        )
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cluster_documents")
async def cluster_documents(corpus: List[str], n_clusters: int = 5):
    try:
        clusters = document_clustering_service.cluster_documents(corpus, n_clusters)
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/summarize_text")
async def summarize_text(request: SummarizeTextRequest):
    try:
        summary = text_summarization_service.summarize(
            request.document, request.num_sentences
        )
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/summarize_batch")
async def summarize_batch(request: SummarizeBatchRequest):
    try:
        summaries = text_summarization_service.summarize_batch(
            request.documents, request.num_sentences
        )
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
