from fastapi import APIRouter
from models import ExampleRequest, ExampleResponse
from services.retriever import get_random_example

router = APIRouter()


@router.post("/example", response_model=ExampleResponse)
def get_example(request: ExampleRequest) -> ExampleResponse:
    example = get_random_example(request.domain)
    return ExampleResponse(
        example_id=example.example_id,
        question=example.question,
        context_preview=example.context_preview,
    )
