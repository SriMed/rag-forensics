from fastapi import APIRouter
from models import ExampleRequest, ExampleResponse

router = APIRouter()


@router.post("/example", response_model=ExampleResponse)
def get_example(request: ExampleRequest) -> ExampleResponse:
    return ExampleResponse(
        example_id=f"{request.domain}-001",
        question="What is the capital of France?",
        context_preview="France is a country in Western Europe. Its capital city is Paris.",
    )
