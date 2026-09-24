from fastapi import APIRouter
from . import inference

router = APIRouter()
router.include_router(inference.router, prefix="/inference", tags=["inference"])
