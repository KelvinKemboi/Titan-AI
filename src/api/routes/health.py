from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.deps import get_db

router = APIRouter()


@router.get("/health")
def health(db: Session = Depends(get_db)):
    """Liveness & DB connectivity check. 200 only if Postgres actually
    responds to a query, not just if the process is up."""
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "db": "unreachable", "error": str(exc)},
        ) from exc

    return {"status": "ok", "db": "ok"}
