from src.data.db import SessionLocal


def get_db():
    """FastAPI dependency yielding a request-scoped DB session, drawn from
    the shared connection pool in src/data/db.py."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
