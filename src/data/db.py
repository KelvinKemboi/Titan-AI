import os

from sqlalchemy import create_engine # manages the connection pool and provides a source of database connections.
from sqlalchemy.orm import sessionmaker # generates new Session objects that are used to interact with the database.

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://titan:titan@localhost:5433/titan"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
