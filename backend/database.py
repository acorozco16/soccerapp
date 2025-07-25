from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import os
from pathlib import Path

# Database configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR}/soccer_analysis.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DATABASE_URL}")