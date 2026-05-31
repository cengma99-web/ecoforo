import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from ecoforo.db.models import Base


@pytest.fixture
def db_engine():
    """In-memory SQLite for tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Session bound to in-memory DB, auto-rollback after test."""
    with Session(db_engine) as session:
        yield session
        session.rollback()
