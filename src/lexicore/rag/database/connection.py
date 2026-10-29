from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from lexicore.configrations.config import settings

engine = create_engine(settings.DATABASE_URL,pool_pre_ping=True)

session_maker = sessionmaker(bind=engine , autoflush=False, autocommit=False)

def get_db():
    db = session_maker()
    try:
        yield db
    finally:
        db.close()
