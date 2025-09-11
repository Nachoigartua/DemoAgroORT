import psycopg2
from psycopg2.extras import RealDictCursor
from .config import settings

def get_conn():
    return psycopg2.connect(dbname=settings.POSTGRES_DB,user=settings.POSTGRES_USER,password=settings.POSTGRES_PASSWORD,host=settings.DB_HOST,port=settings.DB_PORT,cursor_factory=RealDictCursor)
