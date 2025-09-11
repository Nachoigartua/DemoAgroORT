from fastapi import Request, HTTPException
from .config import settings
from .cache import get_redis

def check_api_key(header: str | None):
    if not header or header != settings.API_KEY:
        raise HTTPException(status_code=401, detail='Unauthorized')

def rate_limit(request: Request, limit:int=60, window:int=60):
    r = get_redis()
    ip = request.headers.get('x-forwarded-for') or request.client.host
    key = f'ratelimit:{ip}:{request.url.path}'
    c = r.incr(key)
    if c == 1: r.expire(key, window)
    if c > limit: raise HTTPException(status_code=429, detail='Rate limit exceeded')
