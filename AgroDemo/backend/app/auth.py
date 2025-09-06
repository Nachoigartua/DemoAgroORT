from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from fastapi import Depends

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Aquí iría la validación real con el sistema principal
    if not token.credentials or token.credentials != "test-token":
        raise HTTPException(status_code=401, detail="Token inválido")
    return {"user_info": "demo"}
