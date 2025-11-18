from jose import jwt
from models.user import User
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

SECRET_KEY = "REPLACE_WITH_A_REAL_SECRET"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

from werkzeug.security import check_password_hash
from fastapi import HTTPException


class LoginRequest(BaseModel):
    username: str
    password: str
def login_user(
    db: Session, 
    req: LoginRequest
  ):
    user = db.query(User).filter(User.username == req.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not check_password_hash(user.password_hash, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {
          "sub": str(user.id),
          "exp": expires
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {
        "user_id": str(user.id),
        "username": user.username,
        "token": token,
        "expires_at": expires
    }

