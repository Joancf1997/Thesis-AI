from models.user import User
from pydantic import BaseModel
from fastapi import HTTPException
from sqlalchemy.orm import Session


def load_users(
    db: Session
):
  users = db.query(User).order_by(User.created_at.desc()).all()
  return [
      {
          "id": str(u.id),
          "username": u.username,
          "created_at": u.created_at
      }
      for u in users
  ]


class CreateUserRequest(BaseModel):
    username: str
    password: str
def create_user(
    db: Session, 
    req: CreateUserRequest
  ):
    print(req)
    from werkzeug.security import generate_password_hash
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(
        username=req.username,
        password_hash=generate_password_hash(req.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "id": str(user.id),
        "username": user.username,
        "created_at": user.created_at
    }
