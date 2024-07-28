from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    repeat_password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserRead(UserBase):
    id: int

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    repeat_password: Optional[str] = None
    
class PredictionCreate(BaseModel):
    produk_tahu: str
    aroma: str
    tekstur: str
    cita_rasa: str
    masa_kadaluarsa: str
    prediction_nb: str
    score_nb: float
    prediction_cart: str
    score_cart: float

    class Config:
        from_attributes = True