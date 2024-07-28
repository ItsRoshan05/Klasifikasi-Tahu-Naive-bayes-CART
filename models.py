from sqlalchemy import Column, Integer, String, Float, DateTime
from db import Base
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    auth_token = Column(String(255), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, server_default=func.now())  # Menambahkan kolom created_at

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    produk_tahu = Column(String(255))
    aroma = Column(String(255))
    tekstur = Column(String(255))
    cita_rasa = Column(String(255))
    masa_kadaluarsa = Column(String(255))
    prediction_nb = Column(Integer)
    score_nb = Column(Float)
    prediction_cart = Column(Integer)
    score_cart = Column(Float)
    created_at = Column(DateTime, server_default=func.now())  # Menambahkan kolom created_a