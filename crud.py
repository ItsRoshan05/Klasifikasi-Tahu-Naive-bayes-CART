from sqlalchemy.orm import Session
import models
import schemas
from utils import hash_password
from sqlalchemy import func

def create_user(db: Session, user: schemas.UserCreate):
    if user.password != user.repeat_password:
        raise ValueError("Passwords do not match")
    hashed_password = hash_password(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_users(db: Session):
    return db.query(models.User).all()


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        return None
    if user_update.username:
        db_user.username = user_update.username
    if user_update.email:
        db_user.email = user_update.email
    if user_update.password:
        db_user.hashed_password = hash_password(user_update.password)
    db.commit()
    db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user

def create_prediction(db: Session, prediction: schemas.PredictionCreate):
    print("Saving prediction data to database:")
    print(prediction.dict())  # Debug output
    db_prediction = models.Prediction(
        produk_tahu=prediction.produk_tahu,
        aroma=prediction.aroma,
        tekstur=prediction.tekstur,
        cita_rasa=prediction.cita_rasa,
        masa_kadaluarsa=prediction.masa_kadaluarsa,
        prediction_nb=prediction.prediction_nb,
        score_nb=prediction.score_nb,
        prediction_cart=prediction.prediction_cart,
        score_cart=prediction.score_cart
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


def get_user_count(db: Session):
    return db.query(func.count(models.User.id)).scalar()

def get_prediction_data_count(db: Session):
    return db.query(func.count(models.Prediction.id)).scalar()

def get_growth_data(db: Session):
    result = (
        db.query(
            func.date_format(models.Prediction.created_at, '%Y-%m').label('month'),
            func.count(models.Prediction.id).label('count')
        )
        .group_by('month')
        .order_by('month')
        .all()
    )
    return [{"month": r.month, "count": r.count} for r in result]


def get_pie_chart_data(db: Session):
    result = (
        db.query(
            models.Prediction.prediction_nb.label('category'),
            func.count(models.Prediction.id).label('count')
        )
        .group_by('category')
        .all()
    )
    return [{"category": r.category, "count": r.count} for r in result]


def get_predictions(db: Session):
    return db.query(models.Prediction).all()

def delete_prediction(db: Session, prediction_id: int):
    db_prediction = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if db_prediction:
        db.delete(db_prediction)
        db.commit()
    return db_prediction
