from fastapi import FastAPI, Request, Depends, HTTPException, Form, Path
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import joblib
import pandas as pd
import models
import schemas
import utils
import crud
from db import SessionLocal, engine
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


app = FastAPI()

# Memuat model dan encoder
nb_model = joblib.load('model/naive_bayes_model.pkl')
cart_model = joblib.load('model/cart_model.pkl')
encoder = joblib.load('model/onehot_encoder.pkl')

# Setup template dan static files
templates = Jinja2Templates(directory="templates")
models.Base.metadata.create_all(bind=engine)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency untuk mendapatkan database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.middleware("http")
async def check_login(request: Request, call_next):
    token = request.cookies.get("auth_token")
    public_routes = ["/login", "/register", "/logout", "/static", "/favicon.ico"]
    
    # Untuk memastikan bahwa wildcard atau path static dapat diakses
    if any(request.url.path.startswith(route) for route in public_routes) or token:
        response = await call_next(request)
    else:
        response = RedirectResponse(url="/login")
    
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email)
    
    # Check if user exists and password is correct
    if not db_user or not utils.verify_password(password, db_user.hashed_password):
        # Return to login page with error message
        return RedirectResponse(url="/login?error=Email%20Atau%20Password%20Salah", status_code=303)
    
    # Generate and store token
    auth_token = utils.create_token(email)
    db_user.auth_token = auth_token
    db.commit()

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="auth_token", value=auth_token)
    return response

@app.get("/api/dashboard-data")
def get_dashboard_data(db: Session = Depends(get_db)):
    user_count = crud.get_user_count(db)
    prediction_data_count = crud.get_prediction_data_count(db)
    growth_data = crud.get_growth_data(db)
    pie_data = crud.get_pie_chart_data(db) 
    return {
        "user_count": user_count,
        "prediction_data_count": prediction_data_count,
        "growth_data": growth_data,
        "pie_data": pie_data
    }

# Halaman registrasi
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = None):
    return templates.TemplateResponse("register.html", {"request": request, "error": error})

@app.post("/register")
async def register(username: str = Form(...), email: str = Form(...), password: str = Form(...), repeat_password: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email)
    if db_user:
        return RedirectResponse(url="/register?error=Email sudah terdaftar", status_code=303)
    
    if password != repeat_password:
        return RedirectResponse(url="/register?error=Passwords do not match", status_code=303)
    
    hashed_password = utils.hash_password(password)
    new_user = models.User(username=username, email=email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    return RedirectResponse(url="/login", status_code=303)


# Halaman utama
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("auth_token")
    username = None
    
    if token:
        db_user = crud.get_user_by_email(db, token)  # Assuming token is used to retrieve user by email
        if db_user:
            username = db_user.username

    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": {},
        "prediction_nb": None,
        "score_nb": None,
        "prediction_cart": None,
        "score_cart": None,
        "username": username
    })

# Halaman testing
@app.get("/testing", response_class=HTMLResponse)
async def testing(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})

@app.get("/info-app", response_class=HTMLResponse)
async def info(request: Request):
    return templates.TemplateResponse("info_app.html", {"request": request})


@app.get("/form", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("users/user_create.html", {"request": request})


@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(request: Request, db: Session = Depends(get_db)):
    if request.method == "POST":
        form = await request.form()
        features = {
            "produk_tahu": form.get("produk_tahu"),
            "aroma": form.get("aroma"),
            "tekstur": form.get("tekstur"),
            "cita_rasa": form.get("cita_rasa"),
            "masa_kadaluarsa": form.get("masa_kadaluarsa")
        }
    elif request.method == "GET":
        features = {
            "produk_tahu": request.query_params.get("produk_tahu"),
            "aroma": request.query_params.get("aroma"),
            "tekstur": request.query_params.get("tekstur"),
            "cita_rasa": request.query_params.get("cita_rasa"),
            "masa_kadaluarsa": request.query_params.get("masa_kadaluarsa")
        }
    else:
        raise HTTPException(status_code=405, detail="Method Not Allowed")

    input_data = pd.DataFrame([features])
    
    try:
        input_encoded = encoder.transform(input_data)
        
        prediction_nb = nb_model.predict(input_encoded)
        score_nb = nb_model.predict_proba(input_encoded).max()
        
        prediction_cart = cart_model.predict(input_encoded)
        score_cart = cart_model.predict_proba(input_encoded).max()

        # Data untuk disimpan
        prediction_data = schemas.PredictionCreate(
            produk_tahu=features["produk_tahu"],
            aroma=features["aroma"],
            tekstur=features["tekstur"],
            cita_rasa=features["cita_rasa"],
            masa_kadaluarsa=features["masa_kadaluarsa"],
            prediction_nb=prediction_nb[0],
            score_nb=score_nb,
            prediction_cart=prediction_cart[0],
            score_cart=score_cart
        )
        
        # Simpan ke database
        # print("Attempting to save prediction data:")
        crud.create_prediction(db, prediction_data)

    except Exception as e:
        return templates.TemplateResponse("prediksi.html", {
            "request": request,
            "error": str(e),
            "features": features
        })

    return templates.TemplateResponse("prediksi.html", {
        "request": request,
        "prediction_nb": prediction_nb[0],
        "score_nb": score_nb,
        "prediction_cart": prediction_cart[0],
        "score_cart": score_cart,
        "features": features
    })

# Logout
@app.post("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="auth_token")
    return response

# CRUD User Endpoints
@app.get("/users/", response_class=HTMLResponse)
async def list_users(request: Request, db: Session = Depends(get_db)):
    users = crud.get_users(db)
    return templates.TemplateResponse("users/user_list.html", {"request": request, "users": users})

@app.get("/users/{user_id}", response_class=HTMLResponse)
async def view_user(request: Request, user_id: int = Path(...), db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return templates.TemplateResponse("users/user_detail.html", {"request": request, "user": user})

@app.get("/users/create", response_class=HTMLResponse)
async def create_user_form(request: Request):
    return templates.TemplateResponse("users/user_create.html", {"request": request})

@app.post("/users/")
async def create_user(username: str = Form(...), email: str = Form(...), password: str = Form(...), repeat_password: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if password != repeat_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    hashed_password = utils.hash_password(password)
    new_user = models.User(username=username, email=email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    return RedirectResponse(url="/users", status_code=303)

@app.get("/users/{user_id}/edit", response_class=HTMLResponse)
async def edit_user_form(request: Request, user_id: int = Path(...), db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return templates.TemplateResponse("users/user_update.html", {"request": request, "user": user})

@app.post("/users/{user_id}")
async def update_user(
    user_id: int,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(None),
    repeat_password: str = Form(None),
    db: Session = Depends(get_db)
):

    
    if password and repeat_password and password != repeat_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    
    user_update = schemas.UserUpdate(
        username=username,
        email=email,
        password=password,
        repeat_password=repeat_password
    )

    updated_user = crud.update_user(db, user_id, user_update)
    if updated_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return RedirectResponse(url=f"/users/{user_id}", status_code=303)

@app.post("/users/{user_id}/delete", response_class=RedirectResponse)
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    deleted_user = crud.delete_user(db, user_id)
    if deleted_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return RedirectResponse(url="/users/", status_code=303)


# Endpoint CRUD prediksi

@app.get("/predictions/", response_class=HTMLResponse)
async def list_users(request: Request, db: Session = Depends(get_db)):
    predictions = crud.get_predictions(db)
    return templates.TemplateResponse("predictions/user_list.html", {"request": request, "predictions": predictions})

@app.post("/predictions/{prediction_id}/delete", response_class=RedirectResponse)
async def delete_user(prediction_id: int, db: Session = Depends(get_db)):
    deleted_user = crud.delete_prediction(db, prediction_id)
    if deleted_user is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return RedirectResponse(url="/predictions/", status_code=303)

@app.get("/encoded-data", response_class=HTMLResponse)
async def get_encoded_data(request: Request):
    # Dummy data loading and encoding for demonstration
    data = pd.read_csv('data/revisi_data.csv')
    X = data[['Produk_Tahu', 'aroma', 'tekstur', 'cita_rasa', 'masa_kadaluarsa']]
    
    # OneHotEncoder for encoding categorical variables
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)
    columns_encoded = encoder.get_feature_names_out()
    
    # Convert the encoded data into a DataFrame
    df_encoded = pd.DataFrame(X_encoded, columns=columns_encoded)
    
    # Convert DataFrame to list of dictionaries for easier rendering in templates
    result = df_encoded.to_dict(orient='records')
    columns = df_encoded.columns.tolist()
    
    # Render the template with the encoded data
    return templates.TemplateResponse("encoded_data.html", {
        "request": request,
        "data": result,
        "columns": columns
    })



@app.get("/original-data", response_class=HTMLResponse)
async def get_original_data(request: Request):
    result = data.to_dict(orient='records')
    columns = data.columns.tolist()
    return templates.TemplateResponse("data.html", {"request": request, "data": result, "columns": columns})

@app.get("/data-naivebayes", response_class=HTMLResponse)
async def get_original_data(request: Request):
    result = data.to_dict(orient='records')
    columns = data.columns.tolist()
    return templates.TemplateResponse("datanb.html", {"request": request, "data": result, "columns": columns})

@app.get("/data-cart", response_class=HTMLResponse)
async def get_original_data(request: Request):
    result = data.to_dict(orient='records')
    columns = data.columns.tolist()
    return templates.TemplateResponse("datacart.html", {"request": request, "data": result, "columns": columns})

@app.get("/split-data", response_class=HTMLResponse)
async def get_split_data(request: Request):
    # Load data
    X = data[['Produk_Tahu', 'aroma', 'tekstur', 'cita_rasa', 'masa_kadaluarsa']]
    y = data['Kualitas']
    
    # Drop rows with NaN values
    X = X.dropna()
    y = y[X.index]  # Ensure target variable y matches the filtered X

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Convert to DataFrame and concatenate with target variable
    X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
    y_train_df = pd.DataFrame(y_train).reset_index(drop=True)
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
    
    train_data = pd.concat([X_train_df, y_train_df], axis=1).to_dict(orient='records')
    test_data = pd.concat([X_test_df, y_test_df], axis=1).to_dict(orient='records')
    
    # Extract column names for DataTables
    train_columns = train_data[0].keys() if train_data else []
    
    return templates.TemplateResponse("splitdata.html", {
        "request": request,
        "train_data": train_data,
        "test_data": test_data,
        "train_columns": train_columns
    })
