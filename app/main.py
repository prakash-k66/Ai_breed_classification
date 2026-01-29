from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.predict import router as predict_router

app = FastAPI(title="Cattle & Buffalo Breed Identification System")

# Include prediction API
app.include_router(predict_router)

# Serve static UI (HTML, CSS, JS)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Health check
@app.get("/health")
def health_check():
    return {"status": "API is running"}
