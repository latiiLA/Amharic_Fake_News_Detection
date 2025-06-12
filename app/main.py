from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re


# Load model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

app = FastAPI()

class NewsInput(BaseModel):
    text: str

amharic_punctuations = "።፣፤፥፦፧፨!፡፠፿፭፫፱፳፻፰፲፮፲፯፷፰፱፲፳፴፵፶፷፸፹፺፻፼፽፾፿፟ፚፙፘፗፖፕፔፓፒፑፐፏፎፍፌፋፊፉፆፅፄፃፂፁፀ"

def clean_text(text):
    text = re.sub(f"[{amharic_punctuations}]", "", text)
    text = re.sub(r"[^\u1200-\u137F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI"}


@app.post("/predict")
def predict_news(data: NewsInput):
    cleaned = clean_text(data.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return {"prediction": "Fake" if prediction == 0 else "Real"}