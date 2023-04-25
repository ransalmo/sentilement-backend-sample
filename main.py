from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from message import Message

sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def get_root():
    return {'Este es el backend de una aplicación de análisis de sentimientos'}


@app.get('/analysis/')
async def query_analysis(text: str):
    return analyze_sentiment(text)


@app.post('/analysispost/')
async def query_analysis_post(content: Message):
    return analyze_sentiment(content.content)


def analyze_sentiment(text: str):
    """Get and process result"""
    result = sentiment_pipeline(text)
    sent = ''
    if result[0]['label'] == '1 star':
        sent = 'Muy negativo'
    elif result[0]['label'] == '2 star':
        sent = 'Negativo'
    elif result[0]['label'] == '3 stars':
        sent = 'Neutral'
    elif result[0]['label'] == '4 stars':
        sent = 'Positivo'
    else:
        sent = 'Muy positivo'
    prob = result[0]['score']
    # Format and return results
    return {'sentiment': sent, 'probability': prob, 'text': text}
