from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from message import Message
from sentimentAnalysis.qatar.qatar_model_wrapper import QatarModelWrapper

sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
qatar = QatarModelWrapper()
app = FastAPI()
# se usa cors para permitir el acceso al api solo a clientes autorizados, en este caso se permite a todos por cuestiones practicas
origins = [
    "*"
]

app.add_middleware(
    TrustedHostMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # se pueden especificar los metodos http permitidos
    allow_headers=["*"], # se pueden especificar los headers permitidos
)

@app.get('/')
def get_root():
    return {'Este es el backend de una aplicación de análisis de sentimientos'}


# @app.post('/v1/analysispost/')
# async def query_analysis_post(content: Message):
#     return analyze_sentiment(content.content)

@app.post('/v2/qatarreview/')
async def query_analysis_post(content: Message):
    return qatar.get_review(content.content)


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

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)