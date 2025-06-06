from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from huggingface_hub import login
from model import MistralQA
import pandas as pd
import io
import os

# Инициализируем фреймворк
app = FastAPI()

# Токен для доступа в Hugging Face
with open('./hf_key.txt', 'r') as f:
    hf_key = f.read().splitlines()[0]

# Авторизуемся на Hugging Face
login(token=hf_key)

# Инициализируем LLM-модель
qa = MistralQA()

@app.get("/answer", response_class=JSONResponse)
async def answer(question: str):
    """Принимаем строку с вопросом и возвращаем ответ"""
    prepromt = 'Ты - помощник, отвечающий на вопросы. Используй предоставленный контекст для точных ответов.' +\
        'Если ответа нет в контексте, скажи об этом. Будь кратким и информативным. Ответ дай на русском языке. '

    qa_answer = qa.answer(prepromt + question)
    return JSONResponse(
                content={
                   'answer': qa_answer,
                }
           )


@app.get("/save", response_class=JSONResponse)
async def save():
    """Сохраняем базу данных модели на диск"""
    qa.save()
    return JSONResponse(
                content={
                   'message': "Model saved.",
                }
           )

@app.get("/reload", response_class=JSONResponse)
async def save():
    """Перезагружаем базу данных модели"""
    qa.load()
    return JSONResponse(
                content={
                   'message': "Model reloaded.",
                }
           )


@app.post("/learn", response_class=JSONResponse)
async def learn(file: UploadFile = File(...)):
    """Запускаем дообучение на новых данных"""
    # Проверяем так:
    # curl -X POST -F "file=@example.csv" http://localhost:8000/upload

    # Проверка расширения файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен иметь расширение .csv"
        )

    try:
        # Чтение содержимого файла
        contents = await file.read()

        # Создание pandas DataFrame из содержимого файла

        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        qa.load_qa_from_df(df)

        # Возвращаем информацию о файле
        return JSONResponse(
            content={
                "message": "Model fine-tuned."
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )


if __name__ == "__main__":

    if os.path.isdir('./my_mistral_model'):
        qa.load()
    else:
        # начальный запуск, модель еще не fine-tuned
        qa.add_website("https://www.consultant.ru/document/cons_doc_LAW_114247/")
        qa.add_website("https://www.consultant.ru/cons/cgi/online.cgi?req=doc&base=LAW&n=493210&dst=100001#FJKp4lUmAbBG6UBo")
        qa.add_website("https://www.consultant.ru/cons/cgi/online.cgi?req=doc&base=LAW&n=485337#YKyp4lUNwfLyBFmJ")
        qa.add_website("https://rstkirov.ru/tarify/tarify/")
        qa.add_website("https://www.socialkirov.ru/social/root/uszn/subsidiijku/info.htm")
        qa.load_qa_from_parquet('./data/sberquad-validation.parquet')
        qa.load_qa_from_csv('./data/_вопросы_с_ответами.csv')

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
