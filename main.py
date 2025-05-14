from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse

app = FastAPI()


@app.get("/answer", response_class=PlainTextResponse)
async def answer(text: str):
    """Принимает строку и возвращает её в верхнем регистре"""
    return text.upper()


@app.get("/save", response_class=PlainTextResponse)
async def save():
    """Возвращает строку 'SAVED!'"""
    return "SAVED!"


@app.post("/learn", response_class=PlainTextResponse)
async def learn(file: UploadFile = File(...)):
    """Принимает файл и возвращает его размер"""
    content = await file.read()
    file_size = len(content)
    return f"File size is {file_size} bytes"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
