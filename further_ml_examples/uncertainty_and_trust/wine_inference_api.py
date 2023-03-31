from fastapi import FastAPI

app = FastAPI()

# todo create an api to serve results
# use mlflow model serve4

@app.get("/")
async def root():
    return {"message": "Hello World"}