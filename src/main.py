from fastapi import  FastAPI
from routers import images


app = FastAPI()


@app.get("/")
async def root():
    '''
    endpoint usado para debugging
    '''
    return {"Hello":"World"}

app.include_router(images.router)