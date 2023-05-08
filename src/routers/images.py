import base64
from fastapi import APIRouter,UploadFile,Request
import schemas
import tensorflow as tf
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import numpy as np

router = APIRouter(
    prefix="/images",
    tags=["images"],
    responses={404: {"Description": "Not Found"}}
)

templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model("unet_model.h5")

@router.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.post(("/upload"))
async def create_upload_file(request: Request,file: UploadFile):
    
    #leemos la imagen
    data = await file.read()
    
    #preprocesamos la imagen
    image = Image.open(io.BytesIO(data))
    image_array = np.asarray(image)
    image_array = image_array.reshape(1,32,32,3)/255

    #pasamos la imagen por el modelo de reconstruccion
    restored = model.predict(image_array)

    #transformamos la salida en imagen
    restored = restored.reshape(32,32,3)*255
    imagen = restored.astype(np.uint8)
    jpeg_image = Image.fromarray(imagen)

    #transformamos la imagen en un stream de bytes para mandarla a la pagina html
    buf = io.BytesIO()
    jpeg_image.save(buf, format='JPEG')
    encoded_image = base64.b64encode(data).decode("utf-8")
    encoded_image2 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return templates.TemplateResponse(
        "upload2.html", {"request": request,
                        "img1":encoded_image,
                        "img2":encoded_image2})

