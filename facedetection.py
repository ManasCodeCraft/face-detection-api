from fastapi import FastAPI, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
import cv2
from mtcnn import MTCNN
import numpy as np
import base64

app = FastAPI()

DEFAULT_COLOR = "(0,255,0)"

@app.post("/detect-face-info/")
async def detect_face(file: UploadFile):
    allowed_extensions = {"jpg", "jpeg", "png","webp"}
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only image files (jpg, jpeg, png) are allowed.")
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        pixels = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detector = MTCNN()

        faces = detector.detect_faces(pixels)

        if not faces:
            return JSONResponse(content={"message": "No faces found in the image."})
        
        faces_data = {}
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(pixels, (x, y), (x + width, y + height), (0, 255, 0), 2)
            faces_data_element = {
            "xmin" : x,
            "ymin" : y,
            "xmax" : x+width,
            "ymax" : y+height,
            "width" : width,
            "height" : height
            }
            faces_data["face"+str(len(faces_data)+1)] = faces_data_element

        faces_data["total_faces"] = len(faces)
        faces_data["message"] = "Faces detected successfully"

        return JSONResponse(content=faces_data)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    
@app.post("/detect-faces-image/")
async def detect_face(
    file: UploadFile,
    box_color: str = Query(None, description="Bounding box color in BGR format (e.g., '0,255,0' for green)")):
    allowed_extensions = {"jpg", "jpeg", "png","webp"}
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only image files (jpg, jpeg, png) are allowed.")
    try:
        try:
          if box_color:
              box_color = tuple(map(int, box_color.split(",")))
          else:
              box_color = DEFAULT_COLOR
        except ValueError:
              box_color = DEFAULT_COLOR

        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        pixels = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detector = MTCNN()

        faces = detector.detect_faces(pixels)

        if not faces:
            return JSONResponse(content={"message": "No faces found in the image."})
        
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(pixels, (x, y), (x + width, y + height), (0, 255, 0), 2)

        pixels_rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

        _, img_data = cv2.imencode("."+file_extension, pixels_rgb)
        img_base64 = base64.b64encode(img_data).decode()

        return JSONResponse(content={"image": img_base64})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
