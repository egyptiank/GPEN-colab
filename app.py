import os
import gradio as gr
import cv2


os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth && mv RetinaFace-R50.pth GPEN/weights/")
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-512.pth && mv GPEN-512.pth GPEN/weights/")
os.system("cd GPEN")
def updateF():
  os.system("rm -rf examples/imgs")
  os.system("rm -rf examples/outs") 
  os.system("mkdir examples/imgs")
  os.system("mkdir examples/outs")

def predict(img):
  updateF()
  cv2.imwrite('examples/imgs/temp.jpg', img)
  os.system("python face_enhancement.py")
  return "examples/outs/temp.jpg"

# launch a gradio interface
gr.Interface(predict, "image", "image").launch()
