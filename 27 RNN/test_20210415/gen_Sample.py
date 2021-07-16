from PIL import Image,ImageDraw,ImageFont,ImageFilter
import random
import os

def ranChr():

    return str(random.randint(0,9))

def Color1():
    return (
        random.randint(64,255),
        random.randint(64,255),
        random.randint(64,255),
    )
def Color2():
    return (
        random.randint(32,128),
        random.randint(32,128),
        random.randint(32,128),
    )

font = ImageFont.truetype("arial.ttf",30)
h=60
w=120
for i in range(1000):
    img = Image.new("RGB",(w,h),(255,255,255))
    draw = ImageDraw.Draw(img)
    for x in range(w):
        for y in range(h):
            draw.point((x,y),fill=Color1())
    filename =""
    for j in range(4):
        char = ranChr()
        draw.text((30*j+10,10),char,font=font,fill=Color2())
        filename+=char
    if not os.path.exists("data"):
        os.makedirs("data")

    img.save("{}/{}.jpg".format("data",filename))
    print(i)