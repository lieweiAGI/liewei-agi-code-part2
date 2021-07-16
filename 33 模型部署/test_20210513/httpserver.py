from flask import Flask,request,jsonify
app = Flask(__name__)
import io
from PIL import Image
from net import Netv3
import torch,torchvision

@app.route("/")
def hello():
    return "Hello World"

@app.route("/xxx",methods=["POST"])
def xxx():
    # print(request.args.get("name"))
    name = request.form.get("name")

    file = request.files.get("file")
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))
    # image.show()

    image_data = torchvision.transforms.ToTensor()(image)
    image_data = image_data.reshape(-1,784)
    net = Netv3()
    net.load_state_dict(torch.load("169.t"))
    y = net(image_data)
    return jsonify({"name":name,"filelen":len(img_bytes),"out_tensor:":str(y.data),"out_put":str(y.argmax().item())})

if __name__ == '__main__':
    #启动服务端
    app.run()
