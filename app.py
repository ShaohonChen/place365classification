import paddle
from paddle.vision import transforms
import gradio as gr
import numpy as np

class Place365cls:
    def __init__(self, model_pth='./place365.pdparams'):
        # init model
        network = paddle.vision.models.resnet18(pretrained=False)
        network.fc = paddle.nn.Linear(512, 365)
        model = paddle.Model(network)
        model.load(model_pth)
        self.model = model
        self.sfm=paddle.nn.Softmax()

        # init transform
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224, 224)),
            transforms.Normalize(mean=[127.5], std=[127.5]),
            transforms.ToTensor()
        ])

        # inti category name
        self.id2categories_name = dict()
        with open('categories_places365.txt') as f:
            lines = f.readlines()
            for line in lines:
                category_name, id = line.split(' ')
                id = int(id)
                category_name = category_name[3:]
                self.id2categories_name[id] = category_name

    def predict(self, img, top_n=5):
        img = self.transform(img)
        img = img.unsqueeze(0)
        out = self.model.predict_batch(img)[0]
        out = self.sfm(paddle.Tensor(out))
        score, label = paddle.topk(out[0], k=5)
        return {self.id2categories_name[int(c)]: float(s) for c, s in zip(label, score)}


"""
        label = np.argsort(out)[-top_n:][::-1]
        top_labels=dict()
        for c in label:
            c=int(c)
            c_name=self.id2categories_name[c]
            score=float(out[c])
            top_labels[c_name]=score
        return top_labels
"""

cls_model = Place365cls()


def predict(image):
    result = cls_model.predict(image, top_n=5)
    return result


if __name__ == "__main__":
    interface = gr.Interface(fn=predict, inputs="image",
                             outputs=gr.Label(),
                             title="Place365 classification"
                             )
    interface.launch()
