import paddle
from paddle.vision import transforms
import gradio as gr


class Place365cls:
    def __init__(self, model_pth='save/place365'):
        # init model
        network = paddle.vision.models.resnet50(pretrained=False)
        network.fc = paddle.nn.Linear(2048, 365)
        model = paddle.Model(network)
        model.load(model_pth)
        self.model = model

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
        score, label = paddle.topk(out, k=5)
        return {self.id2categories_name[int(c)]: float(s) for c, s in zip(label, score)}


cls_model = Place365cls()


def predict(image):
    result = cls_model.predict(image, top_n=5)
    result_json = {item["label"]: float(item["score"]) for item in result}
    return result_json


if __name__ == "__main__":
    interface = gr.Interface(fn=predict, inputs="image",
                             outputs=gr.Label(),
                             title="ResNet50"
                             )
    interface.launch()
