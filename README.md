# Place365 Scene Classification with Paddle
Online demo is in [![SwanHub](https://img.shields.io/static/v1?label=Demo&message=SwanHub&color=blue)](https://swanhub.co/shaohon/place365classification/demo)!

## Project Introduction

This project is a scene classification model using Resnet50 based on the PaddlePaddle framework. It has been trained using the Place365 dataset, and it achieved a Top1 accuracy of 53% and a Top10 accuracy of 81% on this dataset.

## Installation

Here are the steps to install the project:

1. Clone the project repository:

   ```shell
   git clone https://github.com/shaohon/place365classification.git
   ```

2. Enter the project directory:

   ```shell
   cd place365classification
   ```

3. Install the requirements:

   ```shell
   pip install -r requirements.txt
   ```

## Usage

### Use on SwanHub

you can test the model in [![SwanHub](https://img.shields.io/static/v1?label=Demo&message=SwanHub&color=blue)](https://swanhub.co/shaohon/place365classification/demo)

### Run on local machine

Follow the next steps to run locally.

1. Download weights in [download link](https://swanhub.co/shaohon/place365classification/blob/main?path=place365.pdparams)
and put the file in the project root.

3. Run the prediction script in local:

```shell
python app.py
```

### Deploy local API predict server

You can use [SwanAPI](https://github.com/SwanHubX/SwanAPI/blob/main/README_EN.md) to package the model as an API service.

First, install swanapi using the following command

```shell
pip install swanapi -i https://pypi.org/simple
```

Then, you can directly turn the model into a prediction service.

```shell
python predict.py
```

You can also build a deep learning inference image with just one command.

```shell
swanapi build -t my-dl-model
```

Run ```python tools/post.py``` to test whether the service is online

## Contributing

Your contributions to improve this project are always welcome. You can participate by following these steps:

1. Fork this repo to your own GitHub account.
2. Make a new branch and make your changes: `git checkout -b your_branch_name`
3. Commit your changes: `git commit -m 'Add some features'`
4. Push your branch to your GitHub repo: `git push origin your_branch_name`
5. Submit a Pull Request and wait for review.

## Contact us

If you ever encounter any issues while using the project, or have any suggestions, please don't hesitate to contact us:

- Author: shaohon
- Email: shaohon.chen@swanhub.co

Thank you for your support!

## License

This project is released under the GNU General Public License (GPL). This is a widely used free software license, which guarantees end users the freedom to run, study, share, and modify the software.

For the full license text, please see the [LICENSE](LICENSE) file in the root directory of this source tree. If the LICENSE file does not exist, you can add one. Note that you should include the full GPL license text in this LICENSE file.

This ends the README of the scene classification open source project, and I hope you find it helpful. If you need additional information or have any other needs, please let us know.
