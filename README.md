# IAT TP1 : CNN, Image classification of obstacles

The project is meant to evaluate different Convolutional Neural Networks (CNN) on image classification.
CNN implementation relies on the python library _tensorflow_.

## How to run the app?

Steps:

1. Download images dataset (in project directory):  `./download-dataset.sh`
2. Create and deplace in the app virtual environment (same): `./app-env.sh`
3. Activate the virtual environment (in .venv/): `source iat-tp1/bin/activate`
4. Install python dependencies: `pip install -r requirements.txt`
5. Run app (in project directory): `python3 main.py`

If you want to run a serie of test, run `python3 test.py`.

## How to see logs?

Use Tensorboard: 
1. Run Tensorboard: `tensorboard --logdir=./logs/fit`
2. Open Tensorbord in browser on [https://localhost:6006/](https://localhost:6006/)
