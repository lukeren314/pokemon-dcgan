# Pokemon DCGAN

Welcome to my Pokemon Sprite Generator!
This project uses an algorithm called Deep Convolutional Generative Adversarial Network in order to generate unique sprites that look like Pokemon(-ish). Although
this algorithm was designed to generate Pokemon, it can be applied to any dataset
of images you like.

In order to run the code and generate sprites for yourself, first download a folder of images you would like to use into the project directory and rename the folder to "data".

Make sure you have [Python](python.org) installed and that it is added to your environment variables.

Next, open a command prompt/terminal in the project directory and run the following pieces of code to create a virtual environment.

```bash
python -m venv env
```

Once you've created your virtual environment, enter into it using

```bash
env\Scriptcs\activate
```

if you are one Windows, and if you are on Unix

```
source env/bin/activate
```

Once you've connected to your virtual environment, install the required modules using the [pip](https://pip.pypa.io/en/stable/) package manager.

```bash
pip install -r requirements.txt
```

Then, run the next two commands to train and then sample your model. use python3 instead if you are on unix.

```bash
python dcgan.py.py
python sample.py
```
