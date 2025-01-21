# Mountain Recognition with SpaCy

This project demonstrates how to use SpaCy to create a Named Entity Recognition (NER) model that identifies mountain names in text.

## Project Structure

The project is organized as follows:

- **`data/`**: Contains the `mountains.csv` dataset used for training the model.
- **`model/`**: Directory where the trained SpaCy model is stored (`model-best`).
- **`requirements.txt`**: List of dependencies required to run the project.
- **`train_model.py`**: Python script that handles the training process for the NER model.
- **`inference.py`**: Python script for testing the trained model on sample text.
- **`Mountain_Recognition.ipynb`**: Jupyter notebook that walks through the entire process from data generation to model training and inference.
- **`Mountain_Recognition_Recommendations.pdf`**: A PDF document containing recommendations on how to improve model performance, including dataset expansion, model improvements, and more.


## Steps to Use the Project

### 1. Set up the Environment

To get started, you'll need Python 3.6 or later. First, clone the repository and install the necessary dependencies.

```bash
# Clone the repository
git clone https://github.com/InnaK342/mountain-recognition-ner.git
cd mountain-recognition-ner

# Install required dependencies
pip install -r requirements.txt
```
### 2. Prepare the Data

The `data/mountains.csv` file contains sentences mentioning various mountain names. This data has been pre-labeled for training the model. If you need to generate more data, you can refer to the `notebooks/Mountain_Recognition.ipynb` file for how the data was generated.

### 3. Train the Model

Run the `src/train_model.py` script to train the NER model. This script will use the labeled data to create a model that can recognize mountain names.
```bash
python src/train_model.py
```
This command will:

- Load the `data/mountains.csv` file for training.
- Train a SpaCy model using the labeled data.
- Save the trained model in the `model/` directory as `model-best`.

### 4. Test the Model (Inference)
Once the model is trained, you can use the `src/inference.py` script to test the model on sample texts. This script will load the trained model and recognize mountain names in any input text.

If you want to pass a custom sentence as input, you can do so directly in the command line:

```bash
python src/inference.py "Mount Fuji is a beautiful mountain in Japan."
```

Output:

```text
Mountain names found: ['Fuji']
```

If you do not pass any input sentence as a command-line argument, the script will prompt you to enter a sentence interactively:

```bash
python src/inference.py
```
Example interaction:

```text
Enter a sentence (or type 'exit' to quit): The weather is lovely today.
No mountain names found. Here's the original sentence: The weather is lovely today.
If the model does not detect any mountain names, it will return the original sentence as it was inputted. This ensures that the user gets the feedback they need even if no mountain names are found in the input.
```

### 5. Additional Notes

*   **Colab Notebook**: If you'd like to see the entire process step-by-step, including data preprocessing, training, and inference, check out the `Model_Recognition.ipynb` file.
*   **Model Storage**: After training, you can save the trained model in a specific location. In the `src/inference.py` script, make sure the `model_path` points to the correct directory where the model is stored.
