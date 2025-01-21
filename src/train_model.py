import os
import re
import spacy
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin

# List of mountain names for the task (for labeling the dataset)
MOUNTAIN_NAMES = [
    'Everest', 'Kilimanjaro', 'Vesuvius', 'Fuji', 'St. Helens', 'K2', 'Olympus', 'McKinley', 'Denali', 'Cook',
    'Rainier', 'Kailash Mountain', 'Rocky Mountains', 'Andes Mountain range', 'Blanc', 'Hengshan', 'Appalachian Mountains',
    'Eiger', 'Elbrus', 'Popa', 'Lemmon', 'Robson', 'Rushmore', 'El Capitan', 'Huangshan'
]


def label_sentences(sentences, mountain_names):
    """ Labels sentences by marking mountain names. """
    labeled_sentences = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence)
        labeled_words = []
        for word in words:
            if any(mountain_name.lower() == word.lower() for mountain_name in mountain_names):
                labeled_words.append('MOUNTAIN')
            else:
                labeled_words.append('O')
        labeled_sentences.append(labeled_words)
    return labeled_sentences


def split_dataset(df, test_size=0.2):
    """ Splits the dataset into training and evaluation sets. """
    return train_test_split(df, test_size=test_size, random_state=42)


def convert_to_spacy_format(df, label="MOUNTAIN"):
    """ Converts data into SpaCy format. """
    spacy_data = []
    for _, row in df.iterrows():
        text = row["sentence"]
        entities = []
        for match in re.finditer(r'\b(?:' + '|'.join(map(re.escape, MOUNTAIN_NAMES)) + r')\b', text, re.IGNORECASE):
            entities.append((match.start(), match.end(), label))
        spacy_data.append((text, {"entities": entities}))
    return spacy_data


def save_to_spacy(data, output_path, nlp):
    """ Saves data in SpaCy format. """
    db = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        entities = annotations["entities"]
        spans = [doc.char_span(start, end, label=label) for start, end, label in entities]
        spans = [span for span in spans if span is not None]
        doc.ents = spans
        db.add(doc)
    db.to_disk(output_path)


def train_spacy_model():
    """ Full workflow for training the SpaCy model. """
    # Load the dataset (mountains.csv)
    dataset_path = "data/mountains.csv"  # Path to your CSV file
    df = pd.read_csv(dataset_path)

    # Label the data
    df['labels'] = label_sentences(df['sentence'], MOUNTAIN_NAMES)

    # Split the dataset into training and evaluation
    train_sentences, eval_sentences = split_dataset(df)

    # Convert data to SpaCy format
    train_data_spacy = convert_to_spacy_format(pd.DataFrame(train_sentences, columns=df.columns))
    eval_data_spacy = convert_to_spacy_format(pd.DataFrame(eval_sentences, columns=df.columns))

    # Create a blank SpaCy model
    nlp = spacy.blank("en")

    # Save training and evaluation datasets in SpaCy format
    save_to_spacy(train_data_spacy, "data/train.spacy", nlp)
    save_to_spacy(eval_data_spacy, "data/eval.spacy", nlp)

    # Train the model
    os.system("python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency")
    os.system("python -m spacy train config.cfg --output ./model/output --paths.train ./data/train.spacy --paths.dev ./data/eval.spacy")

    # Archive the trained model
    shutil.make_archive("model/spacy_model", "gztar", "./model/output", "model-best")

    print("Model training and saving completed.")


if __name__ == "__main__":
    train_spacy_model()