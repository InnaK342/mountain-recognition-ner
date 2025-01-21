import spacy
import sys


def load_model(model_path):
    """Load the trained SpaCy model."""
    try:
        nlp = spacy.load(model_path)
        print(f"Model loaded from {model_path}")
        return nlp
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def recognize_mountains(nlp, text):
    """Recognize mountain names in the provided text."""
    doc = nlp(text)
    mountain_names = [ent.text for ent in doc.ents if ent.label_ == "MOUNTAIN"]
    return mountain_names


def main():
    # Path to the trained SpaCy model
    model_path = "model/model-best"

    # Load the model
    nlp = load_model(model_path)

    # Check if sentences are passed as command-line arguments
    if len(sys.argv) > 1:
        sentences = " ".join(sys.argv[1:])
    else:
        # Otherwise, take input from the user interactively
        print("Enter a sentence (or type 'exit' to quit):")
        sentences = input()

    # Perform inference on the input text
    mountain_names = recognize_mountains(nlp, sentences)

    if mountain_names:
        print(f"Mountain names found: {', '.join(mountain_names)}")
    else:
        # If no mountain names found, return the original sentence
        print(f"No mountain names found. Here's the original sentence: {sentences}")


if __name__ == "__main__":
    main()
