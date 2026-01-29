from model import load_model

NUM_CLASSES = 10  # change to your actual number
model = load_model(NUM_CLASSES)

print(model)
