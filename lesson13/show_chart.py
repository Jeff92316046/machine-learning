import numpy as np
import json

import matplotlib.pyplot as plt

# Load history from output.json
with open('lesson14/output.json', 'r') as f:
    history = json.load(f)

# Fill with your actual data
# For demonstration, let's create some dummy data

# Show chart every 10 data points
for i in range(10, len(history['loss'])+1, 10):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, i+1), history['loss'][:i], label='loss')
    plt.plot(range(1, i+1), history['val_loss'][:i], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss up to epoch {i}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, i+1), history['accuracy'][:i], label='accuracy')
    plt.plot(range(1, i+1), history['val_accuracy'][:i], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy up to epoch {i}')
    plt.legend()

    plt.tight_layout()
    plt.show()