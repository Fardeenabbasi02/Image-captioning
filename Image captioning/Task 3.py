import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import nltk
nltk.download('punkt')

# Load pre-trained VGG16 model + higher level layers
def extract_image_features(image_path):
    model = VGG16(include_top=False, weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # VGG expects 224x224 images
    img = np.array(img)
    if img.shape[2] == 4:  # Check if the image has an alpha channel (RGBA)
        img = img[..., :3]  # Convert to RGB (ignore alpha)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize image

    # Extract image features
    features = model.predict(img)
    features = np.reshape(features, (features.shape[0], -1))  # Flatten the features
    return features

# Simple function to visualize the image
def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Text Preprocessing - Tokenize and prepare the text sequences
def preprocess_text(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(captions)
    max_sequence_len = max([len(seq) for seq in sequences])
    
    return tokenizer, vocab_size, max_sequence_len

# Load and preprocess captions (dummy captions for demo purposes)
def load_captions():
    # In practice, load actual dataset with image-path, caption pairs
    captions = ["A cat sitting on a mat", 
                "A dog playing with a ball",
                "A group of people at the beach",
                "A man riding a horse",
                "A woman holding an umbrella"]
    return captions

# Define the captioning model
def build_captioning_model(vocab_size, max_sequence_len):
    # Image feature extractor (input shape is 4096, as we used VGG16 features)
    image_input = Input(shape=(4096,))
    image_features = Dropout(0.5)(image_input)
    image_features = Dense(256, activation='relu')(image_features)

    # Sequence input for text (captions)
    caption_input = Input(shape=(max_sequence_len,))
    caption_features = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    caption_features = LSTM(256)(caption_features)

    # Combine the two streams
    combined = tf.keras.layers.add([image_features, caption_features])
    combined = Dense(256, activation='relu')(combined)
    outputs = Dense(vocab_size, activation='softmax')(combined)

    # Build and compile model
    model = Model(inputs=[image_input, caption_input], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Generate captions (Greedy search or Beam search can be used here)
def generate_caption(model, image_features, tokenizer, max_sequence_len):
    input_seq = [tokenizer.word_index['startseq']]
    
    for _ in range(max_sequence_len):
        input_seq_padded = pad_sequences([input_seq], maxlen=max_sequence_len)
        y_pred = model.predict([image_features, input_seq_padded], verbose=0)
        next_word_id = np.argmax(y_pred)
        next_word = tokenizer.index_word[next_word_id]
        input_seq.append(next_word_id)
        if next_word == 'endseq':
            break
    
    caption = ' '.join([tokenizer.index_word[i] for i in input_seq])
    return caption.replace('startseq', '').replace('endseq', '').strip()

# Main function to run the image captioning pipeline
def run_image_captioning(image_path):
    # Load and preprocess image
    image_features = extract_image_features(image_path)

    # Load and preprocess text data (captions)
    captions = load_captions()
    tokenizer, vocab_size, max_sequence_len = preprocess_text(captions)

    # Build the image captioning model
    model = build_captioning_model(vocab_size, max_sequence_len)
    
    # For demo purposes, let's assume the model is already trained.
    # You would normally train the model using the captions and images dataset.
    
    # Generate caption for the image
    caption = generate_caption(model, image_features, tokenizer, max_sequence_len)
    print(f"Generated caption: {caption}")

    # Show the image
    show_image(image_path)

if _name_ == '_main_':
    # Sample image path (replace with actual image)
    image_path = 'sample_image.jpg'
    run_image_captioning(image_path)