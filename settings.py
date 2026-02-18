class Config:
    WINDOW_SIZE = 3
    MIN_COUNT = 1
    NEGATIVE_SAMPLES = 5
    
    EMBED_DIM = 50
    LEARNING_RATE = 0.005
    EPOCHS = 5
    
    # Clip logits to prevent exp() overflow in sigmoid
    MAX_EXP = 6.0
