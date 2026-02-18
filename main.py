# main.py
import numpy as np
from datasets import load_dataset
from settings import Config
from preprocessing import TextProcessor
from model import Word2Vec

def get_hf_data():
    print("Downloading dataset fka/prompts.chat...")
    # Load the dataset (using 'train' split by default)
    ds = load_dataset("fka/prompts.chat", split="train")
    
    # Extract text from the 'prompt' column and join into one massive string
    # In production, you'd stream this, but for 1.25k rows, RAM is fine.
    print(f"Loaded {len(ds)} rows. Merging text...")
    text_blob = " ".join([item['prompt'] for item in ds if item['prompt']])
    return text_blob

def run():
    cfg = Config()
    processor = TextProcessor()
    
    # Switch data source to Hugging Face
    raw_text = get_hf_data()
    
    # The rest of the pipeline remains exactly the same
    data = processor.process(raw_text, cfg.MIN_COUNT)
    print(f"Vocab: {processor.vocab_size}, Total Tokens: {len(data)}")
    
    model = Word2Vec(processor.vocab_size, cfg)

    # Training loop
    for epoch in range(cfg.EPOCHS):
        # Progress indicator
        sys.stdout.write(f"\rEpoch {epoch+1}/{cfg.EPOCHS} running...")
        sys.stdout.flush()
        
        for i, center_id in enumerate(data):
            start = max(0, i - cfg.WINDOW_SIZE)
            end = min(len(data), i + cfg.WINDOW_SIZE + 1)
            
            w_in_grad = np.zeros(cfg.EMBED_DIM)
            
            for j in range(start, end):
                if i == j: continue
                context_id = data[j]
                
                # Positive
                w_in_grad += model.train_pair(center_id, context_id, 1)
                
                # Negative
                for _ in range(cfg.NEGATIVE_SAMPLES):
                    neg_id = np.random.randint(0, processor.vocab_size)
                    if neg_id == context_id: continue
                    w_in_grad += model.train_pair(center_id, neg_id, 0)

            model.w_in[center_id] -= w_in_grad
            
        print(f" Done.")

    # Test
    print("\n--- Similarity Check ---")
    try:
        # These words definitely exist in a 'prompts' dataset
        check_words = [('code', 'python'), ('ai', 'human'), ('error', 'bug')]
        
        for w1, w2 in check_words:
            if w1 in processor.word2id and w2 in processor.word2id:
                v1 = model.w_in[processor.word2id[w1]]
                v2 = model.w_in[processor.word2id[w2]]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                print(f"{w1} <-> {w2}: {sim:.4f}")
            else:
                print(f"Skipping {w1}/{w2} (not in vocab)")
                
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    import sys
    run()
