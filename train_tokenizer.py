"""
Train a new SentencePiece tokenizer for Sheikh-1B-Instruct
with vocabulary size of 32,768 tokens.
"""

import os
from transformers import PreTrainedTokenizerFast
from tokenizers import processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC, Lowercase, Strip

def create_tokenizer(vocab_size=32768, output_dir="/workspaces/sheikh-1B-Instruct/model"):
    """Create and train a new tokenizer with the specified vocab size"""
    # Initialize a new BPE tokenizer
    tokenizer = Tokenizer(BPE())
    
    # Add normalizers
    tokenizer.normalizer = NFKC()
    
    # Add pre-tokenizer
    tokenizer.pre_tokenizer = ByteLevel()
    
    # Setup special tokens
    special_tokens = ["<s>", "</s>", "<unk>", "[INST]", "[/INST]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    # Create sample text for training
    sample_text = "\n".join([
        "This is a sample text to initialize the tokenizer vocabulary.",
        "It includes special tokens like [INST] and [/INST]",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
        "Deep learning models are transforming AI.",
        "Here's some code: def hello(): print('world')",
        "[INST] What is machine learning? [/INST]",
        "Machine learning is a subset of artificial intelligence.",
        "1234567890!@#$%^&*()",
        "http://example.com",
        "email@example.com",
        "</s> <s> <unk>"
    ]) * 100  # Repeat to have enough data
    
    # Save sample text
    tmp_file = "/tmp/sample_text.txt"
    with open(tmp_file, "w") as f:
        f.write(sample_text)
    
    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train([tmp_file], trainer)
    
    # Convert to PreTrainedTokenizerFast
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<s>",  # Reuse BOS as padding token
        model_max_length=32768,  # Match your model's max position embeddings
    )
    
    # Add special tokens to token maps
    wrapped_tokenizer.add_special_tokens({
        "additional_special_tokens": ["[INST]", "[/INST]"]
    })
    
    # Save the tokenizer
    print(f"Saving tokenizer to {output_dir}")
    wrapped_tokenizer.save_pretrained(output_dir)
    
    # Clean up
    os.remove(tmp_file)
    
    print(f"Vocabulary size: {wrapped_tokenizer.vocab_size}")
    print("Sample tokens:", wrapped_tokenizer.encode("Hello, world! [INST] How are you? [/INST]"))
    
    return wrapped_tokenizer

if __name__ == "__main__":
    # Create tokenizer with 32k vocab
    tokenizer = create_tokenizer(vocab_size=32768)
    
    print("\nTokenizer files saved successfully!")
    print("You can now use this tokenizer with your Sheikh-1B model.")
