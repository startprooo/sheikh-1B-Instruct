"""
Test suite for Sheikh-1B-Instruct model.
"""

import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

class TestSheikh1B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load model and tokenizer once for all tests"""
        cls.model_path = "./model"
        cls.config = LlamaConfig.from_pretrained(cls.model_path)
        cls.tokenizer = LlamaTokenizer.from_pretrained(cls.model_path)
        
        # Only load model if weights exist
        try:
            cls.model = LlamaForCausalLM.from_pretrained(cls.model_path)
            cls.has_weights = True
        except:
            cls.has_weights = False
    
    def test_config(self):
        """Test model configuration"""
        self.assertEqual(self.config.hidden_size, 4096)
        self.assertEqual(self.config.num_attention_heads, 32)
        self.assertEqual(self.config.num_hidden_layers, 32)
        self.assertEqual(self.config.vocab_size, 32768)
    
    def test_tokenizer(self):
        """Test tokenizer functionality"""
        # Test special tokens
        self.assertIn("<s>", self.tokenizer.special_tokens_map.values())
        self.assertIn("</s>", self.tokenizer.special_tokens_map.values())
        self.assertIn("[INST]", self.tokenizer.additional_special_tokens)
        self.assertIn("[/INST]", self.tokenizer.additional_special_tokens)
        
        # Test encoding/decoding
        text = "Hello, world! [INST] How are you? [/INST]"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded)
    
    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    def test_gpu_support(self):
        """Test GPU support if available"""
        if not self.has_weights:
            self.skipTest("No model weights available")
        
        self.model.to("cuda")
        self.assertTrue(next(self.model.parameters()).is_cuda)
    
    def test_generation(self):
        """Test text generation if weights are available"""
        if not self.has_weights:
            self.skipTest("No model weights available")
        
        prompt = "[INST] What is machine learning? [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), len(prompt))

if __name__ == "__main__":
    unittest.main()
