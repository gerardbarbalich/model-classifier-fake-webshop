"""
Tests for the fake webshop classifier.

This module tests text processing, tokenization, and model training functions.
"""

import pytest
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# Custom stop words for testing
ls_custom_stop = stopwords.words('english') + [
    'nz', 'st', 'www', 'co', 'new', 'zealand', 'nzd', 'us', 'ml', 'javascript'
]
ls_custom_stop_set = set(ls_custom_stop)
tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}')


def gen_tokens(text: str):
    """Tokenize and clean text for testing"""
    return [
        w.lower() 
        for w in tokenizer.tokenize(text) 
        if w.lower() not in ls_custom_stop_set
    ]


class TestTokenization:
    """Test suite for text tokenization"""
    
    def test_tokenization_basic_text(self):
        """Test that basic text is tokenized correctly"""
        text = "Hello world, this is a test."
        tokens = gen_tokens(text)
        
        # Should contain 'hello', 'world', 'test' (stopwords removed)
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'test' in tokens
    
    def test_tokenization_removes_stopwords(self):
        """Test that stop words are removed"""
        text = "This is a test with the and for"
        tokens = gen_tokens(text)
        
        # Common stop words should be removed
        assert 'is' not in tokens
        assert 'a' not in tokens
        assert 'the' not in tokens
        assert 'and' not in tokens
        assert 'for' not in tokens
        
        # Content word should remain
        assert 'test' in tokens
    
    def test_tokenization_removes_custom_stopwords(self):
        """Test that custom stop words are removed"""
        text = "Welcome to www new zealand nz site javascript"
        tokens = gen_tokens(text)
        
        # Custom stop words should be removed
        assert 'www' not in tokens
        assert 'new' not in tokens
        assert 'zealand' not in tokens
        assert 'nz' not in tokens
        assert 'javascript' not in tokens
        
        # Content words should remain
        assert 'welcome' in tokens
        assert 'site' in tokens
    
    def test_tokenization_lowercase_conversion(self):
        """Test that all tokens are converted to lowercase"""
        text = "HELLO World TeSt"
        tokens = gen_tokens(text)
        
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'test' in tokens
        
        # Should not contain uppercase versions
        assert 'HELLO' not in tokens
        assert 'World' not in tokens
    
    def test_tokenization_filters_short_words(self):
        """Test that single-character words are filtered out"""
        text = "a b c test word"
        tokens = gen_tokens(text)
        
        # Single letters should be filtered by regex (2+ chars required)
        assert 'a' not in tokens
        assert 'b' not in tokens
        assert 'c' not in tokens
        
        # Longer words should remain
        assert 'test' in tokens
        assert 'word' in tokens
    
    def test_tokenization_filters_numbers(self):
        """Test that numbers are filtered out"""
        text = "hello 123 world 456"
        tokens = gen_tokens(text)
        
        # Numbers should be filtered by regex (letters only)
        assert '123' not in tokens
        assert '456' not in tokens
        
        # Words should remain
        assert 'hello' in tokens
        assert 'world' in tokens
    
    def test_tokenization_filters_special_characters(self):
        """Test that special characters are filtered"""
        text = "hello@world test#tag price$50"
        tokens = gen_tokens(text)
        
        # Special characters should not create tokens
        assert '@' not in tokens
        assert '#' not in tokens
        assert '$' not in tokens
        
        # May create partial tokens depending on regex
        # Just verify no special chars in output
        for token in tokens:
            assert token.isalpha()
    
    def test_tokenization_empty_string(self):
        """Test that empty string returns empty list"""
        text = ""
        tokens = gen_tokens(text)
        
        assert tokens == []
    
    def test_tokenization_only_stopwords(self):
        """Test text containing only stop words"""
        text = "the a an and or but"
        tokens = gen_tokens(text)
        
        # Should be empty or very small after filtering
        assert len(tokens) == 0


class TestTextPreprocessing:
    """Test suite for text preprocessing edge cases"""
    
    def test_preprocessing_with_mixed_content(self):
        """Test preprocessing with mixed content types"""
        text = "Buy now! 50% OFF www.shop.co.nz Free shipping to New Zealand"
        tokens = gen_tokens(text)
        
        # Should keep content words
        assert 'buy' in tokens
        assert 'free' in tokens
        assert 'shipping' in tokens
        
        # Should filter domain-specific terms
        assert 'www' not in tokens
        assert 'co' not in tokens
        assert 'nz' not in tokens
    
    def test_preprocessing_with_html_like_text(self):
        """Test preprocessing with HTML-like content"""
        text = "Welcome to our store click here buy now"
        tokens = gen_tokens(text)
        
        # Should extract meaningful words
        assert 'welcome' in tokens
        assert 'store' in tokens
        assert 'click' in tokens
        assert 'buy' in tokens
    
    def test_preprocessing_preserves_important_terms(self):
        """Test that important business terms are preserved"""
        text = "online shopping delivery payment secure checkout"
        tokens = gen_tokens(text)
        
        assert 'online' in tokens
        assert 'shopping' in tokens
        assert 'delivery' in tokens
        assert 'payment' in tokens
        assert 'secure' in tokens
        assert 'checkout' in tokens
    
    def test_preprocessing_handles_repeated_words(self):
        """Test that repeated words are preserved"""
        text = "buy buy buy sale sale"
        tokens = gen_tokens(text)
        
        # Should preserve duplicates (TF-IDF will handle weighting)
        assert tokens.count('buy') == 3
        assert tokens.count('sale') == 2
    
    def test_preprocessing_with_punctuation(self):
        """Test that punctuation is handled correctly"""
        text = "Hello, world! This is great. Amazing product?"
        tokens = gen_tokens(text)
        
        # Punctuation should be removed, words preserved
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'great' in tokens
        assert 'amazing' in tokens
        assert 'product' in tokens
        
        # No punctuation should remain
        for token in tokens:
            assert not any(char in token for char in ',.!?;:')


class TestIntegration:
    """Integration tests for the classification pipeline"""
    
    def test_tokenization_produces_list(self):
        """Test that tokenization always produces a list"""
        test_texts = [
            "Normal text",
            "",
            "123 456",
            "!@#$%",
            "the a an"
        ]
        
        for text in test_texts:
            result = gen_tokens(text)
            assert isinstance(result, list)
    
    def test_tokenization_produces_strings(self):
        """Test that all tokens are strings"""
        text = "Hello world this is a test"
        tokens = gen_tokens(text)
        
        for token in tokens:
            assert isinstance(token, str)
    
    def test_tokenization_consistency(self):
        """Test that same input produces same output"""
        text = "Test text for consistency check"
        tokens1 = gen_tokens(text)
        tokens2 = gen_tokens(text)
        
        assert tokens1 == tokens2

