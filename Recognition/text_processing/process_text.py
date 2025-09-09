"""
Text processing for combining characters and tones
"""

from collections import Counter
from ..utils.config import (
    TONE_MAP, SPECIAL_CHARACTER_REPLACE, SPECIAL_CHARACTER_BLOCK, 
    VALID_BEFORE, CLASSES, CHAR_DISPLAY_MAP, BASE_CHAR_MAP
)
from ..utils.recognition_utils import special_characters_prediction


class TextProcessor:
    """Processes and combines characters and tones into complete text"""
    
    def __init__(self):
        """Initialize text processor"""
        self.sentence = ""
        self.current_word = ""
        self.just_processed_character = False
        
        # Cache for performance
        self._display_text_cache = ""
        self._full_text_cache = ""
        self._cache_dirty = True
    
    def apply_tone(self, char, tone):
        """Apply tone to a character"""
        return TONE_MAP.get(char.upper(), {}).get(tone, char)
    
    def most_common_value(self, lst):
        """Get the most common value from a list"""
        if not lst:
            return None
        return Counter(lst).most_common(1)[0][0]
    
    def process_character(self, raw_character):
        """
        Process and add character to current word
        
        Args:
            raw_character (str): Raw character from classifier
            
        Returns:
            bool: True if character was processed successfully
        """
        character = special_characters_prediction(
            self.sentence + self.current_word, raw_character
        )
        
        if not character:
            return False
            
        mapped_character = CHAR_DISPLAY_MAP.get(character, character)
        
        if self.current_word:
            last_char = self.current_word[-1]
            
            # Check base character to keep accented characters
            last_base_char = BASE_CHAR_MAP.get(last_char, last_char)
            current_base_char = BASE_CHAR_MAP.get(mapped_character, mapped_character)
            if last_base_char == current_base_char:
                return True  # Keep current_word, don't add new character
            
            # Special character logic
            if self._should_skip_character(last_char, mapped_character, raw_character):
                return False
            
            # Handle special character replacement
            if self._should_replace_character(raw_character, last_char):
                self.current_word = (
                    self.current_word[:-1] + 
                    SPECIAL_CHARACTER_REPLACE[raw_character][last_char]
                )
                self._cache_dirty = True
                return True
            
            # Handle Đ character
            if self._should_process_d_character(last_char, mapped_character):
                return True
            
            # Add new character
            self.current_word += mapped_character
            self._cache_dirty = True
        else:
            # First character of word
            if mapped_character not in ["Mu", "Munguoc", "Rau"]:
                self.current_word = mapped_character
                self._cache_dirty = True
        
        return True
    
    def _should_skip_character(self, last_char, mapped_character, raw_character):
        """Check if character should be skipped"""
        # Skip if character is same as last (except special characters)
        if (last_char == mapped_character and 
            mapped_character not in ["Mu", "Munguoc", "Rau"]):
            return True
        
        # Check special characters Mu, Munguoc, Rau
        if (mapped_character == "Mu" and 
            last_char not in VALID_BEFORE["Mu"]):
            return True
        
        if (mapped_character == "Munguoc" and 
            last_char not in VALID_BEFORE["Munguoc"]):
            return True
        
        if (mapped_character == "Rau" and 
            last_char not in VALID_BEFORE["Rau"]):
            return True
        
        # Check block characters
        if (last_char in SPECIAL_CHARACTER_BLOCK and 
            raw_character in SPECIAL_CHARACTER_BLOCK[last_char]):
            return True
        
        # Check valid before
        if (mapped_character in VALID_BEFORE and 
            last_char not in VALID_BEFORE[mapped_character]):
            return True
        
        return False
    
    def _should_replace_character(self, raw_character, last_char):
        """Check if character should be replaced"""
        return (raw_character in SPECIAL_CHARACTER_REPLACE and 
                last_char in SPECIAL_CHARACTER_REPLACE[raw_character])
    
    def _should_process_d_character(self, last_char, mapped_character):
        """Process D and Đ characters"""
        if last_char == "D" and mapped_character == "Đ":
            self.current_word = self.current_word[:-1] + "Đ"
            return True
        
        if last_char == "Đ" and mapped_character in ["D", "DD"]:
            return True  # Skip
        
        return False
    
    def apply_tone_to_word(self, tone):
        """Apply tone to the last character of current word"""
        if self.current_word:
            last_char = self.current_word[-1]
            new_char = self.apply_tone(last_char, tone)
            self.current_word = self.current_word[:-1] + new_char
            self._cache_dirty = True
    
    def finalize_word(self):
        """Add current word to sentence"""
        if self.current_word:
            self.sentence += self.current_word + " "
            self.current_word = ""
            self._cache_dirty = True
    
    def clear_text(self):
        """Clear all text"""
        self.sentence = ""
        self.current_word = ""
        self._cache_dirty = True
    
    def delete_last_word(self):
        """
        Delete the last character from current word or sentence
        
        Returns:
            bool: True if something was deleted
        """
        deleted = False
        
        if self.current_word and len(self.current_word) > 0:
            # If there's current word, delete last character
            self.current_word = self.current_word[:-1]
            self._cache_dirty = True
            deleted = True
        elif self.sentence and len(self.sentence) > 0:
            # If there's sentence, delete last character (ignore trailing space)
            sentence = self.sentence.rstrip()  # Remove trailing space
            if len(sentence) > 0:
                # Delete last character of sentence (not space)
                self.sentence = sentence[:-1] + " "  # Keep space
            else:
                # If sentence only has space, clear it
                self.sentence = ""
            self._cache_dirty = True
            deleted = True
            
        return deleted
    
    def get_display_text(self):
        """Get formatted display text with caching"""
        if self._cache_dirty:
            display_text = ""
            for char in self.sentence + self.current_word:
                display_text += CHAR_DISPLAY_MAP.get(char, char)
            self._display_text_cache = display_text.strip()
            self._cache_dirty = False
        return self._display_text_cache
    
    def get_full_text(self):
        """Get full text including current word with caching"""
        if self._cache_dirty:
            self._full_text_cache = self.sentence + self.current_word
            self._cache_dirty = False
        return self._full_text_cache
    
    def get_current_word(self):
        """Get current word being formed"""
        return self.current_word
    
    def get_sentence(self):
        """Get completed sentence"""
        return self.sentence.strip()
    
    def has_current_word(self):
        """Check if there's a current word being formed"""
        return len(self.current_word) > 0
