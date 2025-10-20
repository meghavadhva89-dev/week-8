from collections import defaultdict


class MarkovText(object):

    def __init__(self, corpus, k=1):
        self.corpus = corpus
        self.k = k  # window size for k-word states
        self.term_dict = None  # built by get_term_dict()

    def get_term_dict(self):
        """
        Build a term dictionary of Markov states.
        
        For k=1 (default):
            Keys: unique tokens in the corpus
            Values: list of all tokens that follow each key
            
        For k>1:
            Keys: tuples of k consecutive tokens
            Values: list of all tokens that follow each k-word sequence
        
        Note: Duplicates ARE included in the lists. This is important
        because it preserves the frequency information - if a word
        follows another word more often, it should appear multiple times
        in the list. This allows random.choice() to naturally select
        more frequent successors with higher probability, creating more
        realistic text generation.
        """
        # Tokenize the corpus by splitting on whitespace
        tokens = self.corpus.split()
        
        # Create a defaultdict to automatically handle new keys
        term_dict = defaultdict(list)
        
        # Iterate through tokens, building the dictionary
        # Stop at len(tokens)-k to avoid index out of bounds
        for i in range(len(tokens) - self.k):
            # Create the key (single word for k=1, tuple for k>1)
            if self.k == 1:
                current_state = tokens[i]
            else:
                current_state = tuple(tokens[i:i + self.k])
            
            # Get the next token
            next_token = tokens[i + self.k]
            
            # Add to dictionary
            term_dict[current_state].append(next_token)
        
        # Convert defaultdict to regular dict and store it
        self.term_dict = dict(term_dict)
        
        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        """
        Generate text using the Markov chain property.
        
        Args:
            seed_term: Optional starting word(s). For k=1, a string.
                For k>1, a tuple of k words. If None, picks random
                state from corpus.
            term_count: Number of words to generate (default 15)
            
        Returns:
            A string of generated text
            
        Raises:
            ValueError: If seed_term is not in the corpus
        """
        import random
        
        # Make sure term_dict is built
        if self.term_dict is None:
            self.get_term_dict()
        
        # Handle seed term
        if seed_term is None:
            # Pick a random state from the dictionary keys
            current_state = random.choice(list(self.term_dict.keys()))
        else:
            # For k>1, convert string to tuple if needed
            if self.k > 1 and isinstance(seed_term, str):
                raise ValueError(
                    f"For k={self.k}, seed_term must be a tuple "
                    f"of {self.k} words"
                )
            # Validate that seed_term exists in corpus
            if seed_term not in self.term_dict:
                raise ValueError(
                    f"Seed term '{seed_term}' not found in corpus"
                )
            current_state = seed_term
        
        # Start building the generated text
        if self.k == 1:
            generated_words = [current_state]
        else:
            generated_words = list(current_state)
        
        # Generate the remaining words
        words_to_generate = term_count - len(generated_words)
        for _ in range(words_to_generate):
            # Check if current state has any following words
            if (current_state not in self.term_dict or
                    len(self.term_dict[current_state]) == 0):
                # No more words to follow, stop generation
                break
            
            # Randomly choose next word from possible next words
            next_word = random.choice(self.term_dict[current_state])
            generated_words.append(next_word)
            
            # Update current state for next iteration
            if self.k == 1:
                current_state = next_word
            else:
                # For k>1, create new state from last k words
                current_state = tuple(generated_words[-self.k:])
        
        # Join all words into a single string
        return ' '.join(generated_words)
