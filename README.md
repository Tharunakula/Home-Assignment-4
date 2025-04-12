**CS5720 Neural network and Deep learning,**

**CRN:-23848**

**Name:- Akula Tharun.**

**student ID:-700759411.**

**Home Assignment 4.**

**Q1 NLP Preprocessing Pipeline**

**1.Import necessary libraries:**

      * nltk: The Natural Language Toolkit library, which provides various NLP functionalities.
      * stopwords: A list of common English stopwords from nltk.corpus.
      * PorterStemmer: An algorithm for stemming words from nltk.stem.
      * word_tokenize: A function to split a sentence into individual words (tokens) from nltk.tokenize.

      
**2.Define the preprocess_sentence function:**

     * Takes a sentence as input.
     * Tokenization:
        * word_tokenize(sentence) splits the input sentence into a list of individual words and punctuation marks.
        * The resulting tokens list is printed.
     * Stop Word Removal:
        * stopwords.words('english') retrieves a set of common English stopwords. Using a set allows for faster lookups.
        * A list comprehension [w for w in tokens if not w.lower() in stop_words] iterates through the tokens. It keeps a word (w) only if its lowercase version is 
          not present in the stop_words set. This ensures case-insensitive stop word removal.
        * The filtered_tokens list (without stopwords) is printed.
     * Stemming:
        * PorterStemmer() creates an instance of the Porter stemming algorithm.
        * Another list comprehension [porter.stem(word) for word in filtered_tokens] iterates through the filtered_tokens and applies the porter.stem() method to 
          each word to get its root form.
        * The stemmed_words list is printed.

        
**3.Example Usage:**

     * The provided sentence "NLP techniques are used in virtual assistants like Alexa and Siri." is assigned to the sentence variable.
     * The preprocess_sentence() function is called with this sentence to demonstrate the preprocessing steps.
