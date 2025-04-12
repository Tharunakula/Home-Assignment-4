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


**Q2: Named Entity Recognition with SpaCy**

**1.import spacy:**  This line imports the spaCy library, which is essential for natural language processing tasks in Python.

**2.nlp = spacy.load("en_core_web_sm"):**

    * This loads a pre-trained English language model provided by spaCy.
    * "en_core_web_sm" specifies the model to load.
      * "en": Indicates the language is English.
      * "core":  Indicates a general-purpose model.
      * "web":  Indicates the model is trained on web text.
      * "sm":   Indicates it's a small-sized model (for efficiency).  spaCy offers models in different sizes (small, medium, large).  Larger models are generally 
         more accurate but slower.
    * The loaded model (nlp) is an object that can then be used to process text.

**3.sentence = "...":** This line defines the input text string that you want to analyze.

**4.doc = nlp(sentence):**

    * This is where the actual processing happens.
    * You pass the input sentence to the nlp object (the loaded language model).
    * The nlp object analyzes the sentence, performing tokenization, part-of-speech tagging, and, importantly, named entity recognition.
    * The result is stored in a Doc object (named doc here).  The Doc object contains all the linguistic information about the sentence.

**5.for ent in doc.ents::**

    * This loop iterates over the named entities that spaCy detected in the processed text.
    * doc.ents is a sequence of Span objects, each representing a single named entity.

**6.print(f"Entity: {ent.text}, Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}"):**  Inside the loop, this line prints information about each named entity:

    * ent.text:  The text of the named entity itself (e.g., "Barack Obama").
    * ent.label_: The type of named entity (e.g., "PERSON", "GPE" for geopolitical entity, "DATE").  spaCy's pre-trained models come with a set of predefined 
      labels.
    * ent.start_char: The starting character index of the entity in the original sentence.
    * ent.end_char: The ending character index of the entity in the original sentence.
