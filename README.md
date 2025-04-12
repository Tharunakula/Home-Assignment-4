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

**Q3: Scaled Dot-Product Attention**

**1.softmax(x) Function:**

    * Computes the softmax function in a numerically stable way by subtracting the maximum value before exponentiation. This prevents potential overflow issues.
    * Normalizes the exponentiated values into a probability distribution.

**2.scaled_dot_product_attention(Q, K, V) Function:**

    * Takes the Query (Q), Key (K), and Value (V) matrices as input.
    * Calculates the dot product of Q and the transpose of K (K.T) to measure the similarity between each query and key.
    * Scales the result by the square root of the key dimension (d) to prevent the dot products from becoming too large.
    * Applies the softmax function to the scaled dot products to obtain the attention weights. These weights represent the importance of each key-value pair for 
      each query.
    * Computes the weighted sum of the values (V) using the attention weights.

**3.Test Case:**

    * Provides example Q, K, and V matrices to demonstrate the usage of the scaled_dot_product_attention function.
    * Calls the function with these inputs and prints the resulting attention weights and output.


**Q4: Sentiment Analysis using HuggingFace Transformers**


**1.Import pipeline:**

    * Imports the pipeline function from the transformers library.
    * The pipeline function simplifies the process of using pre-trained models for various NLP tasks.

**2.Load sentiment analysis pipeline:**

    * Creates a sentiment analysis pipeline using pipeline("sentiment-analysis").
    * This downloads and loads a pre-trained model suitable for sentiment analysis (e.g., DistilBERT, RoBERTa).
    * The loaded pipeline is assigned to sentiment_pipeline.

**3.Input sentence:**

    * Defines the sentence to be analyzed: "Despite the high price, the performance of the new MacBook is outstanding."

**4.Perform sentiment analysis:**

    * Passes the input sentence to sentiment_pipeline to get the sentiment prediction.
    * The result is a list, and [0] retrieves the prediction for the first (and only) sentence.
    * The prediction is a dictionary with keys like "label" (e.g., "POSITIVE", "NEGATIVE") and "score" (confidence).

**5.Print the result:**

    * Prints the predicted sentiment label and its confidence score, formatted to four decimal places.
