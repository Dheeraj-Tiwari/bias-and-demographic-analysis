import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re  # For regular expressions (for bias detection)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# NLTK Downloads (Check before downloading)
try:
    nltk.data.find('tokenizers/punkt_tab/english.pickle')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt/english.pickle')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords/english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores

def generate_summary(text, num_sentences=3, bias_keywords=None, demographic_groups=None):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [w.lower() for w in words if w.isalnum() and w.lower() not in stop_words]

    word_frequencies = {}
    for word in filtered_words:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[i] = sentence_scores.get(i, 0) + word_frequencies[word]

    sorted_scores = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences_indices = [index for index, score in sorted_scores[:num_sentences]]
    summary_sentences = [sentences[i] for i in top_sentences_indices]

    summary = " ".join(summary_sentences)

    bias_info = {}
    if bias_keywords:
        for keyword in bias_keywords:
            occurrences = len(re.findall(r'\b' + keyword + r'\b', summary.lower()))
            if occurrences > 0:
                bias_info[keyword] = {"occurrences": occurrences, "sentiment": {}}
                for sentence in sent_tokenize(summary):
                    if keyword in sentence.lower():
                        sentiment = analyze_sentiment(sentence)
                        bias_info[keyword]["sentiment"][sentence] = sentiment

    demographic_info = {}
    if demographic_groups:
        for group, keywords in demographic_groups.items():
            group_occurrences = 0
            group_sentiment = {}
            for keyword in keywords:
                occurrences = len(re.findall(r'\b' + keyword + r'\b', summary.lower()))
                group_occurrences += occurrences
                for sentence in sent_tokenize(summary):
                    if keyword in sentence.lower():
                        sentiment = analyze_sentiment(sentence)
                        group_sentiment[sentence] = sentiment
            if group_occurrences > 0:
              demographic_info[group] = {"occurrences": group_occurrences, "sentiment": group_sentiment}

    return summary, bias_info, demographic_info


# Example Usage
topics = {
    "gender_bias": "The scientist, she worked tirelessly in the lab. She was dedicated to her research. The engineer, he was also very skilled. He designed innovative solutions.  The nurse, she was kind and caring.",
    "race_bias": "The hardworking man arrived from his native country. He quickly adapted. The lazy man stayed back. He didn't work hard. The successful businessman was white.",
    "neutral_topic": "Photosynthesis is the process by which plants convert light energy into chemical energy. This process is crucial for plant growth."
}

bias_keywords_gender = ["she", "her", "he", "him", "scientist", "engineer", "nurse"]
bias_keywords_race = ["native", "lazy", "hardworking", "man", "businessman"]

demographic_groups = {
    "women": ["she", "her", "woman", "female", "nurse"],
    "men": ["he", "him", "man", "male", "engineer", "businessman"],
    "people_of_color": ["native", "immigrant", "person of color", "non-white"],
    "white_people": ["white", "caucasian"]
}

for topic, text in topics.items():
    if topic == "gender_bias":
        summary, bias_info, demographic_info = generate_summary(text, bias_keywords=bias_keywords_gender, demographic_groups=demographic_groups)
    elif topic == "race_bias":
        summary, bias_info, demographic_info = generate_summary(text, bias_keywords=bias_keywords_race, demographic_groups=demographic_groups)
    else:
        summary, bias_info, demographic_info = generate_summary(text, demographic_groups=demographic_groups)

    print(f"Summary of {topic}:\n{summary}")

    if bias_info:
        for keyword, info in bias_info.items():
            print(f"  Bias Keyword: {keyword}")
            print(f"    Occurrences: {info['occurrences']}")
            for sentence, sentiment in info['sentiment'].items():
                print(f"    Sentence: {sentence}")
                print(f"      Sentiment: {sentiment}")
                print(f"      Compound: {sentiment['compound']}")
                print(f"      Positive: {sentiment['pos']}")
                print(f"      Negative: {sentiment['neg']}")
                print(f"      Neutral: {sentiment['neu']}")

    else:
        print("  No bias keywords found.")

    if demographic_info:
        for group, info in demographic_info.items():
            print(f"  Demographic Group: {group}")
            print(f"    Occurrences: {info['occurrences']}")
            for sentence, sentiment in info["sentiment"].items():
                print(f"    Sentence: {sentence}")
                print(f"      Sentiment: {sentiment}")
                print(f"      Compound: {sentiment['compound']}")
                print(f"      Positive: {sentiment['pos']}")
                print(f"      Negative: {sentiment['neg']}")
                print(f"      Neutral: {sentiment['neu']}")
    else:
        print("  No demographic information found.")

    print("-" * 20)