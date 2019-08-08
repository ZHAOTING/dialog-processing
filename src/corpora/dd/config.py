class Config(object):
    def __init__(self, task):
        self.dialog_acts = ["<pad>", "inform", "question", "directive", "commissive"]
        self.emotions = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
        self.topics = ["Ordinary Life", "School Life", "Culture & Education", "Attitude & Emotion", "Relationship", "Tourism", "Health", "Work", "Politics", "Finance"]
        self.vocab_size = 10000

        ## Management
        self.raw_data_dir = "../data/dd/raw_data"

        ## Task-specific management
        self.task = task
        self.task_data_dir = f"../data/dd/{task}"
        self.dataset_path = f"{self.task_data_dir}/dataset.txt"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"

        ## Pretrained embeddings (for initialization and evaluation)
        self.word_embedding_path = f"{self.task_data_dir}/glove_twitter_200.json"
        self.eval_word_embedding_path = f"{self.task_data_dir}/google_news_300.json"
