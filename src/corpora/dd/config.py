class Config(object):
    def __init__(self, task):
        self.dialog_acts = ["<pad>", "inform", "question", "directive", "commissive"]
        self.emotions = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
        self.topics = ["Ordinary Life", "School Life", "Culture & Education", "Attitude & Emotion", "Relationship", "Tourism", "Health", "Work", "Politics", "Finance"]

        # Data processing
        self.download_url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
        self.id2dialog_act = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}
        self.id2topic = {1: "Ordinary Life", 2: "School Life", 3: "Culture & Education", 4: "Attitude & Emotion", 5: "Relationship", 6: "Tourism", 7: "Health", 8: "Work", 9: "Politics", 10: "Finance"}
        self.id2emotion = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}

        # Management
        self.raw_data_dir = "../data/dd/raw_data"

        # Task-specific management
        self.task = task
        self.task_data_dir = f"../data/dd/{task}"
        self.dataset_path = f"{self.task_data_dir}/dataset.txt"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"

        # Pretrained embeddings (for initialization and evaluation)
        self.word_embedding_path = f"{self.task_data_dir}/glove_twitter_200.json"
        # self.eval_word_embedding_path = f"{self.task_data_dir}/glove_twitter_200.json"
        self.eval_word_embedding_path = f"{self.task_data_dir}/google_news_300.json"

        # Human evaluation task
        self.human_score_names = ["grammar", "fact", "content", "relevance", "overall"]
        if task == "response_eval":
            response_gen_dir = "../data/dd/response_gen"
            self.word_count_path = f"{response_gen_dir}/word_count.txt"
            self.word_embedding_path = f"{response_gen_dir}/glove_twitter_200.json"
            self.eval_word_embedding_path = f"{response_gen_dir}/glove_twitter_200.json"
