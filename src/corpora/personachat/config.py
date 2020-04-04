class Config(object):
    def __init__(self, task):
        # Data processing
        self.download_url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

        # Management
        self.raw_data_dir = "../data/personachat/raw_data"

        # Task-specific management
        self.task = task
        self.task_data_dir = f"../data/personachat/{task}"
        self.dataset_path = f"{self.task_data_dir}/dataset.txt"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"

        # Pretrained embeddings (for initialization and evaluation)
        self.word_embedding_path = f"{self.task_data_dir}/glove_twitter_200.json"
        self.eval_word_embedding_path = f"{self.task_data_dir}/glove_twitter_200.json"

        # Human evaluation task
        self.human_score_names = ["overall"]
        if task == "response_eval":
            response_gen_dir = "../data/personachat/response_gen"
            self.word_count_path = f"{response_gen_dir}/word_count.txt"
            self.word_embedding_path = f"{response_gen_dir}/glove_twitter_200.json"
            self.eval_word_embedding_path = f"{response_gen_dir}/glove_twitter_200.json"
