class Config(object):
    def __init__(self, task):
        self.dialog_acts = ['sd', 'b', 'sv', '%', 'aa', 'ba', 'qy', 'ny', 'fc', 'qw', 'nn', 'bk', 'fo_o_fw_"_by_bc', 'h', 'qy^d', 'bh', '^q', 'bf', 'na', 'ad', '^2', 'b^m', 'qo', 'qh', '^h', 'ar', 'ng', 'br', 'no', 'fp', 'qrr', 'arp_nd', 't3', 'oo_co_cc', 't1', 'bd', 'aap_am', '^g', 'qw^d', 'fa', 'ft']

        ## Data processing
        self.download_url = "http://compprag.christopherpotts.net/code-data/swda.zip"

        ## Management
        self.raw_data_dir = "../data/swda/raw_data"

        ## Task-specific management
        self.task = task
        self.task_data_dir = f"../data/swda/{task}"
        self.dataset_path = f"{self.task_data_dir}/dataset.txt"
        self.word_count_path = f"{self.task_data_dir}/word_count.txt"

        ## Pretrained embeddings (for initialization and evaluation)
        self.word_embedding_path = f"{self.task_data_dir}/glove_twitter_200.json"
        self.eval_word_embedding_path = f"{self.task_data_dir}/glove_twitter_200.json"
