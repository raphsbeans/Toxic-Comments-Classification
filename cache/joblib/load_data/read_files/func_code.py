# first line: 49
@MEMORY.cache
def read_files(fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only support all lower case
        lower = True
    LOGGER.debug("Preparing CSV files...")
    prepare_csv()
    comment = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        dtype=torch.LongTensor,
        lower=lower
    )
    LOGGER.debug("Reading train csv file...")
    train = data.TabularDataset(
        path='cache/dataset_train.csv', format='csv', skip_header=True,
        fields=[
            ('id', None),
            ('comment_text', comment),
            ('toxic', data.Field(
                use_vocab=False, sequential=False, dtype=torch.ByteTensor)),
            ('severe_toxic', data.Field(
                use_vocab=False, sequential=False, dtype=torch.ByteTensor)),
            ('obscene', data.Field(
                use_vocab=False, sequential=False, dtype=torch.ByteTensor)),
            ('threat', data.Field(
                use_vocab=False, sequential=False, dtype=torch.ByteTensor)),
            ('insult', data.Field(
                use_vocab=False, sequential=False, dtype=torch.ByteTensor)),
            ('identity_hate', data.Field(
                use_vocab=False, sequential=False, dtype=torch.ByteTensor)),
        ])
    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path='cache/dataset_test.csv', format='csv', skip_header=True,
        fields=[
            ('id', None),
            ('comment_text', comment)
        ])
    LOGGER.debug("Building vocabulary...")
    comment.build_vocab(
        train, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    LOGGER.debug("Done preparing the datasets")
    return train.examples, test.examples, comment
