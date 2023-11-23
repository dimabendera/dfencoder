"""
EXAMPLE RUN:
 python3 scripts/train_example.py

 TEST 1: validation accuracy:  0.82
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

sys.path.append(os.getcwd())


from dfencoder import AutoEncoder

sns.set()


class ClassifierModel(torch.nn.Module):
    """A simple classifier neural network."""

    def __init__(self, *args, **kwargs):
        super(ClassifierModel, self).__init__(*args, **kwargs)
        self.input_dropout = torch.nn.Dropout(.1)
        self.input_layer = torch.nn.Linear(96, 32)
        self.dropout = torch.nn.Dropout(.5)
        self.dense = torch.nn.Linear(32, 32)
        self.output = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)

        x = self.output(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load and look at the data
    df = pd.read_csv('./adult.csv')

    # this dataset contains nulls and ' ?'. Let's make these all nulls.
    df = df.applymap(lambda x: np.nan if x == ' ?' else x)
    print(df.head())
    print(df.shape)

    train = df.sample(frac=.8, random_state=42)
    test = df.loc[~df.index.isin(train.index)]

    X_train = train
    X_val = test

    model = AutoEncoder(
        encoder_layers=[32, 32, 32],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        activation='relu',
        swap_p=0.2,  # noise parameter
        lr=0.01,
        lr_decay=.99,
        batch_size=1024,
        logger='basic',  # special logging for jupyter notebooks
        verbose=True,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
        min_cats=3  # Define cutoff for minority categories, default 10
    )
    model.fit(X_train, epochs=10, val=X_val)

    z = model.get_deep_stack_features(X_val)
    print(z.shape)
    print(z[0, :])

    print(X_train.salary.unique())
    classifier = ClassifierModel().to(device)

    optim = torch.optim.Adam(
        classifier.parameters(),
        weight_decay=.01
    )

    decay = torch.optim.lr_scheduler.ExponentialLR(optim, .99)

    loss = torch.nn.modules.loss.BCELoss()


    def do_step(classifier, optim, z, target, loss):
        pred = classifier(z)
        target = torch.tensor(target).float().reshape(-1, 1).to(device)
        loss_ = loss(pred, target)
        amnt = loss_.item()
        loss_.backward()
        optim.step()
        optim.zero_grad()
        return amnt


    def do_evaluation(classifier, z, target, loss):
        with torch.no_grad():
            pred = classifier(z)
            probs = pred.cpu().numpy().reshape(-1)
            predictions = np.where(probs > .5, 1, 0)

            accuracy = np.where(target == predictions, 1, 0).sum() / len(predictions)
            f1 = f1_score(target, predictions)

            target_ = torch.tensor(target).float().reshape(-1, 1).to(device)
            loss_ = loss(pred, target_)
            return loss_.item(), accuracy, f1


    batch_size = 256
    n_updates = (len(X_train) // batch_size) + 1

    n_epochs = 10

    # To extract features, we'll set the target column on the input
    # equal to the majority class: <50k

    X_train2 = X_train.copy()
    X_train2['salary'] = ['<50k' for _ in X_train2['salary']]
    z_train = model.get_deep_stack_features(X_train2)

    Y_train = np.where(X_train['salary'].values == '<50k', 0, 1)

    X_test2 = X_val.copy()
    X_test2['salary'] = ['<50k' for _ in X_test2['salary']]
    z_test = model.get_deep_stack_features(X_test2)
    Y_test = np.where(X_val['salary'].values == '<50k', 0, 1)

    for j in range(n_epochs):
        if j % 100 == 0:
            print(f'{j} epochs complete...')
        for i in range(n_updates):
            step = i
            start = int((step) * batch_size)
            stop = int((step + 1) * batch_size)
            in_ = z_train[start:stop]
            target = Y_train[start:stop]
            do_step(classifier, optim, in_, target, loss)
        decay.step()
        bce_loss, accuracy, f1 = do_evaluation(classifier, z_test, Y_test, loss)
    print('\nFinal results: ')
    print('validation loss: ', round(bce_loss, 4))
    print('validation accuracy: ', round(accuracy, 3))
    print('validation f1 score: ', round(f1, 3))

