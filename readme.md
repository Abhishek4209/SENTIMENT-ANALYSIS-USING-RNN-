## **SENTIMENT ANALYSIS ON IMDB DATA SETS USING RNN MODEL**

## RNN BY ABHISHEK UPADHYAY:-
**RNN HAND ON :-**




### Model Architecture:-
```bash
model = Sequential()
model.add(SimpleRNN(32, input_shape = (500,1), return_sequences= False))
model.add(Dense(1, activation = 'sigmoid'))

```
















### Callback function:-
```bash
EarlyStopping(
     monitor="val_loss",
    min_delta=0,
    patience=8,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)
```