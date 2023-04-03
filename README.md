# minecraft-actions
I made a Python program that translates **physical** movements to **in-game** movements.

The program is based off of Nicholas Renotte's **tensorflow** Action Detection model with LSTM layers

Link to tutorial: https://www.youtube.com/watch?v=doDUihpj6ro&t=959s

I modified the code to prepare my own training set.

So far, the model is trained with the following actions:

```
punch
placeblock
run
```

When training the model, I found that it begins to converge at approx. 180 epochs. Any more than that and it will start overfitting.


To train using your own dataset, prepare the MP_Data folder by uncommenting out the prepare_folders() line in prepare_data.py

```
# prepare_folders() # uncomment this out if you haven't created the MP_Data folder yet
```

Also install the required dependencies if you haven't already
```
pip install -r requirements.txt
```
