# minecraft-physical-actions
I'm working on a Python program that translates **physical** movements to **in-game** movements.

![app_desc](https://user-images.githubusercontent.com/78581216/229394241-3614790b-d210-4205-a595-7359bd06b4f6.png)


The program is based off of Nicholas Renotte's **tensorflow** Action Detection model built with LSTM layers.

Link to tutorial: https://youtu.be/doDUihpj6ro

I modified the code to prepare my own training set.

**So far**, the program is clunky. The model occasionally returns the wrong action detected and the implementation of invoking the correct actions still needs to be debugged to properly function. But we're getting there :)

## Functionality
The main program loop turns on the user's webcam with cv2 and tracks hand movement using mediapipe holistic. It defines the layers of the tensorflow model and loads trained weights. Every 30 sequences of frames gets passed to the model to predict an action. If the results threshold is over 0.4, an action was detected. The action string then gets passed to a function that invokes the corresponding keyboard/mouse action via pyautogui.

So far, the model is trained with the following actions:

```
punch
placeblock
run
```

When training the model, I found that it begins to converge at approx. 180 epochs. Any more than that and it will start overfitting.

## How to get started
Install required dependencies.
```
pip install -r requirements.txt
```
To train using your own dataset, prepare the MP_Data folder by uncommenting out the prepare_folders() line in prepare_data.py

```
# prepare_folders() # uncomment this out if you haven't created the MP_Data folder yet
```
Uncomment it out again and run prepare_data.py. It'll prompt you to act out the required actions to be trained. More information on Nicholas' video.

Once the tranining data is prepared, run train.py. It will start training the model with the dataset you've prepared. I set the epochs to max epochs to 180 since again, it converges around that number. But your mileage may vary so change it accordingly!

If your model is finished training, run detect.py as well as Minecraft. It'll open up your webcam and will start observing your hand movements.
