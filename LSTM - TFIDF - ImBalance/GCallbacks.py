import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

if not os.path.exists('Losses'):
    os.makedirs('Losses')

if not os.path.exists('Times'):
    os.makedirs('Times')

if not os.path.exists('Checkpoints'):
    os.makedirs('Checkpoints')

class PlotLosses(Callback):
    def __init__(self):
        super().__init__()
        self.tr_acc = []
        self.tr_loss = []
        self.val_acc = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        # Mendapatkan history dari training
        self.tr_acc.append(logs.get('accuracy'))
        self.tr_loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

        # Menentukan epoch terbaik untuk loss dan accuracy
        index_loss = np.argmin(self.val_loss)
        val_lowest = self.val_loss[index_loss]
        index_acc = np.argmax(self.val_acc)
        acc_highest = self.val_acc[index_acc]
        
        Epochs = [i+1 for i in range(len(self.tr_acc))]
        loss_label = f'best epoch= {str(index_loss + 1)}'
        acc_label = f'best epoch= {str(index_acc + 1)}'
        
        plt.figure(figsize=(20, 8))
        plt.style.use('fivethirtyeight')

        plt.subplot(1, 2, 1)
        plt.plot(Epochs, self.tr_loss, 'r', label='Training loss')
        plt.plot(Epochs, self.val_loss, 'g', label='Validation loss')
        plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(Epochs, self.tr_acc, 'r', label='Training Accuracy')
        plt.plot(Epochs, self.val_acc, 'g', label='Validation Accuracy')
        plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'Losses/plot_epoch_{epoch+1}.png')
        plt.close()

# Callback to save the model at each epoch
checkpoint_callback = ModelCheckpoint(
    filepath='Checkpoints/model_epoch_{epoch:02d}.h5',  # Nama file model yang disimpan setiap epoch
    save_weights_only=True, # Jika ingin menyimpan seluruh arsitektur model. Ganti menjadi False
    save_freq='epoch'
)

# Define the custom callback for timing
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)
        print(f" Epoch {epoch + 1}: Training time: {self.times[-1]:.2f} seconds")

class PlotTimes(Callback):
    def __init__(self, time_history_callback):
        super().__init__()
        self.time_history_callback = time_history_callback

    def on_epoch_end(self, epoch, logs=None):
        Epochs = [i + 1 for i in range(len(self.time_history_callback.times))]
        plt.figure(figsize=(10, 5))
        plt.style.use('fivethirtyeight')

        plt.plot(Epochs, self.time_history_callback.times, 'b', label='Training Time')
        plt.title('Training Time per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Time (seconds)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'Times/plot_epoch_{epoch+1}.png')
        plt.close()
