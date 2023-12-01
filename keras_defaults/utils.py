import matplotlib.pyplot as plt

def print_history(history):
    x_axis = range(1, history.params['epochs'] + 1)
    fig, ax1 = plt.subplots()

    colour = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=colour)
    ax1.plot(x_axis, history.history['loss'], color=colour, label="train loss")
    ax1.plot(x_axis, history.history['val_loss'], color='tab:orange', label="validation loss")
    ax1.tick_params(axis='y', labelcolor=colour)
    ax1.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    colour = 'tab:green'
    ax2.set_ylabel('accuracy', color=colour)  # we already handled the x-label with ax1
    ax2.plot(x_axis, history.history['accuracy'], color=colour, label="train accuracy")
    ax2.plot(x_axis, history.history['val_accuracy'], color='tab:blue', label="validation accuracy")
    ax2.tick_params(axis='y', labelcolor=colour)
    ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()