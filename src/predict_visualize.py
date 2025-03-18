import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(y_test, y_pred, stock_symbol):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label="Actual Prices", color="blue")
    plt.plot(np.arange(len(y_pred)), y_pred, label="Predicted Prices", color="red")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title(f"Actual vs Predicted Prices for {stock_symbol}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/{stock_symbol}_prediction.png')
    plt.show()
    print(f"Visualization saved as visualizations/{stock_symbol}_prediction.png")
