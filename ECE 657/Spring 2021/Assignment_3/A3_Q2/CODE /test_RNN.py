if __name__ == "__main__":
    # 1. Load your saved model
    model_path = "models/20923853_RNN_model.h5"
    model = load_model(model_path)

    # 2. Load your testing data
    # load test data from CSV
    test_path = "data/test_data_RNN.csv"
    test_labels_path = "data/train_data_RNN_labels.csv"
    csvArray = np.loadtxt(test_path)
    csvArray2 = np.loadtxt(test_labels_path)

    # This csvArray is a 2D array, therefore we need to convert it to the original array shape.
    X_shape = np.zeros(shape=(377, 3, 4))
    y_shape = np.zeros(shape=(377, 2))

    # reshaping to get original matrice with original shape.
    X_test = csvArray.reshape(
        csvArray.shape[0], csvArray.shape[1] // X_shape.shape[2], X_shape.shape[2])

    y_test = np.asarray(csvArray2[:, 0]).astype('float32')
    index_list = csvArray2[:, 1]

    # 3. Run prediction on the test data and output required plot and loss

    prediction = model.predict(X_test)
    pred_y = []
    for item in prediction:
        pred_y.extend(item)

    plt.scatter(index_list, y_test, s=2)
    plt.scatter(index_list, pred_y, s=2)
    plt.legend(["Actual", "Prediction"])
    plt.xlabel('Date (Indexed)')
    plt.ylabel('Price')
    plt.title('Stock Price Vs. Time')
    plt.show()
