import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Multiply
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


pred_init = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=3721)

# This MoE model require to Input matrix: 1, Original Data, (N, D = 5); 2, Gamma Matrix, (N, K = 4) Gate control.
# The data will be complie into one dataset.
def dataset_prepare(X, Y, gamma, window_size, batch_size):
    '''
    Arg: 
        X: Data, the dependent variable (N,D)
        Y: Label(0, 1), the Object of prediction (N,1)
        gamma: Gamma Matrix, Calculated from GMM (N, K)
        window_size: The width of time series for LSTM to use, predicting the next day probability of going up
        batch_size: Batch size in training

    Output:
        dataset: Intergreted Dataset. For MoE Input 
    '''
    # Time Series Set
    ts_set = tf.keras.utils.timeseries_dataset_from_array(
        data = X[:-1, :],
        targets = Y[window_size:],
        sequence_length = window_size,
        batch_size = batch_size
    )
    # Gamma Matrix Set
    gamma_aligned = gamma[window_size-1: -1, :]
    gamma_set = tf.data.Dataset.from_tensor_slices(gamma_aligned).batch(batch_size)

    dataset = tf.data.Dataset.zip((ts_set, gamma_set)).map(
        lambda ts_y, gamma :(
            {'TimeSeries_Input': ts_y[0], 'GMM_Gate_Input': gamma}, ts_y[1]
        )
    )
    return dataset

def MoE_model(window_size, num_features, K, LSTM_units = 16, dropout = 0.3):
    '''
    Arg:
        window_size: The width of LSTM
        num_features: number of features of time series data
        K: K of GMM/ gamma matrix
        LSTM_units: Default 16
        dropout: Rate of Dropout, default 0.3

    Output:
        model: MoE model 
    '''
    ts_input = Input(shape = (window_size, num_features), name = 'TimeSeries_Input')
    gate_input = Input(shape = (K,), name = 'GMM_Gate_Input')
    expert_outputs = []

    for i in range(K): 
        expert_lstm = LSTM(
            units=16, dropout = 0.3,
            kernel_initializer='glorot_uniform', 
            recurrent_initializer='orthogonal',  
            bias_initializer='zeros',
            activation='tanh', 
            kernel_regularizer=tf.keras.regularizers.l2(0.005), 
            name=f"Expert_{i}_LSTM"
        )(ts_input)
        
        expert_pred = Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.0001), kernel_initializer= pred_init, name=f"Expert_{i}_Output")(expert_lstm)
        
        expert_outputs.append(expert_pred)

    merged_experts = tf.keras.layers.Concatenate(axis = -1, name = 'MergeExperts')(expert_outputs)

    weighted_experts = Multiply(name="Gate_Multiply")([merged_experts, gate_input])

    final_output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name="Final_Sum_Output")(weighted_experts)

    model = tf.keras.models.Model(inputs=[ts_input, gate_input], outputs=final_output, name="MoE_SP500_Predictor")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    return model

def test_set_eval(model, test_set):
    test_loss, test_acc = model.evaluate(test_set, verbose=1)
    print(f"\n Test set accuracy: {test_acc * 100:.2f}%")

    pred_probs = model.predict(test_set)
    pred_classes = (pred_probs > 0.5).astype(int).flatten()

    true_labels = []
    for _, y in test_set:
        true_labels.extend(y.numpy())
    true_labels = np.array(true_labels).flatten()

    cm = confusion_matrix(true_labels, pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['DOWN (0)', 'UP (1)'])


    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d') 

    plt.title(f'MoE Test Set Confusion Matrix (Acc: {test_acc*100:.2f}%)', fontsize=14, fontweight='bold')
    plt.show()

    print("\n📊 Classification Report:")
    print(classification_report(true_labels, pred_classes, target_names=['DOWN (0)', 'UP (1)']))
    return None