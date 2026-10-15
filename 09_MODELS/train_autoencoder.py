import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import importlib

def train_and_save_model(train_df, val_df, feature_cols):
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    ae_module = importlib.import_module("09_MODELS.autoencoder")
    build_autoencoder = ae_module.build_autoencoder
    
    input_dim = len(feature_cols)
    model = build_autoencoder(input_dim)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    
    history = model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('09_MODELS/autoencoder.keras')
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('08_RESULTS/autoencoder_training_history.csv', index=False)
    
    return model, history
