import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Any, Optional

def get_features_and_targets(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Відокремлює ознаки від цільової змінної та видаляє непотрібні колонки."""
    
    X = df.drop(columns=[target_col, 'Surname'], errors='ignore')
    y = df[target_col]
    return X, y

def get_train_val_split(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Розділяє дані на тренувальну та валідаційну вибірки."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def scale_numeric_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Масштабує числові ознаки за допомогою MinMaxScaler."""
    scaler = MinMaxScaler().fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler

def encode_categorical_features(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Кодує категоріальні ознаки за допомогою OneHotEncoder."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[categorical_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    # Створюємо DataFrame з закодованими даними
    train_encoded = pd.DataFrame(encoder.transform(train_df[categorical_cols]), columns=encoded_cols, index=train_df.index)
    val_encoded = pd.DataFrame(encoder.transform(val_df[categorical_cols]), columns=encoded_cols, index=val_df.index)
    
    # Об'єднуємо та видаляємо старі текстові колонки
    train_df = pd.concat([train_df.drop(columns=categorical_cols), train_encoded], axis=1)
    val_df = pd.concat([val_df.drop(columns=categorical_cols), val_encoded], axis=1)
    
    return train_df, val_df, encoder

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """
    Головна функція для повного циклу обробки сирих даних.
    Повертає: X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder
    """
    target_col = 'Exited'
    
    # 1. Розділення ознак
    X, y = get_features_and_targets(raw_df, target_col)
    
    # 2. Split
    X_train, X_val, y_train, y_val = get_train_val_split(X, y)
    
    # Визначаємо типи колонок
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes('object').columns.tolist()
    
    # 3. Кодування
    X_train, X_val, encoder = encode_categorical_features(X_train, X_val, categorical_cols)
    
    # 4. Масштабування (опціонально)
    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric_features(X_train, X_val, numeric_cols)
        
    # Список фінальних колонок
    input_cols = X_train.columns.tolist()
        
    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame, 
    scaler: Optional[MinMaxScaler], 
    encoder: OneHotEncoder,
    input_cols: List[str]
) -> pd.DataFrame:
    """
    Обробляє нові дані (наприклад, test.csv), використовуючи вже навчені скейлер та енкодер.
    
    Args:
        new_df: Сирий DataFrame з новими даними.
        scaler: Навчений MinMaxScaler (або None).
        encoder: Навчений OneHotEncoder.
        input_cols: Перелік назв колонок, які модель очікує на вході.
        
    Returns:
        pd.DataFrame: Оброблена таблиця, готова до прогнозів моделі.
    """
    # 1. Видаляємо Surname та інші зайві колонки, якщо вони є
    # Нам потрібно залишити тільки ті колонки, на яких вчився енкодер та скейлер
    X = new_df.drop(columns=['Surname'], errors='ignore').copy()
    
    # 2. Визначаємо типи колонок на основі енкодера
    categorical_cols = list(encoder.feature_names_in_)
    numeric_cols = [col for col in X.columns if col not in categorical_cols and np.issubdtype(X[col].dtype, np.number)]
    
    # 3. Кодування категоріальних ознак
    encoded_data = encoder.transform(X[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=X.index)
    
    # Об'єднуємо числові дані з новими закодованими колонками
    X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
    
    # 4. Масштабування числових ознак (якщо scaler було надано)
    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    # 5. Гарантуємо, що порядок і склад колонок точно такий самий, як у X_train
    # Якщо якихось колонок не вистачає (наприклад, після OHE), заповнюємо нулями
    for col in input_cols:
        if col not in X.columns:
            X[col] = 0
            
    return X[input_cols]