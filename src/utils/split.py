from sklearn.model_selection import train_test_split

def stratified_split(df):
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=1/3, stratify=temp["label"], random_state=42
    )
    return train, val, test