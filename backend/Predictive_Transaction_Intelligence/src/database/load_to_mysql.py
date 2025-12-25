import pandas as pd
from mysql_connection import get_mysql_connection

def load_csv_to_mysql(csv_path):
    df = pd.read_csv(csv_path)

    conn = get_mysql_connection()
    cursor = conn.cursor()

    # Create table with the new columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            time_step INT,
            transaction_type VARCHAR(50),
            amount DOUBLE,
            sender_id VARCHAR(50),
            sender_old_balance DOUBLE,
            sender_new_balance DOUBLE,
            receiver_id VARCHAR(50),
            receiver_old_balance DOUBLE,
            receiver_new_balance DOUBLE,
            is_fraud INT,
            is_flagged_fraud INT,
            is_high_value INT,
            sender_balance_diff DOUBLE,
            receiver_balance_diff DOUBLE,
            is_receiver_merchant INT
        );
    """)

    # Prepare insert query with the new columns
    insert_query = """
        INSERT INTO transactions
        (time_step, transaction_type, amount, sender_id, sender_old_balance,
         sender_new_balance, receiver_id, receiver_old_balance, receiver_new_balance,
         is_fraud, is_flagged_fraud, is_high_value, sender_balance_diff,
         receiver_balance_diff, is_receiver_merchant)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Insert each row
    for _, row in df.iterrows():
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    load_csv_to_mysql("../../data/processed/Fraudulent_processed.csv")
