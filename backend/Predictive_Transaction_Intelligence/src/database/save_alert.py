from src.database.mysql_connection import get_mysql_connection

def save_alert(transaction_id, fraud_prob, alert_level, message):
    conn = get_mysql_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO alerts
    (transaction_id, fraud_probability, alert_level, message)
    VALUES (%s, %s, %s, %s)
    """

    cursor.execute(query, (
        transaction_id,
        fraud_prob,
        alert_level,
        message
    ))

    conn.commit()
    cursor.close()
    conn.close()
