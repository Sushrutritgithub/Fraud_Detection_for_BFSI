def trigger_alert(transaction_id, fraud_prob, explanation):
    if fraud_prob > 0.6:
        print(" HIGH RISK ALERT ")
        print(f"Transaction ID: {transaction_id}")
        print(f"Fraud Probability: {fraud_prob:.2f}")
        print(f"Reason: {explanation}")
        print("-" * 50)

    elif fraud_prob > 0.3:
        print(" Medium Risk Transaction")
        print(f"Transaction ID: {transaction_id}")
        print(f"Fraud Probability: {fraud_prob:.2f}")

    else:
        print(f" Low Risk Transaction: {transaction_id} with Fraud Probability: {fraud_prob:.2f}")
