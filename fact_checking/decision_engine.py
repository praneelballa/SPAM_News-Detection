def final_decision(text, model, vectorizer, fact_checker):
    
    # ML prediction
    text_vec = vectorizer.transform([text])
    ml_pred = model.predict(text_vec)[0]   # "fake" or "real"
    
    # Fact check
    fact_result = fact_checker.check_fact(text)
    
    similarity = fact_result["similarity"]
    fact_label = fact_result["label"]
    
    # Threshold (tune this)
    SIM_THRESHOLD = 0.4
    
    if similarity > SIM_THRESHOLD:
        if ml_pred == "fake" and fact_label == "false":
            verdict = "FAKE (High Confidence)"
        
        elif ml_pred == "real" and fact_label == "true":
            verdict = "REAL (High Confidence)"
        
        else:
            verdict = "CONFLICT (Needs Verification)"
    
    else:
        if ml_pred == "fake":
            verdict = "FAKE (Low Confidence)"
        else:
            verdict = "REAL (Low Confidence)"
    
    return {
        "verdict": verdict,
        "ml_prediction": ml_pred,
        "fact_match": fact_result["match"],
        "fact_label": fact_label,
        "similarity": similarity,
        "source": fact_result["source"]
    }