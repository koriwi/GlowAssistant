def should_keep(choice, confidence, threshold, keep_uncertain):
    if confidence <= threshold:
        return keep_uncertain
    return choice == "keep"
