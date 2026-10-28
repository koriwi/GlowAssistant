import json

from .const import QUERY
from .numeric_values import finite_number


def decision_details(questions, answers, config):
    details = {}
    for name, question in questions.items():
        answer = answers.get(name, {}) if isinstance(answers, dict) else {}
        if not isinstance(answer, dict):
            answer = {}
        confidence = answer.get("confidence")
        confidence = confidence if finite_number(confidence) else None
        threshold = config[f"{name if name in ('service', 'domain', 'area', 'target') else 'numeric'}_confidence"]
        info = {"type": question["type"], "confidence": confidence, "minimum_confidence": threshold}
        if question["type"] == "choice":
            choice = answer.get("choice")
            choice = choice if isinstance(choice, str) else None
            probabilities = answer.get("probabilities")
            probabilities = probabilities if isinstance(probabilities, dict) else {}
            probability = probabilities.get(choice)
            info.update(
                choice=choice, probability=probability if finite_number(probability) else None,
                effective_choice=choice if confidence is not None and confidence >= threshold else "unsure",
            )
            criteria = question["criteria"]
            if choice in criteria:
                selected = criteria[choice]
                try:
                    selected = json.loads(selected)
                except (ValueError, TypeError):
                    pass
                info["selected"] = selected
            alternatives = sorted(
                ((label, value) for label, value in probabilities.items() if label != choice and finite_number(value)),
                key=lambda item: item[1], reverse=True,
            )[:3]
            info["alternatives"] = [{"choice": label, "probability": value} for label, value in alternatives]
        else:
            score = answer.get("score")
            info["score"] = score if finite_number(score) else None
        details[name] = info
    if "service" in details and "domain" in details:
        details["domain"]["used"] = details["service"].get("effective_choice") == QUERY
    return details
