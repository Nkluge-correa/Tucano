def enem_generate_options(choices):
    options = ""
    for text, label in zip(choices['text'], choices['label']):
        options += f"{label}. {text}\n"
    return options.strip()

def enem_doc_to_text(doc):
    return f"Pergunta:\n{doc['question']}\nAlternativas:\n{enem_generate_options(doc['choices'])}\nResposta Correta:"

def assin2_float_to_pt_str(doc):
    return "{:.1f}".format(doc['relatedness_score']).replace('.', ',')