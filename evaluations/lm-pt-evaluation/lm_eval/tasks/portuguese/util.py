def enem_generate_options(choices):
    options = ""
    for text, label in zip(choices['text'], choices['label']):
        options += f"{label}. {text}\n"
    return options.strip()

def enem_doc_to_text(doc):
    return f"Pergunta:\n{doc['question']}\nAlternativas:\n{enem_generate_options(doc['choices'])}\nResposta correta:"

def assin2_float_to_pt_str(doc):
    return "{:.1f}".format(doc['relatedness_score']).replace('.', ',')

sparrow_emotion_por_labels = ['Admiration', 'Amusement', 'Anger', 'Annoyance', 'Approval', 'Compassion', 'Confusion', 'Curiosity', 'Desire', 'Disappointment', 'Disapproval', 'Disgust', 'Embarrassment', 'Envy', 'Excitement', 'Fear', 'Gratitude', 'Grief', 'Joy', 'Longing', 'Love', 'Nervousness', 'Optimism', 'Pride', 'Relief', 'Remorse', 'Sadness', 'Surprise']
sparrow_emotion_por_trans = ['Admiração', 'Diversão', 'Raiva', 'Aborrecimento', 'Aprovação', 'Compaixão', 'Confusão', 'Curiosidade', 'Desejo', 'Decepção', 'Desaprovação', 'Nojo', ' Vergonha', 'Inveja', 'Entusiasmo', 'Medo', 'Gratidão', 'Luto', 'Alegria', 'Saudade', 'Amor', 'Nervosismo', 'Otimismo', 'Orgulho', 'Alívio' , 'Remorso', 'Tristeza', 'Surpresa']

def sparrow_emotion_por_trans_label(doc):
    return sparrow_emotion_por_trans[sparrow_emotion_por_labels.index(doc['label'])]
