from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def compute_bleu(model, sources, targets):
    hypotheses = []
    for source in sources:
        y_batch = model.inference(source)
        hypotheses.extend(y_batch)
    references = [[y] for batch in targets for y in batch]
    bleu = corpus_bleu(references, hypotheses,
                       smoothing_function=SmoothingFunction().method1) * 100
    return bleu, hypotheses
