from rouge_score import rouge_scorer
from extractive import ExtractiveSummarizer

fpath = "../models/distilroberta-epoch=3.ckpt"

model = ExtractiveSummarizer.load_from_checkpoint(fpath)


t = """
Paul: Please see the enclosed and call me if you
have any questions. You will note  that I have
assumed that the waiver period would expire either
on a date  certain or if certain events happened
before such date. Please let me know  if there are
other things that may cause the waiver period to
expire early. Carol
""".strip()

print(f"text: {t}\n\n")

summary_sents = model.predict(t, raw_scores=True)

print(summary_sents, "\n")

summary_sents2 = model.predict_sentences(t.split(". "), raw_scores=True)

print(summary_sents2, "\n")


rouge_metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_scorer = rouge_scorer.RougeScorer(
    rouge_metrics, use_stemmer=True
)

tgt = 'The quick brown fox jumps over the lazy dog'
pred = 'The quick brown dog jumps on the log.'

scores = rouge_scorer.score(target=tgt, prediction=pred)
print(scores)