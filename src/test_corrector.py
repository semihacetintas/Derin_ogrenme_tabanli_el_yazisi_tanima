from spelling_corrector import SpellingCorrector

corrector = SpellingCorrector("../data/corpus.txt")

print(corrector.correction("Aov"))
print(corrector.correction("yoe"))
print(corrector.correction("yov"))
print(corrector.correction("you"))
