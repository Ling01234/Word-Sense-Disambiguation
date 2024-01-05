# Project Description
This project focuses on implementing word sense disambiguation algorithms using `wordnet`'s senses. Specifically, four algorithms are explored:
1. **The most freqent baseline**. This serves as a baseline by selecting the #1 indicated sense in the synset from `wordnet`.
2. **Lesk's algorithm**. This utilizes dictionary definitions of a word's senses, disambiguating based on the highest overlap score between the current context and dictionary definitions.
3. **Yarowsky's bootstrapping algorithm**. The idea is to apply a minimally supervised (semi-supervised) machine learning apporach. The high level idea is to manually label a small set of examples, repeatatively train a supervised model, and add highly confident classifications to the seed set.
4. **Baseline + Lesk**. This hybrid approach integrates the baselie method with Lesk's algorithm. Essentially, it involves applying a weight penalty to less frequent senses.

Overall, we see (and expect) that Yarowsky's algorithm performs best.