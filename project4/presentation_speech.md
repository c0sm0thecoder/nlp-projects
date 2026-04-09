---
  Slide 1: Title

  What it says: "Stress-Testing Pre-Trained Language Models" - Sentiment Analysis & QA with BiDAF

  Speech:
  ▎ "Today we're presenting a stress test of multilingual BERT. We're not asking 'does mBERT work?'
   - we're asking 'where does it break?' We designed two experiments specifically to expose its
  limits."

  ---
  Slide 2: Two Tracks

  What it shows: mBERT (167M params) branching into two stress tests

  Speech:
  ▎ "We took bert-base-multilingual-uncased and pushed it in two directions. Track 1 tests its
  vocabulary limits - can it handle agglutinative languages like Azerbaijani that weren't in its
  fine-tuning data? Track 2 tests architectural limits - what happens when we plug frozen BERT
  embeddings into a model designed for different input?"

  ---
  Slide 3: Track 1 Baseline

  What it shows: Model profile + case sensitivity test (100% match)

  Speech:
  ▎ "First, we established a baseline. The model has 12 layers, 768 hidden size, trained on product
   reviews in 6 European languages. We confirmed it's truly uncased - 'great product' and 'GREAT
  PRODUCT' produce identical tokenization, predictions, and confidence. Case does NOT affect
  accuracy."

  ---
  Slide 4: English vs Azerbaijani Results

  What it shows: Comparison table showing confidence drop

  Speech:
  ▎ "Here's where it gets interesting. Look at this table - the model gets the direction right. A
  positive Azerbaijani review still predicts 5 stars, negative still predicts 1-2 stars. BUT look
  at the confidence: English averages 76%, Azerbaijani drops to 51%. That's a 25 percentage point
  gap. The model is guessing, not understanding."

  ---
  Slide 5: The Morphological Map ⭐ KEY SLIDE

  What it shows: "məhsullarımızdan" exploding into 8 subword tokens

  Speech:
  ▎ "This slide explains WHY. Take the Azerbaijani word 'məhsullarımızdan' - it means 'from our
  products.' In English, that's 3 words. In Azerbaijani, it's ONE word with suffixes stacked. But
  mBERT's tokenizer shatters it into 8 disconnected pieces: meh, sul, lar, imi, z, da, n. Each
  suffix carries meaning - plural, possessive, ablative case - but the model sees random fragments.
   The morphological coherence is destroyed."

  ---
  Slide 6: BiDAF Architecture

  What it shows: 7-layer stack with "Target Zone" highlighting layers 1-3

  Speech:
  ▎ "For Track 2, we implemented BiDAF - Bidirectional Attention Flow. It has 7 layers: character
  CNN, word embeddings, highway network, contextual BiLSTM, attention flow, modeling layer, and
  output. The bottom three layers - that's our target zone. That's what we're going to swap out."

  ---
  Slide 7: The Embedding Swap ⭐ KEY SLIDE

  What it shows: Baseline (GloVe+CharCNN+Highway) vs BiDAF-BERT (frozen mBERT + projection)

  Speech:
  ▎ "Here's the experiment. On the left: baseline BiDAF using GloVe word vectors plus character CNN
   combined through a highway network. On the right: we rip all that out and replace it with frozen
   mBERT. The BERT weights are locked - only the projection layer from 768 to 200 dimensions can
  learn. This is a common 'plug-and-play' approach people try."

  ---
  Slide 8: Architecture Showdown ⭐ KEY RESULT

  What it shows: BiDAF wins! 19.7% EM, 30.1% F1 vs BiDAF-BERT's 16.4% EM, 24.0% F1

  Speech:
  ▎ "And here's the upset. The 3-million parameter model with static 1990s-era embeddings BEATS the
   168-million parameter transformer. By 3.3 points on Exact Match, 6.1 points on F1. The tiny
  model wins. How is this possible?"

  ---
  Slide 9: Learning Curves

  What it shows: Loss decreasing but EM diverging

  Speech:
  ▎ "The learning curves reveal something subtle. See how BERT's validation loss is actually LOWER?
   It has better-calibrated probability distributions. But look at Exact Match - it flatlines. BERT
   knows it's uncertain, but it can't pinpoint exact answer boundaries. It's calibrated but
  imprecise."

  ---
  Slide 10: Why GloVe Won ⭐ DIAGNOSTIC

  What it shows: 5 factors explaining the failure

  Speech:
  ▎ "Five factors explain this:
  ▎ 1. Frozen encoder + small data: Only the projection layer learns. 5K examples isn't enough.
  ▎ 2. Tokenization mismatch: BiDAF expects word-level tokens. BERT gives subwords. Attention gets
  diluted.
  ▎ 3. Small training set: The projection can't learn the mapping.
  ▎ 4. Multilingual dilution: mBERT spreads capacity across 104 languages. English suffers.
  ▎ 5. Lost character features: The baseline's character CNN captures morphology. BERT threw that
  away."

  ---
  Slide 11: Dashboard UI

  What it shows: FastAPI dashboard with tokenization visualization

  Speech:
  ▎ "We built a production-ready diagnostic tool. Real-time tokenization visualization for
  Azerbaijani text, interactive training curves, model comparison. This isn't just a notebook -
  it's deployable infrastructure."

  ---
  Slide 12: The Pre-Trained Paradox ⭐ THESIS

  What it shows: Venn diagram - Task 1 failure (fragmentation) + Task 2 failure (mismatch)

  Speech:
  ▎ "Both experiments converge on the same insight: raw embedding quality cannot overcome
  fundamental incompatibility. In Task 1, mBERT's subword tokenization fights against agglutinative
   syntax. In Task 2, it fights against word-level span attention. The problem isn't that mBERT is
  bad - it's that we used it wrong."

  ---
  Slide 13: Conclusions

  What it shows: Three takeaways + future work

  Speech:
  ▎ "Three conclusions:
  ▎ 1. For agglutinative languages: mBERT needs dedicated fine-tuning or a Turkic-specific model.
  ▎ 2. For architecture integration: Don't plug-and-play. Compatibility, data volume, and
  fine-tuning strategy matter more than parameter count.
  ▎ 3. Future work: Full SQuAD training, swap to English-only BERT, and unfreeze the encoder.

  ▎ Thank you. Questions?"