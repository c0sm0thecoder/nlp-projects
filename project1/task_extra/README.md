# Task Extra: Spell Checker with Character Confusion Matrix

This task creates a spell checker using weighted edit distance and shows which letters get mistaken for which letters. It processes all unique words from the Azerbaijani poetry dataset and makes confusion matrices to analyze the errors.

The spell checker finds similar words when there are typos by calculating edit distance with custom costs for Azerbaijani characters. It also tracks what character substitutions happen most often.

Results: 12.83% accuracy on 500 test cases using 132,242 total words. Most common character substitutions were 'g'→'ə', 's'→'i', and 'ş'→'v'. Consonant errors were more common than vowel errors.