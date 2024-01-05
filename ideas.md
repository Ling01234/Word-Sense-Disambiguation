# Todo:
1. modify lesk so that it performs better
2. complete bootstrapping

# Discussions:
- year: BAD RESULT: potential explanation is that the training data containing 'year' is much bigger than the other words. With such a large dataset, there are much more than 2 meanings of 'year' in it, but the model is forced to classify each 'year' into one of the senses, which can cause a misclassification and later impact our testing results.
- game and player have much smaller dataset from the training data, so there are much less chances to mislead our model into thinking it has a different sense
- decreased training data from 5000 to 1000 sentences, massively improved performance
- case: bad result -> training data has many instances where 'cases' was classified as 'case.n.01', and didn't give me model much chance to gather data about the sense 'lawsuit.n.01'. Where as in the test set, many instances was of the lawsuit sense. Thus, we observed a much smaller accuracy score compared to the other words.



# ideas
- model only disambiguate 5
- common words in dev and test set: ["country", "week", "deal", "action", "year"] - ADAM
- my most common words from test set: [('game', 23), ('year', 22), ('player', 21), ('team', 19), ('case', 18)]
  - game:
    - plot.n.1: ['they concocted a plot to discredit the governor', 'I saw through his little game from the start']
    - game.n.02 (a lot more): 'the game lasted two hours', 'a good game of golf', 'what a great game'.
  - year:
    - year.n.01: ['she is 4 years old', 'in the year 1920']
    - class.n.06: 'she was in my year at Hoehandle High'
  - player:
    - player.n.01: "It 's time now to look at professional soccer and rugby and how it can be bad for a player 's health ."
    - player.n.05: 'he was a major player in setting up the corporation'
  - team:
    - team.n.01:
      - "It 's been four years since the U.S. women 's soccer team made history winning the World Cup before more than 90,000 fans at the Rose Bowl ."
      - 'Bi Li , one of the coach patriarchs of the National Women \'s Football Team , revealed the recent " 3 step melody " of the national women \'s football team to journalists a few days ago in the Wuhu arena of the National Women \'s Football League Competition .'
    - team.v.01
      - 'We teamed up for this new project'
- wn.synset.examples(), manual selection from training data, chatgpt sentence generation

  - case:
    - lawsuit.n.01:
      - 'the family brought suit against the landlord'
      - "Judges hearing the case against two men accused in the bombing of Pan Am Flight 103 are expected to announce this week when they 'll reach a verdict ."
      -
    - case.n.01:
      - 'it was a case of bad judgment'
      - 'United Nations health officials say the infection rate for new AIDS cases has remained constant , but the death rate is up .'
      - 'And the growing number of SARS cases in China .'
      -
  -

