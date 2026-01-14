# Proper Nouns Synthetic Data Generation Prompt

Generate realistic misspellings of proper nouns as a dyslexic child might write them. Output as JSONL with one JSON object per line.

## Format
```json
{"input": "sentence with misspelling", "output": "corrected sentence"}
```

## Categories to Cover

### 1. Geographic Places (400 examples)
- Countries: Canada, Australia, Germany, Japan, Mexico, Brazil, etc.
- Cities: Toronto, Vancouver, Calgary, Montreal, New York, London, Paris, etc.
- National Parks: Jasper, Banff, Yellowstone, Yosemite, Grand Canyon, etc.
- Landmarks: Niagara Falls, Rocky Mountains, Great Lakes, etc.

Common dyslexic misspellings:
- Letter reversals: Candaa, Torotno, Vancover
- Phonetic: Jermany, Ostralia, Calgaree
- Missing/extra letters: Montral, Yellowstonee, Niagra

### 2. Character Names from Kids' Media (400 examples)
- Harry Potter: Hermione, Dumbledore, Voldemort, Hogwarts, Gryffindor
- Percy Jackson: Poseidon, Annabeth, Olympus
- Marvel/DC: Spider-Man, Batman, Wolverine, Captain America
- Disney: Cinderella, Rapunzel, Frozen, Elsa, Moana
- Minecraft: Creeper, Enderman, Herobrine
- Pokemon: Pikachu, Charizard, Squirtle

Common misspellings:
- Hermione → Hermoine, Herminey, Hermiony
- Dumbledore → Dumbldore, Dumbledor, Dumbeldore
- Pikachu → Pickachu, Pikachoo, Pikatchu

### 3. Historical Figures (300 examples)
- Scientists: Einstein, Newton, Darwin, Curie
- Leaders: Lincoln, Washington, Churchill, Napoleon
- Explorers: Columbus, Magellan, Armstrong
- Artists: Picasso, Mozart, Beethoven, Shakespeare

### 4. Brand Names Kids Know (300 examples)
- Tech: YouTube, Nintendo, PlayStation, iPhone, Minecraft, Roblox, Fortnite
- Food: McDonald's, Starbucks, Subway
- Stores: Walmart, Target, Amazon

### 5. School Subjects & Terms (300 examples)
- Subjects: Mathematics, Geography, Science, Literature
- Terms: Encyclopedia, Dictionary, Bibliography, Hypothesis

### 6. Religious/Mythological Names (300 examples)
- Greek: Zeus, Athena, Hercules, Apollo, Aphrodite
- Norse: Thor, Odin, Loki
- Religious: Christmas, Hanukkah, Muhammad, Buddha

## Example Outputs
```json
{"input": "I went to Jasper Moutain last summer.", "output": "I went to Jasper Mountain last summer."}
{"input": "My favorite character is Hermoine.", "output": "My favorite character is Hermione."}
{"input": "We learned about Aberham Lincon in school.", "output": "We learned about Abraham Lincoln in school."}
{"input": "I play Fortnight every day.", "output": "I play Fortnite every day."}
{"input": "The Eiffle Tower is in Paris.", "output": "The Eiffel Tower is in Paris."}
{"input": "I want to visit Ostrailia.", "output": "I want to visit Australia."}
{"input": "Pikachoo is my favorite Pokemon.", "output": "Pikachu is my favorite Pokemon."}
{"input": "We celebrate Chrismas in December.", "output": "We celebrate Christmas in December."}
```

## Guidelines
1. Use realistic kid sentences (simple structure, common topics)
2. One misspelling per sentence (occasionally two if natural)
3. Mix severity: some obvious (Canida), some subtle (Calgarry)
4. Include common dyslexic patterns:
   - Letter reversals (b/d, p/q)
   - Transpositions (teh → the)
   - Phonetic spellings
   - Missing/doubled letters
5. Vary sentence structures and contexts
