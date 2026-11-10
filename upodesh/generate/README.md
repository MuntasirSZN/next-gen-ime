# Data generation

### For generating all the possible patterns from regex (like the `preprocessed-patterns.json`)
For Avro Phonetic, `upodesh` needs the all possible Bangla character patterns for Avro Phonetic patterns.

```
cargo r -- explode ../data/source-regex-patterns.json ../data/preprocessed-patterns.json
```
