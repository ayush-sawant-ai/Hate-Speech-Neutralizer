"""Manually reviewed regression dataset covering 11 critical linguistic and context categories.

Note:
The binary classification target represents general toxicity derived from the Jigsaw dataset.
It does NOT establish a separate fine-grained hate-speech category. This test set evaluates
boundary conditions, false positives, false negatives, and context sensitivity.
"""

REGRESSION_TEST_CASES = [
    # 1. Direct insults and harassment
    {
        "category": "direct_insults_and_harassment",
        "text": "You are a complete idiot and nobody cares what you think.",
        "expected_label": "TOXIC",
        "rationale": "Direct personal insult targeted at an individual."
    },
    {
        "category": "direct_insults_and_harassment",
        "text": "Shut your mouth, you pathetic clown.",
        "expected_label": "TOXIC",
        "rationale": "Harassing language and abusive demeanment."
    },
    
    # 2. Identity-targeted hate
    {
        "category": "identity_targeted_hate",
        "text": "People from that country are all criminals and should be thrown out.",
        "expected_label": "TOXIC",
        "rationale": "Generalizing criminal accusation against a nationality/origin group."
    },
    {
        "category": "identity_targeted_hate",
        "text": "Members of that religious group are pests and don't belong here.",
        "expected_label": "TOXIC",
        "rationale": "Dehumanizing language directed at a religious demographic."
    },

    # 3. Threats
    {
        "category": "threats",
        "text": "I will find where you live and destroy you.",
        "expected_label": "TOXIC",
        "rationale": "Direct physical threat of violence and retaliation."
    },
    {
        "category": "threats",
        "text": "I hope your family burns in a terrible fire.",
        "expected_label": "TOXIC",
        "rationale": "Severe violent wish directed at relatives."
    },

    # 4. Profanity without a personal attack
    {
        "category": "profanity_without_attack",
        "text": "Damn, this pizza is so fucking delicious!",
        "expected_label": "SAFE",
        "rationale": "Expressive profane intensifier expressing positive excitement without hostility."
    },
    {
        "category": "profanity_without_attack",
        "text": "Holy shit, that thunder was terrifying.",
        "expected_label": "SAFE",
        "rationale": "Colloquial vulgar exclamation about weather, no target or insult."
    },

    # 5. Strong but legitimate criticism
    {
        "category": "strong_legitimate_criticism",
        "text": "The government's fiscal plan is completely irresponsible and based on false economic data.",
        "expected_label": "SAFE",
        "rationale": "Rigorous political and institutional policy critique without personal attack."
    },
    {
        "category": "strong_legitimate_criticism",
        "text": "This paper's methodology has severe statistical flaws and the conclusion cannot be trusted.",
        "expected_label": "SAFE",
        "rationale": "Severe academic criticism of methodology without toxicity."
    },

    # 6. Negative sentiment without toxicity
    {
        "category": "negative_sentiment_without_toxicity",
        "text": "I feel deeply depressed today and everything seems utterly hopeless.",
        "expected_label": "SAFE",
        "rationale": "Personal emotional distress and sorrow without aggression."
    },
    {
        "category": "negative_sentiment_without_toxicity",
        "text": "The movie was utterly boring, slow-paced, and a waste of two hours.",
        "expected_label": "SAFE",
        "rationale": "Negative creative evaluation devoid of hostility or abuse."
    },

    # 7. Neutral identity mentions
    {
        "category": "neutral_identity_mentions",
        "text": "She is a proud Muslim woman working as an aerospace engineer in Berlin.",
        "expected_label": "SAFE",
        "rationale": "Positive/neutral demographic and professional descriptor."
    },
    {
        "category": "neutral_identity_mentions",
        "text": "The local LGBTQ community organized a health awareness workshop on Saturday.",
        "expected_label": "SAFE",
        "rationale": "Community event announcement with group identity term."
    },

    # 8. Quoted abusive language and statements condemning hate
    {
        "category": "quoted_abuse_or_condemning_hate",
        "text": "The referee told the player 'you are trash' which led to the penalty.",
        "expected_label": "SAFE",
        "rationale": "Factual reporting of a quote within narrative context."
    },
    {
        "category": "quoted_abuse_or_condemning_hate",
        "text": "Hate speech and racism against immigrants must be eradicated from our society.",
        "expected_label": "SAFE",
        "rationale": "Anti-hate condemnation statement opposing discriminatory behavior."
    },

    # 9. Negation and ambiguous wording
    {
        "category": "negation_and_ambiguity",
        "text": "I am not saying you are stupid, but this decision makes no sense.",
        "expected_label": "SAFE",
        "rationale": "Explicit negation of an insult paired with critique of an action."
    },
    {
        "category": "negation_and_ambiguity",
        "text": "They killed it on stage tonight, what a performance!",
        "expected_label": "SAFE",
        "rationale": "Slang phrase meaning outstanding performance despite violent verb."
    },

    # 10. Misspellings, obfuscated words, emojis, and punctuation
    {
        "category": "misspellings_obfuscations_emojis",
        "text": "u r an id10t and a b!tch 😡 🤬",
        "expected_label": "TOXIC",
        "rationale": "Leetspeak and symbol obfuscation of profanity and insults with angry emojis."
    },
    {
        "category": "misspellings_obfuscations_emojis",
        "text": "You are pure tr@sh!!!!",
        "expected_label": "TOXIC",
        "rationale": "Symbol replacement in insult with emphatic punctuation."
    },

    # 11. Long inputs and unsupported language inputs
    {
        "category": "long_or_multilingual_inputs",
        "text": (
            "Over the past twenty years, our organization has maintained a steadfast dedication to "
            "environmental preservation, community development, and sustainable technological growth. "
            "We believe that through transparent collaboration, scientific rigor, and stakeholder "
            "engagement, complex global challenges can be methodically resolved without conflict."
        ),
        "expected_label": "SAFE",
        "rationale": "Formal lengthy institutional statement without toxic sentiment."
    },
    {
        "category": "long_or_multilingual_inputs",
        "text": "Das ist ein wunderbarer und sonniger Tag in München.",
        "expected_label": "SAFE",
        "rationale": "German text ('This is a wonderful and sunny day in Munich') outside model training distribution."
    }
]

