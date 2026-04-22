#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

FRUITS = ["mango", "watermelon", "apple", "banana", "orange", "grape", "pear", "peach"]
VITAMINS = ["vitamin C", "vitamin A", "fiber", "potassium", "antioxidants"]
PLACES = ["Canada", "the US", "Japan", "France", "Australia", "Germany"]
HOBBIES = ["reading", "running", "cooking", "swimming", "gaming", "painting"]
PEOPLE = ["my sister", "my brother", "my friend", "my teacher", "my colleague"]
EVENTS = ["hiking", "traveling", "shopping", "watching movies", "attending a concert"]
OBJECTS = ["car", "bike", "phone", "laptop", "house"]
COLORS = ["blue", "red", "green", "black", "white"]
NUMBERS = [3, 5, 7, 10, 12]
DISTRACTORS = [
    "I bought groceries today.",
    "The weather was warm and sunny.",
    "I watched a short video online.",
    "I need to reply to an email later.",
    "I cooked dinner last night.",
    "I went for a walk after lunch.",
    "The meeting was moved to tomorrow.",
    "I read a few pages of a book.",
]

TOPICS = {
    "daily_life": [
        "I woke up early and made a cup of coffee before starting my day.",
        "After breakfast, I checked my messages and planned a few things.",
        "In the afternoon, I went for a short walk to relax.",
        "In the evening, I stayed home and kept things simple.",
    ],

    "food_related": [
        "I usually keep fruit at home because it is easy to eat.",
        "I often choose something light after lunch.",
        "I have been paying more attention to what I eat recently.",
        "I like food that feels simple and refreshing.",
    ],

    "knowledge": [
        "Mango contains vitamin C and is often considered a healthy fruit.",
        "Watermelon is refreshing and commonly eaten in warm weather.",
        "Different fruits provide different nutrients.",
        "A balanced diet usually includes a variety of foods.",
    ],

    "general": [
        "People often choose food based on taste and convenience.",
        "Habits can change slowly over time.",
        "Small daily choices can make a difference.",
        "It is easier to maintain simple routines.",
    ],

    "social": [
        "My friend prefers watermelon over other fruits.",
        "People around me have different preferences.",
        "I have noticed others talking about fruit more often recently.",
        "Some people like sweeter fruit while others prefer lighter options.",
    ],

    "topic_shift": [
        "Recently I have been thinking about changing a few habits.",
        "I have also been trying to keep a more regular routine.",
        "Some days I focus on food, and other days on work.",
        "I prefer making small changes instead of big ones.",
    ],

    "narrative": [
        "Last weekend I spent some time walking around the city.",
        "I met a friend and we talked about different topics.",
        "It was a quiet and relaxed day overall.",
        "Nothing special happened, but it felt comfortable.",
    ],
    "travel": [
        "Last month I visited a quiet coastal town and walked around for a while.",
        "I took a few photos and spent some time near the water.",
        "The trip felt relaxed and unhurried.",
        "I still remember how calm that day felt.",
    ]
}

BRIDGES = [
    "That reminded me of something from earlier in the week.",
    "I was thinking about it again after that.",
    "It made me notice a small detail I had ignored before.",
    "I kept thinking about that while doing other things.",
    "The conversation stayed in my mind for a while.",
]

QUERY_TEMPLATES = {

    "preference": [
        "What fruit do I like the most?",
        "Which fruit do I usually prefer?",
        "What kind of fruit do I tend to choose?",
        "If I had to pick, which fruit would I go for?",
        "What fruit do I seem to enjoy the most?",
    ],

    "reason": [
        "Why do I like {fruit}?",
        "What is the reason I prefer {fruit}?",
        "Why do I tend to choose {fruit}?",
        "What makes me like {fruit}?",
        "Why do I keep choosing {fruit}?",
    ],

    "fact": [
        "Where was I born?",
        "What place was I originally from?",
        "Which country was I born in?",
        "Where do I come from?",
    ],

    "habit": [
        "What do I usually do in the morning?",
        "What is part of my morning routine?",
        "What do I tend to do early in the day?",
        "What activity do I regularly do in the morning?",
    ],

    "relation": [
        "Where does {person} live?",
        "Which place does {person} live in?",
        "Where is {person} based?",
    ],

    "attribute": [
        "What color is my {object}?",
        "Which color did I choose for my {object}?",
        "What color is the {object} I have?",
    ],

    "event": [
        "What did I do last weekend?",
        "What activity did I do recently?",
        "What did I spend time doing last weekend?",
    ],

    "knowledge": [
        "What does {fruit} contain?",
        "Which nutrient is found in {fruit}?",
        "What is one thing {fruit} provides?",
    ],

    "numeric": [
        "How far do I run every day?",
        "What distance do I usually run?",
        "How many kilometers do I run daily?",
    ],

    "shift": [
        "What fruit do I like recently?",
        "Which fruit do I prefer these days?",
        "What fruit have I been choosing more lately?",
        "What fruit do I seem to like now?",
    ]
}

HARD_QUERY_TEMPLATES = {
    "preference": [
        "Based on what I mentioned earlier, which fruit would I most likely choose?",
        "From my earlier comments, what fruit seems to be my usual pick?",
        "What fruit do I appear to rely on most often?",
        "If I had to choose one fruit based on what I said before, which would it be?",
    ],

    "reason": [
        "Based on what I said earlier, why would that fruit make sense for me?",
        "What earlier detail explains why I like {fruit}?",
        "What is the underlying reason I keep choosing {fruit}?",
        "What about {fruit} seems to be the main reason I prefer it?",
    ],

    "fact": [
        "From the background I mentioned earlier, where am I originally from?",
        "Which place was tied to my background?",
        "What location did I mention as my birthplace before?",
    ],

    "habit": [
        "What morning activity seems to be part of my routine?",
        "From my earlier remarks, what do I usually do early in the day?",
        "What habit did I seem to describe as regular?",
    ],

    "relation": [
        "Based on what I said earlier, where does {person} live?",
        "What place was associated with {person} in my earlier remarks?",
        "Which location did I mention for {person}?",
    ],

    "attribute": [
        "From the details I gave earlier, what color is my {object}?",
        "What color did I mention for the {object} I own?",
        "Which color was tied to my {object} earlier?",
    ],

    "event": [
        "Based on my earlier description, what did I do last weekend?",
        "What activity did I say I spent time doing recently?",
        "Which event or outing did I refer to earlier?",
    ],

    "knowledge": [
        "What nutrient was associated with {fruit} in what I said earlier?",
        "From the earlier description, what does {fruit} provide?",
        "What was one thing I mentioned {fruit} contains?",
    ],

    "numeric": [
        "What distance did I mention I run every day?",
        "From the earlier details, how far do I usually run?",
        "What number was tied to my daily running routine?",
    ],

    "shift": [
        "Based on my recent comments, which fruit do I seem to prefer now?",
        "What fruit appears to be my current preference?",
        "Which fruit did I suggest I have been choosing more lately?",
    ]
}

DIFFICULTY_CFG = {
    "easy":   {"n_aux": 2, "min_sent": 1, "max_sent": 2, "target_tokens": 160, "hard_query_prob": 0.30},
    "medium": {"n_aux": 4, "min_sent": 2, "max_sent": 4, "target_tokens": 520, "hard_query_prob": 0.50},
    "hard":   {"n_aux": 7, "min_sent": 3, "max_sent": 6, "target_tokens": 1100, "hard_query_prob": 0.83},
}

FILLER_TOPICS = ["daily_life", "food_related", "knowledge", "general", "social", "topic_shift", "narrative"]

def make_long_distractor(min_paragraphs=1, max_paragraphs=2):
    """
    Returns one natural-looking distractor paragraph.
    The paragraph is built from 2-5 short sentences, with slight topic drift.
    """
    topics = list(TOPICS.keys())
    n_paragraphs = random.randint(min_paragraphs, max_paragraphs)

    paragraphs = []
    base_topic = random.choice(topics)
    current_topic = base_topic

    for _ in range(n_paragraphs):
        # 70% same topic, 30% slight drift
        if random.random() < 0.3:
            current_topic = random.choice([t for t in topics if t != current_topic])

        num_sentences = random.randint(2, 4)
        sentences = random.sample(TOPICS[current_topic], k=min(num_sentences, len(TOPICS[current_topic])))

        # Add a gentle bridge sentence sometimes
        if random.random() < 0.5:
            bridge = random.choice(BRIDGES)
            sentences = [bridge] + sentences

        paragraphs.append(" ".join(sentences))

    return " ".join(paragraphs)

def pick(xs):
    return random.choice(xs)

def sample_distractors(n):
    distractors = []
    types = []
    strengths = []

    topic_keys = list(TOPICS.keys())

    for _ in range(n):
        r = random.random()

        # 控制干扰强度（逻辑不变）
        if r < 0.3:
            topic = "daily_life"
            strength = "weak"
        elif r < 0.55:
            topic = "food_related"
            strength = "medium"
        elif r < 0.8:
            topic = "knowledge"
            strength = "medium"
        elif r < 0.92:
            topic = "social"
            strength = "strong"
        else:
            topic = "topic_shift"
            strength = "strong"

        # 👇 核心升级：多句 + 混合 topic
        num_sentences = random.randint(3, 7)

        sentences = []
        current_topic = topic

        for i in range(num_sentences):

            # 30% 概率轻微 topic 漂移（更自然）
            if random.random() < 0.21:
                current_topic = random.choice(topic_keys)

            sentences.append(random.choice(TOPICS[current_topic]))

        # 可选：加自然连接句
        if random.random() < 0.4:
            sentences.insert(0, random.choice([
                "That made me think about something else.",
                "I was also thinking about this earlier.",
                "It reminded me of something from before.",
                "I kept thinking about it for a while.",
            ]))

        text = " ".join(sentences)

        distractors.append(text)
        types.append(topic)
        strengths.append(strength)

    # 主干扰类型
    interference_type = max(set(types), key=types.count)

    # 强度
    if "strong" in strengths:
        interference_strength = "strong"
    elif "medium" in strengths:
        interference_strength = "medium"
    else:
        interference_strength = "weak"

    return distractors, interference_type, interference_strength
    

def gap_size(difficulty):
    return {
        "easy": 1,      # ~100 tokens
        "medium": 3,    # ~300 tokens
        "hard": 8       # ~800–1500 tokens
    }[difficulty]

def domains():
    return [
        "preference", "fact", "habit", "reason", "relation",
        "attribute", "event", "knowledge", "numeric", "shift"
    ]

def make_block(topic, min_sent=2, max_sent=4):
    """
    Generate one natural conversational turn block.
    Each block is 2-4 sentences, so the full sample becomes multi-turn and long.
    """
    n = random.randint(min_sent, max_sent)
    sents = random.choices(TOPICS[topic], k=n)
    if random.random() < 0.45:
        sents.insert(0, random.choice(BRIDGES))
    return " ".join(sents)

####difficulty controls:
def pad_turns_to_target(turns, query, difficulty, min_sent, max_sent, filler_topics_used):
    cfg = DIFFICULTY_CFG[difficulty]
    max_extra_turns = 20
    extra_turns = 0
    while estimate_tokens(" ".join(turns) + " " + query) < cfg["target_tokens"] and extra_turns < max_extra_turns:
        topic = random.choice(FILLER_TOPICS)
        turns.append(make_block(topic, min_sent, max_sent))
        filler_topics_used.append(topic)
        extra_turns += 1
    return turns

####

def build_query(domain, context_vars, hard=False):
    templates = HARD_QUERY_TEMPLATES if hard else QUERY_TEMPLATES
    template = random.choice(templates[domain])

    try:
        return template.format(**context_vars)
    except KeyError:
        return template
    
def sample_query_mode(difficulty):
    cfg = DIFFICULTY_CFG[difficulty]
    return "hard" if random.random() < cfg["hard_query_prob"] else "normal"

def build_turns(domain, difficulty):
    """
    Multi-turn version:
    - Turn 1: introduce the key memory
    - Turn 2-4: natural drift / supporting details / other memory slots
    - Turn 5-6: update or reinforce
    - Query: final recall question
    """
    cfg = DIFFICULTY_CFG[difficulty]
    n_aux = cfg["n_aux"]
    block_min = cfg["min_sent"]
    block_max = cfg["max_sent"]

    turns = []
    meta = {"difficulty_cfg": cfg.copy()}
    filler_topics_used = []

    def add(topic):
        turns.append(make_block(topic, block_min, block_max))
        filler_topics_used.append(topic)

    query_mode = sample_query_mode(difficulty)
    hard_query = (query_mode == "hard")

    if domain == "preference":
        fruit = pick(FRUITS)
        turns.append(f"I really like {fruit}. It is one of the fruits I enjoy most.")
        add("food_related")
        add("daily_life")
        if n_aux >= 4:
            add("social")
        if n_aux >= 5:
            turns.append("I usually keep some fruit at home so I can eat it after lunch.")
        if n_aux >= 6:
            add("topic_shift")
        query = build_query("preference", {}, hard=hard_query)
        gold = [fruit]
        answer_type = "entity"
        meta["slot"] = "preference"

    elif domain == "fact":
        place = pick(PLACES)
        turns.append(f"I was born in {place}, and I still mention it sometimes when people ask about my background.")
        add("daily_life")
        add("narrative")
        if n_aux >= 4:
            add("narrative")
        if n_aux >= 5:
            turns.append("Even now, I sometimes compare different places to where I grew up.")
        if n_aux >= 6:
            add("topic_shift")
        query = build_query("fact", {}, hard=hard_query)
        gold = [place]
        answer_type = "location"
        meta["slot"] = "birth_place"

    elif domain == "habit":
        hobby = pick(HOBBIES)
        turns.append(f"I usually enjoy {hobby} in the morning before the day gets too busy.")
        add("daily_life")
        add("narrative")
        if n_aux >= 4:
            add("narrative")
        if n_aux >= 5:
            turns.append("It has become one of the routines that helps me start the day calmly.")
        if n_aux >= 6:
            add("topic_shift")
        query = build_query("habit", {}, hard=hard_query)
        gold = [hobby]
        answer_type = "activity"
        meta["slot"] = "habit"

    elif domain == "reason":
        fruit = pick(FRUITS)
        reason = pick(VITAMINS)
        turns.append(f"I like {fruit} because it has {reason} and it also feels easy to eat.")
        add("food_related")
        add("daily_life")
        if n_aux >= 4:
            turns.append("When I want something simple and healthy, I usually think about that fruit first.")
        if n_aux >= 5:
            add("social")
        if n_aux >= 6:
            turns.append("The health benefits are one reason I keep choosing it.")
        query = build_query("reason", {"fruit": fruit}, hard=hard_query)
        gold = [reason, f"it has {reason}", f"because it has {reason}"]
        answer_type = "reason"
        meta["slot"] = "motivation"

    elif domain == "relation":
        person = pick(PEOPLE)
        place = pick(PLACES)
        turns.append(f"{person.title()} lives in {place}, and I usually hear about it when we talk about family.")
        add("daily_life")
        add("travel")
        if n_aux >= 4:
            add("social")
        if n_aux >= 5:
            turns.append("I sometimes forget the details of other people, but I remember that place clearly.")
        if n_aux >= 6:
            add("topic_shift")
        query = build_query("relation", {"person": person}, hard=hard_query)
        gold = [place]
        answer_type = "location"
        meta["slot"] = "relationship"

    elif domain == "attribute":
        obj = pick(OBJECTS)
        color = pick(COLORS)
        turns.append(f"My {obj} is {color}, and I picked that color because I liked how it looked.")
        add("daily_life")
        add("narrative")
        if n_aux >= 4:
            add("social")
        if n_aux >= 5:
            turns.append("I still remember that detail whenever I talk about that object.")
        if n_aux >= 6:
            add("topic_shift")
        query = build_query("attribute", {"object": obj}, hard=hard_query)
        gold = [color]
        answer_type = "attribute"
        meta["slot"] = "object_color"

    elif domain == "event":
        event = pick(EVENTS)
        place = pick(["Yosemite", "Tokyo", "Paris", "Sydney", "Seoul"])
        turns.append(f"Last weekend I went {event} in {place}, and it felt like a nice break.")
        add("daily_life")
        add("travel")
        if n_aux >= 4:
            add("social")
        if n_aux >= 5:
            turns.append("That day stood out because it felt different from my usual routine.")
        if n_aux >= 6:
            add("topic_shift")
        query = build_query("event", {}, hard=hard_query)
        gold = [event, place, f"{event} in {place}"]
        answer_type = "event"
        meta["slot"] = "episodic_memory"

    elif domain == "knowledge":
        fruit = pick(FRUITS)
        vitamin = pick(VITAMINS)
        turns.append(f"{fruit.title()} contains {vitamin}, which is one reason I keep it in mind.")
        add("food_related")
        add("general")
        if n_aux >= 4:
            turns.append("I remember that kind of detail because it connects taste with health.")
        if n_aux >= 5:
            add("topic_shift")
        if n_aux >= 6:
            turns.append("That kind of information usually sticks with me better than random facts.")
        query = build_query("knowledge", {"fruit": fruit}, hard=hard_query)
        gold = [vitamin]
        answer_type = "fact"
        meta["slot"] = "knowledge_link"

    elif domain == "numeric":
        num = pick(NUMBERS)
        turns.append(f"I run {num} kilometers every day, usually before the rest of the day gets busy.")
        add("daily_life")
        add("narrative")
        if n_aux >= 4:
            add("topic_shift")
        if n_aux >= 5:
            turns.append("It is a simple routine, but I try to keep track of it carefully.")
        if n_aux >= 6:
            add("general")
        query = build_query("numeric", {}, hard=hard_query)
        gold = [str(num), f"{num} kilometers"]
        answer_type = "number"
        meta["slot"] = "numeric_memory"

    elif domain == "shift":
        old = pick(FRUITS)
        new = pick([f for f in FRUITS if f != old])
        turns.append(f"My favorite fruit used to be {old}. I liked it for a long time.")
        add("daily_life")
        turns.append(f"Recently I have been eating a lot of {new}, especially when I want something refreshing.")
        if n_aux >= 4:
            add("food_related")
        if n_aux >= 5:
            turns.append("It feels like my preference has changed a bit over time.")
        if n_aux >= 6:
            add("social")
        query = build_query("shift", {}, hard=hard_query)
        gold = [new]
        answer_type = "update"
        meta["slot"] = "preference_shift"
        meta["old_preference"] = old
        meta["new_preference"] = new

    else:
        raise ValueError(f"Unknown domain: {domain}")

    turns = pad_turns_to_target(turns, query, difficulty, block_min, block_max, filler_topics_used)

    context_tokens = estimate_tokens(" ".join(turns))
    approx_tokens = estimate_tokens(" ".join(turns) + " " + query)

    # 先把 distractor_count 定义出来
    distractor_count = len(filler_topics_used)

    if filler_topics_used:
        interference_type = max(set(filler_topics_used), key=filler_topics_used.count)
    else:
        interference_type = "mixed"

    if any(t in ("social", "topic_shift") for t in filler_topics_used):
        interference_strength = "strong"
    elif any(t in ("food_related", "knowledge") for t in filler_topics_used):
        interference_strength = "medium"
    else:
        interference_strength = "weak"

    meta["query_mode"] = query_mode
    meta["interference_type"] = interference_type
    meta["interference_strength"] = interference_strength
    meta["target_tokens"] = cfg["target_tokens"]

    gap = context_tokens  # 这里补上 gap

    return turns, query, gold, answer_type, gap, meta, query_mode, distractor_count, context_tokens, approx_tokens


def estimate_tokens(text):
    return int(len(text.split()) * 1.3)

def make_sample(domain, idx, difficulty, eval_mode):
    turns, query, gold, answer_type, gap, meta, query_mode, distractor_count, context_tokens, approx_tokens = build_turns(domain, difficulty)
    messages = [{"role": "user", "content": t} for t in turns]
    sample = {
        "id": f"imt_{domain}_{difficulty}_{idx:06d}",
        "task_type": domain,
        "answer_type": answer_type,
        "eval_mode": eval_mode,
        "difficulty": difficulty,
        "gap_tokens": context_tokens,
        "distractor_count": distractor_count,
        "turns": messages,
        "query_turn": {"role": "user", "content": query},
        "gold_answers": gold,
        "gold_explanation": gold[0] if domain == "reason" else None,
        "metadata": {
            **meta,
            "approx_tokens": approx_tokens,
            "context_tokens": context_tokens,
            "query_mode": query_mode,
        }
    }
    return sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="imt_mem_v2.jsonl")
    ap.add_argument("--n_per_domain", type=int, default=300)
    ap.add_argument("--seed", type=int, default=429)
    ap.add_argument("--eval_modes", type=str, default="incremental,full_context")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    eval_modes = [x.strip() for x in args.eval_modes.split(",") if x.strip()]
    difficulties = ["easy", "medium", "hard"]

    total = 0
    with out.open("w", encoding="utf-8") as f:
        for domain in domains():
            for diff in difficulties:
                for i in range(args.n_per_domain):
                    for mode in eval_modes:
                        sample = make_sample(domain, i, diff, mode)
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        total += 1

    print(f"Wrote {total} samples to {out}")


if __name__ == "__main__":
    main()