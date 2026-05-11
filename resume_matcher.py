import math

SKILL_ALIASES = {
    "python": "python", "pyhton": "python", "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp", "r": "r", "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

RESUMES = [
    ("Arjun Sharma",   "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",     "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",    "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",    "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",   "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan","javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",    "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",    "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",   "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",     "python, R, statistics, ML, regression, clustering, Power-BI"),
]

JDS = [
    ("JD-1", "Kakao", "ML Engineer",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization",
     "NLP, BERT, Feature Engineering, Statistics"),
    ("JD-2", "Naver", "Backend Engineer",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes",
     "REST API, CI/CD, Redis"),
    ("JD-3", "Line", "Frontend Engineer",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS",
     "Node.js, GraphQL, Redux, Jest, AWS"),
]


def normalize_skills(raw):
    s = raw.lower()
    tokens = [t.strip() for t in s.split(",")]
    result = []
    for tok in tokens:
        matched = False
        for phrase in sorted(SKILL_ALIASES.keys(), key=lambda x: -len(x)):
            if len(phrase.split()) > 1 and tok == phrase:
                result.append(SKILL_ALIASES[phrase])
                matched = True
                break
        if not matched:
            if tok in SKILL_ALIASES:
                result.append(SKILL_ALIASES[tok])
    seen = set()
    deduped = []
    for sk in result:
        if sk not in seen:
            seen.add(sk)
            deduped.append(sk)
    return deduped


def build_vocabulary(normalized_resumes):
    vocab_set = set()
    for _, skills in normalized_resumes:
        for sk in skills:
            vocab_set.add(sk)
    return sorted(vocab_set)


def compute_df(normalized_resumes, vocab):
    df = {sk: 0 for sk in vocab}
    for _, skills in normalized_resumes:
        skill_set = set(skills)
        for sk in vocab:
            if sk in skill_set:
                df[sk] += 1
    return df


def compute_tfidf(skills, vocab, df):
    N = len(skills)
    skill_set = set(skills)
    vector = []
    for sk in vocab:
        if sk in skill_set:
            tf = 1.0 / N
            idf = math.log(10.0 / df[sk])
            vector.append(tf * idf)
        else:
            vector.append(0.0)
    return vector


def build_jd_vector(jd_required, jd_preferred, vocab):
    combined = normalize_skills(jd_required) + normalize_skills(jd_preferred)
    jd_set = set(combined)
    return [1 if sk in jd_set else 0 for sk in vocab]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def rank_candidates(tfidf_vectors, jd_vector, names, top_n=3):
    scores = []
    for i, vec in enumerate(tfidf_vectors):
        sim = cosine_similarity(vec, jd_vector)
        scores.append((names[i], round(sim, 2)))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_n]


def main():
    normalized_resumes = [(name, normalize_skills(raw)) for name, raw in RESUMES]

    vocab = build_vocabulary(normalized_resumes)
    df = compute_df(normalized_resumes, vocab)

    tfidf_vectors = [
        compute_tfidf(skills, vocab, df)
        for _, skills in normalized_resumes
    ]
    names = [name for name, _ in normalized_resumes]

    for jd_id, company, role, required, preferred in JDS:
        jd_vec = build_jd_vector(required, preferred, vocab)
        top3 = rank_candidates(tfidf_vectors, jd_vec, names)
        print(f"{jd_id} — {company} ({role})")
        print(", ".join(f"{name}({score})" for name, score in top3))
        print()


if __name__ == "__main__":
    main()
