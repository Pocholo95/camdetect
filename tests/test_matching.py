import numpy as np

from app import db
from app.recognizer import Recognizer, l2_normalize


def _rand_unit_vector(rng, dim=512):
    v = rng.normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)


def test_margin_rule_rejects_ambiguous_match(conn):
    """Spec 12.3: dos embeddings casi identicos de personas distintas -> Desconocido."""
    rng = np.random.default_rng(42)
    base = _rand_unit_vector(rng)
    noise = rng.normal(scale=0.01, size=512).astype("float32")
    almost_same = l2_normalize(base + noise)

    p1 = db.create_person(conn, "Persona A", is_known=True)
    p2 = db.create_person(conn, "Persona B", is_known=True)
    db.add_embedding(conn, p1, base, condition="day", source="upload")
    db.add_embedding(conn, p2, almost_same, condition="day", source="upload")

    recognizer = Recognizer(load_model=False)
    query = l2_normalize(base + rng.normal(scale=0.02, size=512).astype("float32"))

    person_id, score = recognizer.match_known(conn, query)

    assert person_id is None, "un empate entre dos personas debe resultar en Desconocido"


def test_clear_match_above_threshold_and_margin(conn):
    rng = np.random.default_rng(7)
    known_vec = _rand_unit_vector(rng)
    unrelated_vec = _rand_unit_vector(rng)

    p1 = db.create_person(conn, "Persona C", is_known=True)
    p2 = db.create_person(conn, "Persona D", is_known=True)
    db.add_embedding(conn, p1, known_vec, condition="day", source="upload")
    db.add_embedding(conn, p2, unrelated_vec, condition="day", source="upload")

    recognizer = Recognizer(load_model=False)
    query = l2_normalize(known_vec + rng.normal(scale=0.005, size=512).astype("float32"))

    person_id, score = recognizer.match_known(conn, query)

    assert person_id == p1
    assert score > 0.9


def test_score_below_sim_threshold_is_unknown(conn):
    rng = np.random.default_rng(11)
    known_vec = _rand_unit_vector(rng)
    p1 = db.create_person(conn, "Persona E", is_known=True)
    db.add_embedding(conn, p1, known_vec, condition="day", source="upload")

    recognizer = Recognizer(load_model=False)
    unrelated_query = _rand_unit_vector(rng)

    person_id, score = recognizer.match_known(conn, unrelated_query)

    assert person_id is None
