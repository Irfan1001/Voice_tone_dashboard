"""Role resolution edge cases: who is the customer, and when do we refuse to say.

Model-free and audio-free: these exercise the two pure functions that make the
decision, so the whole surface is testable without pyannote or Whisper.

Getting the role wrong is worse than not resolving it - emotion would be measured
on the agent and reported as the customer's - so most cases here are about
REFUSING to decide: ties, thin margins, single speakers, empty transcripts.
"""

from __future__ import annotations

from app.features.roles import pick_customer, score_text

# Real ASR output from the trial calls, with the agent and dealership names replaced
# by placeholders. Only the SCRIPT SHAPE is load-bearing - the patterns deliberately
# match none of the names, which `test_names_and_companies_never_match` pins down.
AGENT_001 = "Hi, I'm Dana from Northgate Motors. How can I help?"
CUSTOMER_001 = "Yes, hi, are you a real person? Hello, hello, hello"
AGENT_003 = ("Hi, I'm Dana from Northgate, Motors. How can I help? I can help with "
             "that. What type of service do you need?")
CUSTOMER_003 = "Dana, I want an appointment for my car. Please, I need to have an order."


# --------------------------------------------------------------- score_text


def test_agent_greeting_scores_agent_patterns():
    s = score_text(AGENT_001)
    assert s.agent >= 2
    assert s.customer == 0


def test_scoring_is_case_insensitive():
    assert score_text("HOW CAN I HELP YOU TODAY?").agent >= 1


def test_names_and_companies_never_match():
    """The patterns must key on script SHAPE, not on who or where. The agent name and
    dealership appear in every transcript and would silently become the detector if
    anyone added them."""
    assert score_text("Dana Northgate Motors Riverside Branch").agent == 0


def test_customer_phrases_are_scored_separately():
    s = score_text("I'm calling about my car, it's not working and I need to fix it")
    assert s.customer >= 2
    assert s.agent == 0


def test_empty_transcript_scores_nothing():
    s = score_text("")
    assert (s.agent, s.customer, s.strong) == (0, 0, 0)


# ------------------------------------------------------------ pick_customer


def two(a_text: str, b_text: str):
    return {"A": score_text(a_text), "B": score_text(b_text)}


def test_clear_agent_greeting_resolves_the_roles():
    customer, agent, margin = pick_customer(two(AGENT_001, CUSTOMER_001), min_margin=1)
    assert (agent, customer) == ("A", "B")
    assert margin >= 2


def test_real_transcripts_from_call_003_resolve_correctly():
    customer, agent, _ = pick_customer(two(AGENT_003, CUSTOMER_003), min_margin=1)
    assert (agent, customer) == ("A", "B")


def test_a_tie_refuses_to_decide():
    """Two identical openings cannot be told apart. Returning either one would be a
    coin flip presented as a measurement."""
    customer, agent, _ = pick_customer(two(AGENT_001, AGENT_001), min_margin=1)
    assert customer == "" and agent == ""


def test_neither_speaker_scripted_refuses_to_decide():
    customer, agent, _ = pick_customer(two("Hello?", "Yeah hi"), min_margin=1)
    assert customer == "" and agent == ""


def test_margin_below_threshold_refuses_to_decide():
    scores = two(AGENT_001, CUSTOMER_001)
    customer, agent, _ = pick_customer(scores, min_margin=99)
    assert customer == "" and agent == ""


def test_single_speaker_refuses_to_decide():
    customer, agent, _ = pick_customer({"A": score_text(AGENT_001)}, min_margin=1)
    assert customer == "" and agent == ""


def test_customer_stating_their_name_is_not_mistaken_for_the_agent():
    """Customers say "my name is" constantly, because they are asked for it. If ASR
    misses the agent's greeting, a customer answering "My name is John Smith" would
    out-score the agent and invert the roles. Hence: a decision requires at least
    one STRONG pattern - something only an agent says.
    """
    scores = two("My name is John Smith", "mm hm, okay")
    customer, agent, _ = pick_customer(scores, min_margin=1)
    assert customer == "" and agent == "", (
        "an ambiguous phrase alone must not decide the role")


def test_a_strong_pattern_still_decides_even_with_a_thin_margin():
    """The strong requirement must not make the detector useless: one unambiguous
    agent phrase against a silent customer is still a decision."""
    customer, agent, margin = pick_customer(
        two("How can I help you?", "yeah"), min_margin=1)
    assert (agent, customer) == ("A", "B")


def test_customer_phrases_reduce_a_speakers_agent_score():
    """Differential scoring: a speaker who sounds like a customer should not win on
    an incidental agent-ish phrase."""
    s = score_text("I'm calling about my car, let me check my order number")
    assert s.customer > 0
    assert s.agent - s.customer < 1
