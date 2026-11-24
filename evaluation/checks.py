"""Checks of an answer that need no LLM and no product code: the exact-match metrics, the rule checks and chrF of
domain descriptions, and the two relaxed gates of the email-template score.

Not here, because they are production code and configuration of the products: the LLM judge's instructions and
the email-template rubric (its 13 categories, rendering and LLM judges). The functions below are the parts of the
paper's scoring that sit outside them.

    exact-match tasks        score_website_cues, score_brand, score_categorisation
    domain descriptions      domain_checks, breaks_rule (the judge-free "rules" metric of the paper), chrf
    email templates          relaxed_html_complete, relaxed_unsubscribe (the relaxed gates)

Answers and references are the raw text of the model's answer and of the reference answer.
"""

import json
import re


def parse_json(text):
    """The answer as JSON, tolerating a ```json fence (as the production checks do); None if it does not parse."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0] if "\n" in t else t.strip("`")
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None


def norm(s) -> str:
    return " ".join(str(s).split()).casefold() if s is not None else ""


def chrf(hyp, ref) -> float | None:
    """chrF (sacrebleu's defaults) of one answer against one reference."""
    from sacrebleu.metrics import CHRF

    if not isinstance(hyp, str) or not isinstance(ref, str):
        return None
    return round(CHRF().sentence_score(hyp, [ref]).score, 2)


# ------------------------------------------------------------------------------------------------- exact-match tasks

def score_website_cues(answer: str, reference: str, keys=("isColorMentioned", "isFontMentioned")) -> dict:
    """Website cues: the answer is a JSON object with one boolean per key. `correct` (the primary metric) needs
    every key to equal the reference; `valid` needs exactly these keys, all boolean."""
    ref, out = parse_json(reference), parse_json(answer)
    pred = {k: out.get(k) if isinstance(out, dict) else None for k in keys}
    valid = isinstance(out, dict) and set(out) == set(keys) and all(isinstance(out[k], bool) for k in keys)
    return {"valid": valid, "pred": pred, "correct": all(pred[k] is ref[k] for k in keys)}


def score_brand(answer: str, reference: str) -> dict:
    """Brand names: the answer is {"brand_name": ..., "source": "detected" | "generated"}. `correct` (the primary
    metric) needs source "detected" and the reference's name, character for character. `detected` is the paper's
    detection; precision is `correct` among the detected answers.

    A plain answer that is only a name (one line, no JSON; surrounding quotes or backticks removed) counts when
    the name matches: without a schema a model often answers just the name. Under the schema of the paper's runs
    every answer is JSON."""
    ref, out = parse_json(reference), parse_json(answer)
    target = ref["brand_name"]
    if isinstance(out, dict):
        name, source = out.get("brand_name"), out.get("source")
        valid = isinstance(name, str) and source in ("detected", "generated")
        found = valid and source == "detected"
    else:
        plain = (answer or "").strip().strip("`\"'“”«»").strip()
        is_name = bool(plain) and "\n" not in plain and len(plain) <= 120
        name, source = (plain if is_name else None), None
        valid = False  # not the production format
        found = is_name
    return {"valid": valid, "pred_name": name, "pred_source": source, "detected": source == "detected",
            "correct": found and name == target, "lenient_correct": found and norm(name) == norm(target)}


def score_categorisation(answer: str, reference: str) -> dict:
    """Categorisation: the answer is {"category": ...}. `exact` is the judge-free metric of the paper: the
    category equals the reference's after whitespace and case normalisation. (The primary metric of this task is
    the judge's verdict.)"""
    ref, out = parse_json(reference), parse_json(answer)
    cat = out.get("category") if isinstance(out, dict) else None
    valid = isinstance(cat, str) and bool(cat.strip())
    return {"valid": valid, "pred_category": cat, "exact": valid and norm(cat) == norm(ref["category"])}


# ----------------------------------------------------------------------------------------------- domain descriptions

def lingua_detect(text: str) -> str | None:
    """ISO 639-1 code of `text` with the lingua library (all its languages), or None for a text under 40
    characters. A stand-in for the detector of the paper's runs, which is part of the rubric library."""
    from lingua import LanguageDetectorBuilder

    global _DETECTOR
    if len((text or "").strip()) < 40:
        return None
    if _DETECTOR is None:
        _DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
    lang = _DETECTOR.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else None


_DETECTOR = None


def domain_checks(answer: str, requested_language: str, domain: str, detect=None) -> dict:
    """The checks of one domain description: a one- or two-sentence marketing text for `domain` in
    `requested_language` (lower-case ISO 639-1 code), which the prompt asks to open with "This domain is" in that
    language. `detect` maps a text to its language code or None (for example `lingua_detect`); without it the
    language is not checked."""
    answer = (answer or "").strip()
    domain = str(domain or "")
    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", answer) if s.strip()]
    detected = detect(answer) if detect else None
    return {
        "detected_language": detected,
        "language_ok": None if detected is None else detected == requested_language,
        "repeats_domain": bool(domain) and domain.lower() in answer.lower(),
        "sentences": len(sentences),
        "max_words_per_sentence": max((len(s.split()) for s in sentences), default=0),
        # the English opener copied into an answer in another language ("This domain adalah ...")
        "english_opening": requested_language != "en" and answer.startswith("This "),
    }


def breaks_rule(checks: dict, capped: bool = False) -> bool:
    """The paper's judge-free "rules" metric is the share of answers for which this is False: the detected
    language is the requested one, the domain name is not repeated, at most two sentences of at most 20 words, no
    English opener in a non-English answer, and the answer did not run to the token cap (`capped`). An answer with
    no detected language is not counted as a failure."""
    return bool(checks["language_ok"] is False or checks["repeats_domain"] or checks["sentences"] > 2
                or checks["max_words_per_sentence"] > 20 or checks["english_opening"] or capped)


# ------------------------------------------------------------------------------------------- email templates: gates
# The production rubric sets the score of an answer to 0 when one of five hard gates fails. Two of them were
# relaxed for the paper's primary email score (the production score is reported next to it):
#   html_complete     the production gate requires the <html> wrapper. Relaxed: a gate that failed because the
#                     wrapper is missing passes when `relaxed_html_complete(html)` holds, that is, for a complete
#                     <body>...</body>. Bare fragments and truncated HTML still fail.
#   unsubscribe_link  the production gate requires the template's placeholder link. Relaxed: any <a> whose text or
#                     href says "unsubscribe" in one of the dataset's languages passes, for example
#                     <a href="#">Unsubscribe</a>.
# The score is then aggregated again from the stored checks; nothing else changes.

def relaxed_html_complete(html: str) -> bool:
    h = html.lower()
    return "<body" in h and ("</body>" in h or "</html>" in h)


UNSUBSCRIBE = re.compile("|".join([
    r"unsub", r"opt[\s-]?out", r"stop receiving",                                                    # en
    r"\bde baja\b", r"desuscri", r"(?:anular|cancelar) (?:la |tu |su )?suscripci",                    # es
    r"descadastr", r"desinscrev", r"sair da lista", r"cancelar (?:a |sua )?(?:inscri|assinatura)",    # pt
    r"d[ée]sabonn", r"d[ée]sinscri",                                                                  # fr
    r"abmeld", r"abbestell", r"austragen",                                                            # de
    r"disiscri", r"(?:annulla|cancella)(?:re)? (?:l'|la )?(?:iscrizione|abbonamento)",                # it
    r"uitschrijv", r"afmeld",                                                                         # nl, da
    r"wypisz", r"wypisa", r"zrezygnuj", r"rezygnacja z",                                              # pl
    r"odhl[áa]s", r"leiratkoz", r"dezabon",                                                           # cs/sk, hu, ro
    r"abonelik(?:ten)? ?[çc][ıi]k", r"abonelikten ayr[ıi]l", r"aboneli[ğg]i (?:iptal|sonland)",        # tr
    r"atsisaky", r"atsisak",                                                                          # lt
    r"berhenti (?:ber)?langganan", r"henti langganan", r"batal(?:kan)? langganan", r"nyahlanggan",    # id, ms
    r"h[ủu]y (?:đ[ăa]ng k[ýy]|nh[ậa]n)", r"ng[ừu]ng nh[ậa]n",                                        # vi
    r"peruuta tilaus", r"lopeta tilaus", r"tilauksen peruutus", r"loobu",                             # fi, et
    r"avslutt abonnement", r"meld (?:deg|dig) av", r"avsluta prenumeration", r"avregistrer",          # no, sv
    r"إلغاء الاشتراك", r"الغاء الاشتراك", r"إلغاء اشتراك",                                             # ar
    r"退订", r"退訂", r"取消订阅", r"取消訂閱", r"配信停止", r"購読解除", r"구독 ?취소", r"수신 ?거부",  # zh, ja, ko
    r"ยกเลิกการ(?:สมัคร|รับ|ติดตาม)", r"ยกเลิกรับ",                                                    # th
    r"सदस्यता (?:रद्द|समाप्त|छोड़)", r"अनसब्सक्राइब", r"সদস্যতা (?:ত্যাগ|বাতিল)", r"আনসাবস্ক্রাইব",       # hi, bn
    r"ان ?سبسکرائب", r"رکنیت (?:ختم|منسوخ)",                                                           # ur
    r"διαγραφ", r"κατάργηση εγγραφής", r"απεγγραφ", r"הסר(?:ה)? מ", r"ביטול (?:ה)?(?:מנוי|הרשמה)",    # el, he
    r"отпис", r"відпис", r"отписване", r"[çc]regjistr", r"[çc]abonoh", r"donar-?se de baixa",         # ru, uk, bg, sq, ca
    r"odjav",                                                                                         # hr/sl/sr
]), re.I)


def relaxed_unsubscribe(html: str) -> bool:
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # answers that start with <?xml ...?>
    for a in BeautifulSoup(html, "html.parser").find_all("a"):
        href = a.get("href") or ""
        if "{{unsubscribe-link}}" in href or UNSUBSCRIBE.search(a.get_text(" ", strip=True)) or UNSUBSCRIBE.search(href):
            return True
    return False
