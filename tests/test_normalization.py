"""Unit tests for src/normalization.py (Phase 2).

Run from the project root:
    .venv/bin/python -m unittest discover -s tests -t . -v
(pytest-compatible as well). Examples are modelled on the Phase 1 EDA findings.
"""
import doctest
import unittest

from src import normalization as N
from src.normalization import (
    alnum_tokens, char_ngrams, detect_script, extract_address_features, extract_name_features,
    normalize_address, normalize_basic_text, normalize_name, normalize_state, numeric_tokens,
    phonetic_key, script_breakdown, strip_latin_accents, transliterate_text, word_tokens,
)


class TestBasicText(unittest.TestCase):
    def test_spec_example(self):
        self.assertEqual(normalize_basic_text("  ACME   Pvt. Ltd. "), "acme pvt ltd")

    def test_capitalisation_and_whitespace(self):
        self.assertEqual(normalize_basic_text("DIVISION OF  CONSUMER\tAFFAIRS"), "division of consumer affairs")

    def test_dotted_abbreviations_collapse(self):
        self.assertEqual(normalize_basic_text("Ram All Consulting L.L.P."), "ram all consulting llp")
        self.assertEqual(normalize_basic_text("Meredithe T. Igoe, MD, P.C."), "meredithe t igoe, md, pc")

    def test_decimal_and_website_kept(self):
        self.assertEqual(normalize_basic_text("Version 2.5"), "version 2.5")
        self.assertEqual(normalize_basic_text("ZIONTERM.COM"), "zionterm.com")

    def test_missing_values(self):
        for v in (None, "", "   ", float("nan")):
            self.assertEqual(normalize_basic_text(v), "")
        for v in (None, "", " NULL ", "N/A", "<NULL>", "na", float("nan")):
            self.assertTrue(N.is_missing_value(v), v)
        self.assertFalse(N.is_missing_value("Nagpur"))

    def test_unicode_quotes_zero_width(self):
        self.assertEqual(normalize_basic_text("Wenonah’s​ Metal"), "wenonah's metal")

    def test_indic_text_untouched(self):
        s = "इंडियन इंटरनेशनल प्राइवेट लिमिटेड"
        self.assertEqual(normalize_basic_text(s), s)


class TestAccents(unittest.TestCase):
    def test_latin_accents_removed(self):
        self.assertEqual(strip_latin_accents("Division óf Cónsumer Térm Índia"), "Division of Consumer Term India")

    def test_indic_marks_preserved(self):
        for s in ("लिमिटेड", "ಕರ್ನಾಟಕ", "ગુજરાત", "தமிழ்நாடு"):
            self.assertEqual(strip_latin_accents(s), s)

    def test_mixed(self):
        self.assertEqual(strip_latin_accents("Résort ગુજરાત"), "Resort ગુજરાત")


class TestScriptDetection(unittest.TestCase):
    def test_families(self):
        self.assertEqual(detect_script("Peak Software LLC"), "latin")
        self.assertEqual(detect_script("Cónsumer"), "latin")
        self.assertEqual(detect_script("राम मार्केटिंग प्राइवेट लिमिटेड"), "indic")
        self.assertEqual(detect_script("North ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్"), "mixed")
        self.assertEqual(detect_script("Москва"), "other")
        for v in (None, "", "123 - #", float("nan")):
            self.assertEqual(detect_script(v), "unknown")

    def test_not_only_devanagari(self):
        cases = {"बिहार": "devanagari", "পশ্চিমবঙ্গ": "bengali", "ਪੰਜਾਬ": "gurmukhi", "ગુજરાત": "gujarati",
                 "ଓଡ଼ିଶା": "oriya", "தமிழ்நாடு": "tamil", "తెలంగాణ": "telugu", "ಕರ್ನಾಟಕ": "kannada",
                 "കേരളം": "malayalam"}
        for text, script in cases.items():
            self.assertEqual(detect_script(text), "indic", text)
            self.assertEqual(set(script_breakdown(text)), {script}, text)


class TestTransliteration(unittest.TestCase):
    def test_legal_words_all_scripts(self):
        limited = ["लिमिटेड", "লিমিটেড", "લિમિટેડ", "ଲିମିଟେଡ୍", "లిమిటెడ్", "ಲಿಮಿಟೆಡ್"]
        for w in limited:
            self.assertEqual(transliterate_text(w), "limited", w)
        self.assertEqual(transliterate_text("प्राइवेट"), "praivet")
        for w in ("एलएलपी", "ఎల్‌ఎల్‌పీ", "എൽഎൽപി", "ಎಲ್ಎಲ್‌ಪಿ"):
            self.assertEqual(transliterate_text(w), "elelpi", w)

    def test_schwa_handling(self):
        self.assertEqual(transliterate_text("इंडियन"), "indiyan")        # final schwa dropped
        self.assertEqual(transliterate_text("ग्लोबल"), "global")
        self.assertEqual(transliterate_text("टेक"), "tek")

    def test_script_specific_rules(self):
        self.assertEqual(transliterate_text("ஃபுட்ஸ்"), "futs")             # Tamil aytham + pa = f
        self.assertEqual(transliterate_text("ഇന്റർനാഷണൽ"), "intarnashanal")  # Malayalam nta, chillus
        self.assertEqual(transliterate_text("ਐਲਐਲਪੀ"), "elelpi")            # Gurmukhi ai ~ e

    def test_latin_passthrough_and_mixed(self):
        self.assertEqual(transliterate_text("Cónsumer Affairs"), "consumer affairs")
        self.assertEqual(transliterate_text("North ఇంటర్నేషనల్"), "north intarneshanal")
        self.assertEqual(transliterate_text(None), "")

    def test_indic_digits(self):
        self.assertEqual(transliterate_text("१२३ ౪౫"), "123 45")

    def test_phonetic_key_cross_script(self):
        self.assertEqual(phonetic_key("private"), phonetic_key("प्राइवेट"))
        self.assertEqual(phonetic_key("global"), phonetic_key("ग्लोबल"))
        self.assertEqual(phonetic_key("trading"), phonetic_key("டிரேடிங்"))          # Tamil: no voicing contrast
        self.assertEqual(phonetic_key("investment"), phonetic_key("ଇନଭେଷ୍ଟମେଣ୍ଟ୍"))  # Oriya: v written as bh
        self.assertEqual(phonetic_key("eastern"), phonetic_key("इस्टर्न"))
        self.assertEqual(phonetic_key(""), "")


class TestNames(unittest.TestCase):
    def test_representations_kept(self):
        n = normalize_name("*** Wenonah'S Metal Works")
        self.assertEqual(n.raw, "*** Wenonah'S Metal Works")
        self.assertEqual(n.basic, "*** wenonah's metal works")
        self.assertEqual(n.punct, "wenonahs metal works")
        self.assertEqual(n.core_latin, "wenonahs metal works")
        self.assertEqual(n.compact, "wenonahsmetalworks")

    def test_legal_form_variants_agree(self):
        variants = ["Wonderland Energy Private Limited", "Wonderland Energy Pvt. Ltd.", "Wonderland Energy-Private-Limited",
                    "Private Wonderland Energy Ltd", "Wonderland Energy Pvt Limited", "Wonderland Energy (P) Ltd"]
        cores = {normalize_name(v).core_latin for v in variants}
        self.assertEqual(cores, {"wonderland energy"})
        self.assertEqual(normalize_name("Wonderland Energy Pvt. Ltd.").legal_forms, ("private limited",))
        self.assertEqual(normalize_name("Wonderland Energy Pvt. Ltd.").canonical, "wonderland energy private limited")

    def test_us_legal_forms(self):
        self.assertEqual(normalize_name("Ridge Entergy (PLLC)").core_latin, "ridge entergy")
        self.assertEqual(normalize_name("Ridge Entergy Pllc").legal_forms, ("pllc",))
        self.assertEqual(normalize_name("Ram All Consulting L.L.P.").core_latin, "ram all consulting")
        self.assertEqual(normalize_name("Inc Family Mdwemst Associates").core_latin, "family mdwemst associates")
        self.assertEqual(normalize_name("Land & Mcbride Incorporated").core_latin, "land and mcbride")
        self.assertEqual(normalize_name("J.P. Morgan & Co").core_latin, "jp morgan")

    def test_conservative_legal_removal(self):
        # 'co' / 'company' are only removed at the end; a lone legal word is never emptied
        self.assertEqual(normalize_name("Co Op Bank").core_latin, "co op bank")
        self.assertEqual(normalize_name("Company").core_latin, "company")
        self.assertEqual(normalize_name("Limited Sai Private Center").core_latin, "sai private center")
        self.assertEqual(normalize_name("Consolidated Education Systems Inc").core_latin, "consolidated education systems")

    def test_indic_legal_forms(self):
        n = normalize_name("इंडियन इंटरनेशनल प्राइवेट लिमिटेड")
        self.assertEqual(n.legal_forms, ("private limited",))
        self.assertEqual(n.core, "इंडियन इंटरनेशनल")                       # original script kept
        self.assertEqual(n.core_latin, "indiyan intarneshnal")
        self.assertEqual(normalize_name("ईस्ट सॉल्यूशंस प्रा. लि.").legal_forms, ("private limited",))
        self.assertEqual(normalize_name("રામ ઓલ કન્સલ્ટિંગ એલએલપી").legal_forms, ("llp",))

    def test_aliases(self):
        n = normalize_name("Quodova a/k/a Indian International Private Limited")
        self.assertEqual(n.alias_marker, "a/k/a")
        self.assertEqual(n.aliases, ("quodova", "indian international"))
        self.assertIn("indian international", n.variants)
        self.assertEqual(n.raw, "Quodova a/k/a Indian International Private Limited")
        self.assertEqual(normalize_name("Nexgildcira formerly known as Kritaya Works Private Limited").aliases,
                         ("nexgildcira", "kritaya works"))
        self.assertEqual(normalize_name("Dovaquo DBA: #Zion Term").aliases, ("dovaquo", "zion term"))
        self.assertEqual(normalize_name("Umbrakeloveo t/a Meredithe Igoe").alias_marker, "t/a")
        self.assertEqual(normalize_name("Tavowex aka FC Rapid Therapy LLC").aliases, ("tavowex", "fc rapid therapy"))

    def test_alias_words_that_are_not_aliases(self):
        for name in ("Dba Brothers Pvt Ltd", "Aka Holdings", "T/A Post Inc."):
            self.assertIsNone(normalize_name(name).alias_marker, name)

    def test_websites(self):
        n = normalize_name("zionterm.com")
        self.assertEqual((n.website, n.website_label, n.core_latin), ("zionterm.com", "zionterm", "zionterm"))
        self.assertEqual(normalize_name("... PEAKBNY.COM").website_label, "peakbny")
        self.assertEqual(normalize_name("www.acme-tools.co.in").website_label, "acmetools")
        self.assertEqual(normalize_name("5ERVICESNAGESHWARWELFARE.COM").website_label, "servicesnageshwarwelfare")
        self.assertEqual(normalize_name("ílluminatifinvest.com").website_label, "illuminatifinvest")
        self.assertEqual(normalize_name("premiersons.com").website_label, normalize_name("Premier & Sons Private Limited").compact)
        self.assertEqual(normalize_name("wenonahsmetalworks.com").website_label,
                         normalize_name("Wenonah's Metal Works").compact)

    def test_not_websites(self):
        for name in ("St. Louis Bakery", "J.P. Morgan", "Vinayaka.Plaza Traders", "No.1 Tailors"):
            self.assertIsNone(normalize_name(name).website, name)

    def test_junk_ocr_accents(self):
        self.assertEqual(normalize_name("... 5ervices Nageshwar Welfare Private Limited").core_latin,
                         "services nageshwar welfare")
        self.assertEqual(normalize_name("Wilcox, Roseline, DDS").core_latin, normalize_name("wi1cox, roseline, dds").core_latin)
        self.assertEqual(normalize_name("#ZION TÉRM").core_latin, "zion term")
        self.assertEqual(normalize_name(">> CRYSTAL TEXTILE").core_latin, "crystal textile")
        self.assertEqual(normalize_name("3rd Street Deli").core_latin, "3rd street deli")   # ordinals untouched

    def test_the_and_ms_prefix(self):
        self.assertEqual(normalize_name("The Fresh Deli Care").core_latin, "fresh deli care")
        self.assertEqual(normalize_name("M/s. Sharma Traders").core_latin, "sharma traders")

    def test_mixed_script_name(self):
        n = normalize_name("North ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్")
        self.assertEqual(n.script, "mixed")
        self.assertEqual(n.core_latin, "north intarneshanal")
        self.assertEqual(n.legal_forms, ("private limited",))

    def test_missing_names(self):
        for v in (None, "", "   ", float("nan")):
            n = normalize_name(v)
            self.assertEqual((n.core_latin, n.script, n.legal_forms), ("", "unknown", ()))

    def test_features(self):
        f = extract_name_features("Dovaquo DBA: #Zion Térm")
        self.assertTrue(f["name_has_alias"])
        self.assertTrue(f["name_has_latin_accent"])
        self.assertFalse(f["name_is_website"])
        f = extract_name_features("एपेक्स मीडिया लिमिटेड")
        self.assertTrue(f["name_is_indic"])
        self.assertEqual(f["name_scripts"], "devanagari")
        self.assertEqual(f["name_legal_forms"], "limited")


class TestStates(unittest.TestCase):
    def test_us(self):
        self.assertEqual(normalize_state("West Virginia", "US"), "wv")
        self.assertEqual(normalize_state("WV", "US"), "wv")
        self.assertEqual(normalize_state("IN", "US"), "in")
        self.assertIsNone(normalize_state("Springfield", "US"))

    def test_india(self):
        self.assertEqual(normalize_state("Maharashtra", "India"), "mh")
        self.assertEqual(normalize_state("MH", "India"), "mh")
        self.assertEqual(normalize_state("महाराष्ट्र", "India"), "mh")
        self.assertEqual(normalize_state("Orissa", "India"), normalize_state("Odisha", "India"))
        self.assertEqual(normalize_state("ଓଡ଼ିଶା", "India"), "od")
        self.assertEqual(normalize_state("TS", "India"), "tg")
        self.assertEqual(normalize_state("Keralam", "India"), "kl")

    def test_ambiguous_codes_need_country(self):
        self.assertEqual(normalize_state("GA", "US"), "ga")
        self.assertEqual(normalize_state("Goa", "India"), "ga")
        self.assertIsNone(normalize_state("AR"))                     # Arkansas vs Arunachal Pradesh
        self.assertEqual(normalize_state("Texas"), "tx")             # unambiguous without country


class TestAddresses(unittest.TestCase):
    def test_zero_padding_suffix_state(self):
        a = normalize_address("00123 YEAGER RD, COALLTON, WV", "US")
        b = normalize_address("123 Yeager Road, Coallton, West Virginia", "US")
        self.assertEqual(a.normalized, b.normalized)
        self.assertEqual((a.house_number_core, a.street, a.city, a.state), ("123", "yeager rd", "coallton", "wv"))
        self.assertEqual(a.raw, "00123 YEAGER RD, COALLTON, WV")

    def test_reordered_components(self):
        a = normalize_address("Point Pleasant, 515 Kitty Hawk Lane, WV", "US")
        b = normalize_address("00515 Kitty Hawk Ln, Point Pleasant, West Virginia", "US")
        self.assertEqual(a.component_set, b.component_set)
        self.assertNotEqual(a.components, b.components)            # order is preserved, not sorted
        self.assertEqual((a.house_number_core, a.street, a.city, a.state), (b.house_number_core, b.street, b.city, b.state))

    def test_house_number_variants(self):
        self.assertEqual(normalize_address("Hastings On Hudson, New York, 181B Farragut Avenue", "US").house_number_core, "181")
        self.assertEqual(normalize_address("181 Farragut Avenue, Hastings-on-hudson, NY", "US").house_number_core, "181")
        a = normalize_address("3700-3702 F M RD 1187, BUURLESON, TX", "US")
        self.assertEqual((a.house_number, a.house_number_core), ("3700-3702", "3700"))
        self.assertEqual(normalize_address("H.NO. 69, FARIDABAD, Haryana", "India").house_number, "69")
        self.assertEqual(normalize_address("C-71, DELHI, दिल्ली", "India").house_number, "c-71")
        self.assertEqual(normalize_address("C-##45, AJMER ROAD, Rajasthan", "India").house_number, "c-45")

    def test_noise_tokens(self):
        a = normalize_address("8144 24th Avenue, NULL, Seattle, Washington", "US")
        self.assertEqual(a.normalized, "8144 24th ave, seattle, wa")
        self.assertEqual(a.noise_removed, 1)
        b = normalize_address("A-1501, null, Borivali West, Mumbai, MH", "India")
        self.assertEqual(b.state, "mh")
        self.assertNotIn("null", b.tokens)
        c = normalize_address("HOUSE NO3-1621, BATHINDA, <NULL>, Punjab", "India")
        self.assertEqual((c.state, c.city), ("pb", "bathinda"))
        d = normalize_address("##36049 HAVERFORD PLACE, AVON, OH", "US")
        self.assertEqual(d.house_number, "36049")
        self.assertEqual(normalize_address("Landers Chapel Road, N/A, Lincolnton, North Carolina", "US").components,
                         ("landers chapel rd", "lincolnton", "nc"))

    def test_missing(self):
        for v in (None, "", "   ", "N/A", "null", "<NULL>", float("nan")):
            a = normalize_address(v, "US")
            self.assertTrue(a.is_missing, v)
            self.assertEqual(a.components, ())

    def test_street_suffixes_directions_ordinals(self):
        self.assertEqual(normalize_address("1245 NINTH ST, TERRE HAUTE, IN", "US").normalized,
                         normalize_address("1245 9th Street, Terre Haute, Indiana", "US").normalized)
        self.assertEqual(normalize_address("East Haddam, CT", "US").city, "e haddam")
        self.assertEqual(normalize_address("CT, LEDGEBROOK RD, E HADDAM", "US").city, "e haddam")

    def test_unit_pmb_pobox(self):
        a = normalize_address("8144 24TH AVE, PMB 8686, SEATTLE, WA", "US")
        self.assertEqual((a.unit, a.city), ("pmb 8686", "seattle"))
        b = normalize_address("7211 RALPH ST, PO BOX 1903, NORFOLK, VA", "US")
        self.assertEqual((b.po_box, b.house_number_core), ("1903", "7211"))
        c = normalize_address("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US")
        self.assertEqual(c.unit, "unit apt g")
        d = normalize_address("P O Box No: 1117 4/1062, Beach Road, Calicut, Kozhikode, Kerala", "India")
        self.assertEqual((d.po_box, d.state), ("1117", "kl"))
        self.assertIsNone(d.house_number)

    def test_places(self):
        a = normalize_address("Room-1, C-32A 3Rd Floor, Friends Colony (East), Delhi, South Delhi, Delhi", "India")
        self.assertEqual(a.state, "dl")
        self.assertEqual(a.city, "s delhi")
        self.assertEqual(a.places, ("friends colony e", "delhi", "s delhi"))
        self.assertEqual(normalize_address("Cardiology, 1324 Yale, MESA CITY, Arizona", "US").city, "mesa")

    def test_native_script_state(self):
        a = normalize_address("OFFICE NO 203, PLOT NO 19, SATRA PLAZA, VASHI THANE, THANE, महाराष्ट्र", "India")
        b = normalize_address("Thane, Plot No 19, Satra Plaza, Vashi Thane, MH, Office No 203", "India")
        self.assertEqual((a.state, b.state), ("mh", "mh"))
        self.assertEqual(a.state_raw, "महाराष्ट्र")
        self.assertTrue(set(a.numbers) == set(b.numbers) == {"203", "19"})

    def test_features(self):
        f = extract_address_features("00123 YEAGER RD, COALLTON, WV", "US")
        self.assertTrue(f["addr_has_zero_padded_number"])
        self.assertTrue(f["addr_is_all_upper"])
        self.assertEqual(f["addr_state"], "wv")
        f = extract_address_features(None, "US")
        self.assertTrue(f["addr_is_missing"])


class TestTokenisation(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(word_tokens("  Ram  All-Consulting "), ["ram", "all-consulting"])
        self.assertEqual(alnum_tokens("Ram All-Consulting (L.L.P.)"), ["ram", "all", "consulting", "llp"])
        self.assertEqual(alnum_tokens("प्राइवेट लिमिटेड"), ["प्राइवेट", "लिमिटेड"])   # matras not split off
        self.assertEqual(numeric_tokens("00123 Rd, 5291b, C-045"), ["123", "5291", "45"])

    def test_char_ngrams(self):
        self.assertEqual(char_ngrams("ab", 3), ["#ab", "ab#"])
        self.assertEqual(char_ngrams("Tek", 3), ["#te", "tek", "ek#"])
        self.assertEqual(char_ngrams(None), [])
        self.assertTrue(all(len(g) == 3 for g in char_ngrams("ग्लोबल टेक", 3)))


class TestDoctests(unittest.TestCase):
    def test_module_doctests(self):
        result = doctest.testmod(N)
        self.assertEqual(result.failed, 0)


if __name__ == "__main__":
    unittest.main()
