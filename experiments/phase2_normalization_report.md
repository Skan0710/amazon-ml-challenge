# Phase 2 — Smart Normalisation report

*Amazon ML Challenge 2026 · business entity resolution (S1 → S2 ∪ S3)*

This report covers **training data only**; the test set was not read.

Artefacts:
* `src/normalization.py`: the module
* `tests/test_normalization.py`: 48 unit tests plus the module doctests
* `notebooks/02_normalization_validation.ipynb`: validation on real data
* `experiments/phase2_normalization_results.json`: every number quoted here

**Scope.** This phase delivers reusable normalisation utilities and validates them. Nothing is materialised at full scale, and there is no candidate generation, no model and no test-set prediction.

---

## 1. Implemented functions

| function | purpose |
|---|---|
| `normalize_basic_text(text)` | Conservative: NFKC, zero-width removal, quote/dash unification, casefold, dotted abbreviations collapsed (`L.L.C.` → `llc`), other non-decimal dots → space. Website tokens are kept intact. |
| `is_missing_value(v)` | `None`, NaN, blank, `null`, `<NULL>`, `N/A`, `na`, `none`, `nan`, `-` |
| `strip_latin_accents(text)` | Removes diacritics **only from Latin letters**; Indic vowel signs are untouched |
| `detect_script(text)` → `latin / indic / mixed / other / unknown` | Script family; `script_breakdown(text)` gives per-script letter counts (devanagari, bengali, gurmukhi, gujarati, oriya, tamil, telugu, kannada, malayalam, …) |
| `transliterate_text(text)` | Offline Brahmic → Latin romanisation (section 8) |
| `phonetic_key(text)` | Loose consonant-skeleton key for cross-script similarity |
| `normalize_name(raw)` → `NameRepr` | All name representations (section 3) |
| `normalize_address(raw, country)` → `AddressRepr` | All address representations and structured parts (section 9) |
| `normalize_state(value, country)` | Explicit state tables, country-aware, no fuzzy matching |
| `extract_name_features(...)`, `extract_address_features(...)` | Flat, model-ready descriptor dicts |
| `word_tokens`, `alnum_tokens`, `numeric_tokens`, `char_ngrams` | Tokenisation utilities (section 12) |
| `normalize_records(names, addresses, countries)`, `normalize_records_chunk(chunk)` | Batch API: compact per-record dicts for later Parquet materialisation and process pools |

Design properties:
* Deterministic and standard-library only.
* `raw` is always kept.
* Results are frozen dataclasses with `to_dict()`.
* `normalize_name` / `normalize_address` are `lru_cache`d, with up to 262k entries each per process. Phase 1 found 30% duplicate names, so caching pays off.
* Every function accepts `None`, NaN, empty strings, punctuation-only text and mixed scripts. The unit tests cover these cases.

## 2. Basic normalisation

`"  ACME   Pvt. Ltd. "` → `"acme pvt ltd"`.

| raw | basic |
|---|---|
| `Ram All Consulting L.L.P.` | `ram all consulting llp` |
| `Meredithe T. Igoe, MD, P.C.` | `meredithe t igoe, md, pc` |
| `ZIONTERM.COM` | `zionterm.com` (website kept) |
| `Version 2.5` | `version 2.5` (decimal kept) |
| `इंडियन इंटरनेशनल प्राइवेट लिमिटेड` | unchanged |

Basic normalisation never strips accents or punctuation other than dots; those steps belong to the higher layers.

## 3. Name normalisation: the representation ladder

| field | meaning | example: `*** Wenonah'S Metal Works` |
|---|---|---|
| `raw` | untouched | `*** Wenonah'S Metal Works` |
| `basic` | section 2 | `*** wenonah's metal works` |
| `punct` | letter/digit tokens of any script; apostrophes dropped; `&` → `and`; Latin accents stripped | `wenonahs metal works` |
| `canonical` | `punct` with legal forms replaced by canonical labels | (same; no legal form) |
| `core` | legal forms, leading `the` / `m/s`, junk and dangling `and` removed; **original script kept**; OCR digit fix (below) | `wenonahs metal works` |
| `transliterated` | `punct` in Latin script | `wenonahs metal works` |
| `core_latin` | `core` in Latin script: **the main cross-script key** | `wenonahs metal works` |
| `compact` | `core_latin` without spaces or `and` | `wenonahsmetalworks` |
| `phonetic` | `phonetic_key(core_latin)` | `pnhs mtl prks` |
| `legal_forms`, `alias_marker`, `aliases`, `website`, `website_label`, `script` | side information | |

**OCR digit fix.** A Latin word with ≥ 3 letters and exactly one digit from {0, 1, 5} gets that digit replaced:
* `5ervices` → `services`, `wi1cox` → `wilcox`, `Jar1yx 5mart` → `jarlyx smart`
* `1st`, `24th` and `b2b` are left alone.

## 4. Legal-form normalisation

This is an explicit, editable table (`LEGAL_FORM_VARIANTS`). Matching is greedy longest-first on Latin tokens. Native-script tokens are aligned one-to-one with their transliterations, so a decision made on the Latin side applies to the native token too.

| canonical | variants (after punctuation normalisation) | removal rule |
|---|---|---|
| private limited | private limited, pvt ltd, pvt limited, private ltd, p ltd, pvt ltd co, **pra li** (Indic `प्रा. लि.`) | anywhere |
| public limited | public limited, public ltd | anywhere |
| llc / llp / pllc / plc / inc / corp | llc · llp, **elelpi** (`एलएलपी`, `ఎల్‌ఎల్‌పీ`, `എൽഎൽപി` …) · pllc · plc · inc, incorporated · corp, corporation | anywhere |
| limited liability company / partnership | full phrases | anywhere |
| limited | ltd (anywhere); limited, **limtid** (Gurmukhi), **limitet** (Tamil), **limittad** (Malayalam) | `ltd` anywhere, others at either edge |
| private | pvt (anywhere); private, **praivet**, **praibhet** (Bengali/Oriya), **piraivet** (Tamil), **praivatt** (Malayalam) | edges |
| co / lp / pc / pa / opc | co, company · lp · pc · pa · opc | **end only** |

Conservative rules:
* **Edge-only forms.** Ambiguous words are only removed at the start or end of the name. `Co Op Bank` keeps `co`, and `Limited Sai Private Center` → `sai private center`.
* **Context-only abbreviations.** `li` and `pra` only count when they sit next to another legal form.
* **Never empty.** A core is never left empty: `Company` stays `company`.
* **Merging.** `private` + `limited` are merged into `private limited`, so `Private Wonderland Energy Ltd` and `Wonderland Energy Pvt. Ltd.` both give core `wonderland energy` with legal form `private limited`.

Where the Indic variants came from:
* They are the transliterations of the most frequent native legal words in S2/S3 **names**, taken from the source files, not from the ground truth.
* Example: `लिमिटेड` appears 296,635 times and `లిమిటెడ్` 51,485 times.

On the 250k-record sample (section 13), a legal form was detected in:

| | India | US |
|---|---:|---:|
| S1 | 84.2% | 57.5% |
| S2 | 75.4% | 49.3% |
| S3 | 72.6% | 48.6% |

The most frequent labels were `private limited`, `llc`, `inc` and `limited`.

## 5. Alias handling

The markers are an explicit list: `a/k/a`, `aka`, `f/k/a`, `fka`, `d/b/a`, `dba` (optionally followed by `:` or `-`), `t/a`, `formerly`, `formerly known as`, `also known as`, `doing business as` and `trading as`.

* A marker only counts when there is **non-empty text on both sides**. `Dba Brothers Pvt Ltd`, `Aka Holdings` and `T/A Post Inc.` are *not* aliases.
* Each side's `core_latin` is exposed in `aliases`, and `NameRepr.variants` returns core, aliases and website label.
* The full name is kept (marker removed in `core`).
* Example: `Quodova a/k/a Indian International Private Limited` → aliases `('quodova', 'indian international')`.
* Sample hit rates:
  * alias markers appear in 2.3% (India) and 3.5% (US) of S3 names;
  * they appear in 0% of S1 and S2 names, confirming the Phase 1 observation that this noise type belongs to S3.

## 6. Website-like names

A name is treated as a website only when the **whole name** (after leading junk) is one domain token with a known TLD: `com, net, org, in, co.in, co, biz, info, us, io, co.uk, …`.

| raw | website | website_label = core |
|---|---|---|
| `zionterm.com` | zionterm.com | `zionterm` |
| `... PEAKBNY.COM` | peakbny.com | `peakbny` |
| `www.acme-tools.co.in` | acme-tools.co.in | `acmetools` |
| `ílluminatifinvest.com` | illuminatifinvest.com | `illuminatifinvest` |
| `5ERVICESNAGESHWARWELFARE.COM` | … | `servicesnageshwarwelfare` |

* **Not websites:** `St. Louis Bakery`, `J.P. Morgan`, `Vinayaka.Plaza Traders` and `No.1 Tailors`.
* **Matching back to the business name:** the label is compared with the other name's `compact` form. `wenonahsmetalworks.com` ↔ `Wenonah's Metal Works`, and `premiersons.com` ↔ `Premier & Sons` (because `compact` drops `and`).
* **Sample rate:** websites are 2.9–4.0% of S2/S3 names and 0% of S1.

## 7. Script detection

`detect_script` returns a **family**: `latin`, `indic`, `mixed`, `other` or `unknown` (no letters).

* **Specific scripts:** `script_breakdown` names the actual script. All nine Indian scripts found in the data are covered (tested):
  * Devanagari, Bengali, Gurmukhi, Gujarati
  * Oriya, Tamil, Telugu, Kannada, Malayalam
* **Mixed names:** `North ఇంటర్నేషనల్ …` → `mixed`.
* **Sample rates:**

  | Indic-script names | S2 | S3 |
  |---|---:|---:|
  | India | 23.3% | 13.2% |
  | US | 0% | 0% |

  Mixed-script names are 0.9% (S2) and 1.7% (S3) of India names.

## 8. Transliteration

**Environment check.** No transliteration package was installed (no `unidecode`, `anyascii`, `indic-transliteration` or ICU), and no external service may be used. So a **built-in, table-driven transliterator** was written instead, and **no dependency was added**.

How it works:
* **One table for nine scripts.** All nine Brahmic Unicode blocks share the same layout: offset `0x15` is KA in every one. A single offset table therefore covers all nine scripts.
* **Script-specific overrides:**
  * nukta letters (`ज़` → z, `फ़` → f, `ਸ਼` → sh)
  * Bengali `ৎ`
  * Malayalam chillus, and `റ്റ` → tt, `ന്റ` → nt
  * Tamil `ஃப` → f, and Tamil `ச` → s in loanwords
  * Gurmukhi `ਐ` → e
  * Indic digits → 0–9
* **Schwa handling:** for Devanagari, Bengali, Gurmukhi, Gujarati and Oriya:
  * the word-final inherent vowel is dropped;
  * the inherent vowel before an independent vowel is dropped;
  * a simple medial rule (V C[a] C V) is applied.

  Dravidian scripts write their vowels explicitly, so these rules don't apply to them.
* **English-oriented output.** Long and short vowels are merged, and retroflex and dental consonants are merged. This fits the data, because the Indic names are phonetic spellings of English words.

Examples from the sample:

| script | raw | core_latin |
|---|---|---|
| Devanagari | `गोल्डन एक्सपोर्ट्स प्रा. लि.` | `goldan eksaports` (legal: private limited) |
| Bengali | `গুরু টেকনোলজি প্রাইভেট লিমিটেড` | `guru teknolji` |
| Gurmukhi | `ਅਰਿਹੰਤ ਇੰਡਸਟ੍ਰੀਜ਼ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ` | `arihant indasatriz` |
| Gujarati | `શિવા ઇન્ફ્રા પ્રાઇવેટ લિમિટેડ` | `shiva inphra` |
| Oriya | `ଟେକ୍ ଇନଭେଷ୍ଟମେଣ୍ଟ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍` | `tek inbheshtament` |
| Tamil | `அபெக்ஸ் பிரீமியர் எனர்ஜி பிரைவேட் லிமிடெட்` | `apeks pirimiyar enarji` |
| Telugu | `సూర్య లాజిస్టిక్స్ ఎల్‌ఎల్‌పీ` | `surya lajistiks` (legal: llp) |
| Kannada | `ಗ್ಯಾಲಕ್ಸಿ ಹೈ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `gyalaksi hai kanstrakshan` |
| Malayalam | `പയനിയർ ഇൻവെസ്റ്റ്മെന്റ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്` | `payaniyar investtments` |

Transliteration is **approximate**. Exact `core_latin` equality across scripts is rare: 3.3% of Indic-vs-Latin true pairs (section 14). For cross-script comparison, use **similarity on `core_latin`** or the **`phonetic` key**:

**`phonetic_key` rules:**
* Aspirates collapse (bh → b, th → t, …).
* c/q → k, z → j.
* v/w/b collapse to p, and voiced stops merge with unvoiced ones. The reasons: Bengali and Oriya write English "v" with ব/ଭ, and Tamil script has no voicing contrast (`trading` → `டிரேடிங்` "tireting").
* A leading vowel becomes `a`, later vowels are dropped, and repeats collapse.

**Before/after this rule change** (measured on the same sample):

| | Indic-vs-Latin true pairs with phonetic ≥ 80 | Script-mismatch control | All-pairs control mean |
|---|---:|---:|---:|
| before | 92.4% | 0.25% | 33.4 |
| after | 97.8% | 0.36–0.47% | 39.2 |

The key is therefore looser overall, so use it mainly for the cross-script subset (section 16).

## 9. Address normalisation

The pipeline:
1. `basic`
2. Split on commas; drop missing-value components (`null`, `N/A`, `<NULL>`, `na`, empty) and count them in `noise_removed`.
3. Remove `#` and `##` (`C-##45` → `c-45`, `Room-#1` → `room-1`).
4. Tokenise:
   * tokens **with digits** keep internal `-` and `/` (`c-71`, `12-1-331/c/1`, `3700-3702`) and lose leading zeros (`00123` → `123`);
   * pure words split on `-` and `/` (`Hastings-on-hudson` → `hastings on hudson`) and are canonicalised.

**Canonical word mappings:**

| type | mappings |
|---|---|
| street types (USPS) | street/str → st, avenue → ave, road → rd, drive → dr, lane → ln, court → ct, place → pl, boulevard → blvd, … |
| directions | north → n, east → e, … |
| ordinal words | ninth → 9th |
| unit words | apartment → apt, suite → ste, floor → fl, building → bldg |
| other | near → nr, opposite → opp |
| city suffix | trailing `cdp` dropped |

**Representations:**
* `basic`
* `components`: canonical comma components in **original order**; `component_set` gives the order-insensitive view.
* `normalized`: `", ".join(components)`
* `tokens`
* the structured parts in section 11

Example of order kept and order-insensitive comparison:
* `Point Pleasant, 515 Kitty Hawk Lane, WV` → `pt pleasant, 515 kitty hawk ln, wv`
* `00515 Kitty Hawk Ln, Point Pleasant, West Virginia` → `515 kitty hawk ln, pt pleasant, wv`

The two have the same `component_set` but different `components`.

## 10. State normalisation

These are explicit tables (`US_STATES`, `INDIA_STATES`) with no fuzzy matching.

* **US:** 50 states, DC and 5 territories, name ↔ code.
* **India:** every state and UT with English variants (`Orissa`/`Odisha`, `Keralam`, `Pondicherry`, `Uttaranchal`, …).
* **Alternative Indian codes:** `TS` → `tg`, `OR` → `od`, `CT` → `cg`, `UT` → `uk`.
* **Native spellings:** the 16 native-script state names that occur in S2/S3, e.g. `महाराष्ट्र`, `ಕರ್ನಾಟಕ`, `தமிழ்நாடு`, `পশ্চিমবঙ্গ`, `ଓଡ଼ିଶା`. They were read from the **source address columns**, not the ground truth.
* **Country-aware:** `IN` = Indiana for the US; `GA` = Georgia (US) or Goa (India). Without a country, anything that resolves in both tables is rejected (e.g. `AR`).
* **Where the state comes from:** the last comma component that is exactly a state. `state_raw` keeps the original.
* **Coverage:** on the sample, a state was resolved for **100% of non-missing addresses** in every source and country. That's 100% for S1, and 96.4–97.2% of all S2/S3 rows, which equals 100% minus the missing addresses.

## 11. Structured address extraction

| field | how |
|---|---|
| `house_number` / `house_number_core` | 1. Explicit marker (`h no`, `house no`, `plot no`, `flat no`, `door no`, `no`, glued `no3-1621`); PO-box numbers are excluded. 2. Otherwise the leading number-like token of a component. `core` = first digit group without zeros: `181B` → `181`, `c-71` → `71`, `3700-3702` → `3700`. |
| `street` | Component with a street-type word (the house component preferred), minus the house number; else the house component if it still has words |
| `city` | Last remaining digit-free component; trailing `city` dropped (`MESA CITY` → `mesa`) |
| `places` | **All** digit-free, non-state components (city, district, localities); order kept |
| `state`, `state_raw` | Section 10 |
| `unit` | `unit … / apt … / ste … / pmb …` components |
| `po_box` | `po box N`, `p o box N`, `post box N` |
| `numbers` | Every digit group, zeros stripped, unique, in order |
| `remaining_tokens` | Tokens not used by the fields above |

Coverage on the 250k sample (% of all rows):

| | S1 India | S1 US | S2 India | S2 US | S3 India | S3 US |
|---|---:|---:|---:|---:|---:|---:|
| state | 100.0 | 100.0 | 97.2 | 96.4 | 97.0 | 96.5 |
| house number | 79.5 | 99.9 | 78.5 | 90.1 | 76.0 | 90.7 |
| city | 99.9 | 99.4 | 96.1 | 95.9 | 96.6 | 95.9 |
| street | 58.4 | 96.9 | 56.0 | 93.2 | 50.1 | 93.1 |
| noise removed (≥ 1 artefact) | 0.04 | 0.0 | 5.9 | 6.4 | 5.3 | 5.9 |

Street coverage is low for India because most Indian addresses have no street-type word (`Friends Colony`, `Sector-14`). Those components end up in `places`.

## 12. Tokenisation utilities

| utility | behaviour |
|---|---|
| `word_tokens` | Whitespace tokens of the basic text (punctuation kept) |
| `alnum_tokens` | Letter/digit tokens of **any script**. The token class includes combining marks, so Indic vowel signs are not split off; Python's `\w` would split them. |
| `numeric_tokens` | Digit groups without leading zeros |
| `char_ngrams(text, n=3, pad=True)` | Per-token character n-grams with `#` boundaries, any script. Returns a list, so callers can build sets or TF-IDF. |

No index is built; this is utilities only.

## 13. Before/after examples

`notebooks/02_normalization_validation.ipynb` §3 shows the full tables. The sample is `hash(entity_id) % 50 = 0`, which gives 250,617 records. It includes US names, Indian Latin names, Indic names (two per script), messy names, websites, legal-form variants, aliases, and US/Indian/messy/missing addresses.

Selected rows:

| raw | core_latin | legal / alias / web |
|---|---|---|
| `Banks Lane (L.L.C.)` | `banks lane` | llc |
| `Pvt Uttam Mánagement Ltd` | `uttam management` | private limited |
| `Regional College P.L.L.C.` | `regional college` | pllc |
| `Fluxriza DBA: Supreme Kpet, Corp` | `fluxriza supreme kpet` | corp · dba → (`fluxriza`, `supreme kpet`) |
| `Onyxveo d/b/a Barreras, Madeline, D.O.` | `onyxveo barreras madeline do` | d/b/a |
| `AAAINDIA.COM` | `aaaindia` | website |

| raw address | normalized | hn / city / state |
|---|---|---|
| `002036 MARY ELLA DRIVE, NULL, GEORGETOWN, IN` | `2036 mary ella dr, georgetown, in` | 2036 / georgetown / in |
| `80 Main St, # Unit 5, Trappe Borough, Pennsylvania` | `80 main st, unit 5, trappe borough, pa` | 80 / trappe borough / pa (unit 5) |
| `###114-115, Kamrej, Surat, ગુજરાત` | `114-115, kamrej, surat, gj` | 114-115 / surat / gj |
| `NO - 13 PADMAVAHI COLONY, BALAJI HILLS, UPPAL, తెలంగాణ` | `no 13 padmavahi colony, balaji hills, uppal, tg` | 13 / uppal / tg |
| `HOUSE NO. 662, SECTOR-10, PANCHKULA, हरियाणा` | `h no 662, sector-10, panchkula, hr` | 662 / panchkula / hr |

## 14. True-pair diagnostics (ground truth used for evaluation only)

**Setup:**
* **True pairs:** 50,934, chosen by `hash(s1_id, match_id) % 150 = 7`. That's 24,706 S2 and 26,228 S3; 30,480 US and 20,454 India.
* **Control:** the same targets re-paired at random with S1 records of the same country.
* **Noise:** control figures move by about ±0.05 points between runs, because the join order changes the random permutation.

**Agreement rates:** boolean metrics are the % of pairs that agree. For house number, state, city, place, street and number, the % is over pairs where both sides have the field.

| metric | true: all | S2 | S3 | US | India | random control |
|---|---:|---:|---:|---:|---:|---:|
| name raw equal | 4.71 | 4.97 | 4.46 | 6.08 | 2.66 | 0.00 |
| name basic equal | 17.52 | 17.83 | 17.23 | 21.33 | 11.84 | 0.00 |
| name punct equal | 26.68 | 26.29 | 27.06 | 31.98 | 18.79 | 0.00 |
| **name core equal** | 52.15 | 53.08 | 51.27 | 56.70 | 45.36 | 0.01 |
| **name core_latin equal** | **52.31** | 53.25 | 51.42 | 56.70 | 45.76 | 0.01 |
| name variant equal (core / alias / website ↔ compact) | **58.30** | 57.12 | 59.41 | 63.18 | 51.03 | 0.01 |
| ≥ 1 shared raw name token | 85.83 | 84.05 | 87.51 | 91.99 | 76.65 | 15.65 |
| ≥ 1 shared core_latin token | 87.96 | 86.97 | 88.90 | 91.82 | 82.21 | **1.53** |
| legal forms equal | 71.38 | 70.71 | 72.01 | 72.75 | 69.35 | 32.30 |
| address basic equal | 7.42 | 10.72 | 4.31 | 7.99 | 6.58 | 0.00 |
| **address normalized equal** | **25.91** | 23.93 | 27.77 | 31.63 | 17.38 | 0.00 |
| ≥ 1 shared normalized component | 95.45 | 95.31 | 95.58 | 95.09 | 95.98 | 6.56 |
| **house_number_core agrees** | **86.78** | 86.21 | 87.32 | 89.46 | 81.98 | 0.49 |
| house_number (full token) agrees | 80.78 | 80.27 | 81.27 | 83.01 | 76.79 | 0.15 |
| state agrees | 99.38 | 99.38 | 99.38 | 99.86 | 98.67 | 6.79 |
| city agrees | 74.86 | 71.26 | 78.24 | 82.69 | 63.29 | 0.80 |
| ≥ 1 shared place | 89.27 | 88.98 | 89.54 | 82.75 | **98.91** | 1.26 |
| street agrees | 90.35 | 90.54 | 90.17 | 91.17 | 88.00 | 0.02 |
| ≥ 1 shared number | 94.46 | 93.95 | 94.94 | 92.62 | 97.23 | 3.87 |

**Mean similarity scores** (RapidFuzz `token_set_ratio`, 0–100; on true pairs where both addresses are present):

| metric | true: all | S2 | S3 | US | India | random control |
|---|---:|---:|---:|---:|---:|---:|
| name, raw | 84.8 | 83.2 | 86.3 | 90.9 | 75.7 | 32.5 |
| name, core_latin | 91.9 | 91.6 | 92.1 | 93.9 | 88.8 | 30.8 |
| name, phonetic | 93.9 | 94.2 | 93.7 | 94.3 | 93.4 | 39.2 |
| address, raw | 89.0 | 91.9 | 86.3 | 88.2 | 90.2 | 36.2 |
| address, normalized | 95.1 | 95.2 | 95.0 | 95.0 | 95.3 | 36.5 |

**Compared with Phase 1 baselines:**

| | Phase 1 | Phase 2 |
|---|---:|---:|
| exact normalised-name agreement | 15.7% (basic), 25.8% (alnum) | 52.3% (`core_latin`) |
| address agreement | 7.7% | 25.9% (`normalized`) |

Random-pair agreement stays at ≈ 0 for all of these.

**Indic-vs-Latin subset** (3,629 true pairs where exactly one side is Indic or mixed script):

| metric | true | control |
|---|---:|---:|
| ≥ 1 shared raw token | 7.96 | 2.09 |
| ≥ 1 shared core_latin token | 40.04 | 0.36 |
| core_latin equal | 3.33 | 0.00 |
| token_set raw (mean) | 14.1 | 10.3 |
| token_set core_latin (mean) | 70.5 | 30.7 |
| phonetic token_set (mean) | 94.3 | 39.5 |
| % with raw ≥ 80 / core_latin ≥ 80 / phonetic ≥ 80 | 2.31 / 30.37 / **97.80** | 0.00 / 0.11 / 0.36 |

Transliteration plus the phonetic key turns a subset that was almost unmatchable, with a raw mean of 14, into one where 97.8% of true pairs score ≥ 80, against 0.36% of random pairs.

**Key-agreement union.** For each combination of simple conditions, this is the % of pairs where at least one condition holds. It is an **upper bound on the recall** of exact keys like these. Candidate *volume* was **not** measured.

| condition | true pairs | random pairs |
|---|---:|---:|
| K1 name variant equal | 58.30 | 0.01 |
| K2 state + house_number_core equal | 69.80 | 0.05 |
| K1 ∪ K2 | 86.68 | 0.05 |
| K1 ∪ K2 ∪ K5 (shared place + shared number) | 92.40 | 0.16 |
| K1 ∪ K2 ∪ K4 (phonetic token_set ≥ 80) | 97.56 | 0.25 |
| **K1 ∪ K2 ∪ K4 ∪ K5** | **98.64** | 0.36 |
| K3 shared core_latin token | 87.96 | 1.53 |
| K3 ∪ K2 | 96.11 | 1.57 |

**What still fails:**
* Random replacement names with a truncated address (`Urban Global Private Limited` ↔ `Koraria`).
* Changed house numbers with a different name (`841` ↔ `341 School House Road`).
* Descriptive word swaps with the house number dropped (`Comer Supreme Interprivate` ↔ `Comer Supreme Service`, `Kay St` without a number).
* Short Latin acronyms written in Indic script (`Ss Energy` ↔ `एसएस एनर्जी`).

## 15. Runtime benchmarks

These were measured on the 250,617-record sample on this machine (8-core Apple Silicon, 8 GB RAM), single process unless noted, starting from a cold cache:

| step | seconds | µs / record |
|---|---:|---:|
| `normalize_name` (cold) | 9.02 | 36.0 |
| `normalize_name` (warm cache, same records) | 0.06 | 0.2 |
| `normalize_address` (cold) | 12.73 | 50.8 |
| `transliterate_text` (14,957 Indic names, cold) | 0.16 | 10.9 |
| `extract_name_features` | 5.09 | 20.3 |
| `extract_address_features` | 0.74 | 2.9 |
| name + address via `normalize_records`, **8 processes** (incl. pool start-up and pickling results back) | 6.63 | 26.4 |

**Extrapolating to all 12.5M records** (name and address): about **18 min** single process, or about **5.5 min** with 8 processes. This is an extrapolation from the measured rates; the full run was **not** executed.

**Memory:** caches reach about 242k entries each for this sample. RSS was not reliably measurable, because garbage collection ran during the benchmark. The validation notebook's total runtime is about 84 s.

## 16. Known limitations

* **Transliteration is approximate** and tuned to English loanwords.
  * Native Hindi words with medial schwa may be off (`नई` → `ni`).
  * Bengali `ব` → `b` (`shib` for Shiv).
  * Exact cross-script equality stays rare (3.3%); rely on phonetic or fuzzy similarity.
* **The phonetic key is lossy.** `united` → `ant`. The v2 merges raise random-pair similarity (all-pairs control mean 33 → 39). Use it mainly where scripts differ, together with other evidence.
* **Legal-form edge removal** can drop genuine words, e.g. a business literally called `Limited …`. Only the listed variants and the observed Indic spellings are covered.
* **OCR folding** only handles 0/1/5 inside words. Other substitutions (`8iva`) remain.
* **Address parsing is heuristic:**
  * `city` = last digit-free component: 63% agreement for India. Use `places` instead (98.9%).
  * Indian `street` is often empty.
  * `house_number_core` takes the first digit group of compound Indian numbers (`12-1-331/c/1` → `12`).
  * Zero-stripping merges zero-padded prefixes such as `001414 79 …`.
  * `na` as a whole component is treated as missing.
* **States:** exact matches only. Native spellings cover the 16 observed states. Codes that are ambiguous between countries require the country.
* **Fully random replacement names** (`Ciraaria`) cannot be fixed by normalisation; they need address evidence.
* **Scope of validation:** everything was validated on samples only, about 250k records and about 51k pairs. Full-scale materialisation and candidate volumes were not run.

## 17. Recommended candidate-generation strategy (for Phase 3; not implemented)

1. **Materialise once.** Run `normalize_records` over all 12.5M records with a process pool (about 6 min estimated) and write Parquet to `data/processed/`, keyed by `entity_id`. Keep raw values in the source files.
2. **Use a union of complementary blockers, all partitioned by country:**
   * **K1:** exact `core_latin`, alias sides and `website_label` ↔ `compact`. 58% of true pairs agree; random pairs ≈ 0.
   * **K2:** `(state, house_number_core)`. 70% agree; random 0.05%.
   * **K4:** phonetic-key tokens or n-grams for Indic/mixed-script names. This closes most of the Indic-vs-Latin gap.
   * **K5:** `(state, shared place, shared number)`, mainly for India.
   * On the sample, at least one of K1/K2/K4/K5 agrees for **98.6%** of true pairs, against 0.36% of random pairs. This is an agreement upper bound; the blockers' actual candidate volume still has to be measured.
3. **Cap heavy keys.** Phase 1 showed name keys with up to 918 candidates per S1 record (`eye group`).
   * Measure key frequencies with DuckDB aggregations before joining.
   * For frequent keys, require `state` as well.
   * For shared-token blocking, use rare tokens only (IDF cap). K3 alone agrees for 88.0% of true pairs but for 1.5% of random pairs, which is far too unselective at 5M-row scale without such a cap.
4. **Fuzzy candidates:** use character 3-grams (`char_ngrams`) on `core_latin` and on the address for typo tolerance, within state or country, with top-k retrieval. This covers the typo-driven misses that exact keys can't reach.
5. **Evaluate every blocker** with the Phase 1 evaluator: recall on the full ground truth, candidates per S1 record (mean, p99, max) and total candidates. Target recall ≥ 97–99% at a manageable volume.
