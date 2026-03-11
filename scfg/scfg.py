from __future__ import annotations

import random
import re
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from scfg.agreement import (
    AgreementConfig,
    FeatureBundle,
    FeatureUnifier,
    build_suffix_inventory,
    deserialize_bundles,
    feature_inventory,
    serialize_bundles,
)

Rule = tuple[tuple[str, ...], tuple[str, ...]]

LATIN_CONSONANTS: list[str] = list("bcdfghjklmnpqrstvwxyz")
LATIN_VOWELS: list[str] = list("aeiou")
LATIN_SONORITY_HIERARCHY: dict[str, float] = {
    "l": 0.15,
    "m": 0.3,
    "n": 0.3,
    "v": 0.45,
    "z": 0.45,
    "f": 0.6,
    "s": 0.6,
    "b": 0.75,
    "d": 0.75,
    "g": 0.75,
    "p": 0.9,
    "t": 0.9,
    "k": 0.9,
}

CYRILLIC_CONSONANTS: list[str] = list("бвгджзйклмнпрстфхцчшщ")
CYRILLIC_VOWELS: list[str] = list("аеёиоуыэюя")
CYRILLIC_SONORITY_HIERARCHY: dict[str, float] = {
    "п": 0.1,
    "б": 0.15,
    "т": 0.1,
    "д": 0.15,
    "к": 0.1,
    "г": 0.15,
    "ф": 0.2,
    "в": 0.25,
    "с": 0.2,
    "з": 0.25,
    "ш": 0.2,
    "ж": 0.25,
    "х": 0.2,
    "ц": 0.2,
    "ч": 0.3,
    "щ": 0.3,
    "м": 0.4,
    "н": 0.4,
    "р": 0.5,
    "л": 0.5,
    "й": 0.6,
}

YIDDISH_CONSONANTS: list[str] = list("בגדהװזשחטיּכּכלמנפּפֿצקרשת")
YIDDISH_VOWELS: list[str] = list("אַאָוּיִײײַױע")
YIDDISH_SONORITY_HIERARCHY: dict[str, float] = {
    "פּ": 0.10,
    "ב": 0.15,
    "ט": 0.10,
    "ד": 0.15,
    "כּ": 0.10,
    "ג": 0.15,
    "ק": 0.10,
    "פֿ": 0.20,
    "װ": 0.25,
    "ס": 0.20,
    "ז": 0.25,
    "ש": 0.20,
    "כ": 0.20,
    "ח": 0.20,
    "ה": 0.20,
    "צ": 0.30,
    "מ": 0.40,
    "נ": 0.40,
    "ל": 0.50,
    "ר": 0.50,
    "י": 0.60,
}

SYLLABLE_STRUCTURES: list[str] = [
    "CVC",
    "CV*C",
    "CVC?",
    "C*VC",
    "CVC*",
    "C*VC*",
    "C*V*C*",
    "C?VC",
    "C?VC*",
    "CV*C*",
    "CV",
    "C?V",
    "CV*",
    "C*V",
]


@dataclass
class CFGParams:
    """
    Parameterization of an XBar-style context-free grammar.
    """

    head_initial: bool = True
    spec_initial: bool = True
    pro_drop: bool = False
    proper_with_det: bool = False
    syllable_structure: str | None = None
    avg_syllables_per_word: float = 2
    max_consonants: int = 2
    rng_seed: int = 42

    verbs: list[str] | int = 3
    nouns: list[str] | int = 3
    propns: list[str] | int = 3
    prons: list[str] | int = 2
    adjs: list[str] | int = 2
    det_def: list[str] | int = 2
    det_indef: list[str] | int = 2
    comps: list[str] | int = 2
    tenses: list[str] = field(default_factory=lambda: ["∅_T_pres"])
    asps: list[str] = field(default_factory=lambda: ["∅_Asp_prog"])
    orthography: str = "latin"

    agreement_enabled: bool = False
    agreement_axes: tuple[str, ...] = ("number", "person")
    agreement_strategy: str = "suffix"
    verb_agreement: bool = True
    noun_number_marking: bool = True
    pronouns_are_featured: bool = True
    latent_gender: bool = False
    realize_gender: bool = False
    gender_values: tuple[str, ...] = ("masc", "fem")

    verb_paradigms: list[dict[str, Any]] | None = None
    noun_paradigms: list[dict[str, Any]] | None = None
    propn_paradigms: list[dict[str, Any]] | None = None
    pronoun_paradigms: list[dict[str, Any]] | None = None
    agreement_suffixes: dict[str, str] | None = None

    space_alpha: float = 0.5
    space_beta: float = 3.0
    syllable_max: int = 4

    def to_dict(self) -> dict[str, Any]:
        param_dict = asdict(self)
        param_dict["verbs"] = self.verb_lemmas
        param_dict["nouns"] = self.noun_lemmas
        param_dict["propns"] = self.propn_lex
        param_dict["prons"] = self.pron_lex
        param_dict["adjs"] = self.adj_lex
        param_dict["det_def"] = self.det_def_lex
        param_dict["det_indef"] = self.det_indef_lex
        param_dict["comps"] = self.comp_lex
        param_dict["tenses"] = self.tense_lex
        param_dict["asps"] = self.asp_lex
        param_dict["verb_paradigms"] = serialize_bundles(self.verb_paradigms)
        param_dict["noun_paradigms"] = serialize_bundles(self.noun_paradigms)
        param_dict["propn_paradigms"] = serialize_bundles(self.propn_paradigms)
        param_dict["pronoun_paradigms"] = serialize_bundles(self.pronoun_paradigms)
        param_dict["agreement_suffixes"] = dict(self.agreement_suffixes)
        return param_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CFGParams":
        clean = dict(data)
        if "agreement_axes" in clean:
            clean["agreement_axes"] = tuple(clean["agreement_axes"])
        clean["verb_paradigms"] = clean.get("verb_paradigms")
        clean["noun_paradigms"] = clean.get("noun_paradigms")
        clean["propn_paradigms"] = clean.get("propn_paradigms")
        clean["pronoun_paradigms"] = clean.get("pronoun_paradigms")
        if "gender_values" in clean:
            clean["gender_values"] = tuple(clean["gender_values"])
        return cls(**clean)

    @property
    def agreement_config(self) -> AgreementConfig:
        return AgreementConfig(
            enabled=self.agreement_enabled,
            active_axes=self.agreement_axes,
            strategy=self.agreement_strategy,
            verb_agreement=self.verb_agreement,
            noun_number_marking=self.noun_number_marking,
            pronouns_are_featured=self.pronouns_are_featured,
            latent_gender=self.latent_gender,
            realize_gender=self.realize_gender,
            gender_values=self.gender_values,
        )

    @property
    def latent_axes(self) -> tuple[str, ...]:
        axes = list(self.agreement_axes)
        if self.latent_gender and "gender" not in axes:
            axes.append("gender")
        return tuple(axes)

    @property
    def surface_axes(self) -> tuple[str, ...]:
        axes = list(self.agreement_axes)
        if self.realize_gender and self.latent_gender and "gender" not in axes:
            axes.append("gender")
        return tuple(axes)

    @property
    def n_verbs(self) -> int:
        return len(self.verb_lex)

    @property
    def n_nouns(self) -> int:
        return len(self.noun_lex)

    @property
    def n_propns(self) -> int:
        return len(self.propn_lex)

    @property
    def n_adjs(self) -> int:
        return len(self.adj_lex)

    @property
    def n_det_defs(self) -> int:
        return len(self.det_def_lex)

    @property
    def n_det_indefs(self) -> int:
        return len(self.det_indef_lex)

    @property
    def n_comps(self) -> int:
        return len(self.comp_lex)

    @property
    def n_tense_lex(self) -> int:
        return len(self.tense_lex)

    @property
    def n_asp_lex(self) -> int:
        return len(self.asp_lex)

    def __post_init__(self):
        def _resolve(val: list[str] | int) -> list[str]:
            if isinstance(val, int):
                return [self._sample_string() for _ in range(val)]
            return list(val)

        self.rng = random.Random(self.rng_seed)

        if self.orthography == "latin":
            self.consonants = LATIN_CONSONANTS
            self.vowels = LATIN_VOWELS
            self.sonority_hierarchy = LATIN_SONORITY_HIERARCHY
        elif self.orthography == "cyrillic":
            self.consonants = CYRILLIC_CONSONANTS
            self.vowels = CYRILLIC_VOWELS
            self.sonority_hierarchy = CYRILLIC_SONORITY_HIERARCHY
        elif self.orthography == "yiddish":
            self.consonants = YIDDISH_CONSONANTS
            self.vowels = YIDDISH_VOWELS
            self.sonority_hierarchy = YIDDISH_SONORITY_HIERARCHY
        else:
            raise ValueError(f"Unknown orthography: {self.orthography}")

        if self.syllable_structure is None:
            self.syllable_structure = self.rng.choice(SYLLABLE_STRUCTURES)

        self.verb_lemmas = _resolve(self.verbs)
        self.noun_lemmas = _resolve(self.nouns)
        self.propn_lex = _resolve(self.propns)
        self.pron_lex = _resolve(self.prons)
        self.adj_lex = _resolve(self.adjs)
        self.det_def_lex = _resolve(self.det_def)
        self.det_indef_lex = _resolve(self.det_indef)
        self.comp_lex = _resolve(self.comps)
        self.tense_lex = _resolve(self.tenses)
        self.asp_lex = _resolve(self.asps)

        self.verb_paradigms = deserialize_bundles(self.verb_paradigms)
        self.noun_paradigms = deserialize_bundles(self.noun_paradigms)
        self.propn_paradigms = deserialize_bundles(self.propn_paradigms)
        self.pronoun_paradigms = deserialize_bundles(self.pronoun_paradigms)

        if self.agreement_enabled:
            if not self.agreement_suffixes:
                self.agreement_suffixes = self._build_agreement_suffixes()
            else:
                self.agreement_suffixes = dict(self.agreement_suffixes)
            if not self.pronoun_paradigms:
                self.pronoun_paradigms = self._build_pronoun_paradigms()
            if not self.noun_paradigms:
                self.noun_paradigms = self._build_noun_paradigms()
            if not self.propn_paradigms:
                self.propn_paradigms = self._build_propn_paradigms()
            if not self.verb_paradigms:
                self.verb_paradigms = self._build_verb_paradigms()
        else:
            self.agreement_suffixes = dict(self.agreement_suffixes or {})
            self.pronoun_paradigms = self.pronoun_paradigms or []
            self.noun_paradigms = self.noun_paradigms or []
            self.propn_paradigms = self.propn_paradigms or []
            self.verb_paradigms = self.verb_paradigms or []

        self.noun_lex = self._flatten_noun_lexicon()
        self.verb_lex = self._flatten_verb_lexicon()
        self.pron_lex = self._flatten_pronoun_lexicon()

    def _build_agreement_suffixes(self) -> dict[str, str]:
        seen: set[str] = set()

        def _sample_unique_affix() -> str:
            while True:
                affix = self._sample_morpheme()
                if affix not in seen:
                    seen.add(affix)
                    return affix

        forms = [
            _sample_unique_affix()
            for _ in feature_inventory(
                self.surface_axes,
                gender_values=self.gender_values,
            )
        ]
        gender_suffixes: dict[str, str] = {}
        if self.realize_gender and self.latent_gender:
            for gender in self.gender_values:
                gender_suffixes[f"gender={gender}"] = _sample_unique_affix()
        return {
            "noun_plural": _sample_unique_affix(),
            **build_suffix_inventory(
                forms,
                self.surface_axes,
                gender_values=self.gender_values,
            ),
            **gender_suffixes,
        }

    def _parse_syllable_format(self, template: str) -> list[str]:
        return re.findall(r"C\*|V\*|C\?|V\?|C|V", template)

    def _generate_cluster(self, size: int) -> str:
        chars: list[str] = list(self.sonority_hierarchy.keys())
        weights: list[float] = [self.sonority_hierarchy[c] for c in chars]
        cluster = ""
        for _ in range(size):
            cluster += self.rng.choices(chars, weights=weights, k=1)[0]
        return cluster

    def _generate_syllable(self, template: str | list[str]) -> str:
        tokens = (
            self._parse_syllable_format(template)
            if isinstance(template, str)
            else template
        )
        result: list[str] = []
        for token in tokens:
            if token == "C":
                result.append(self.rng.choice(self.consonants))
            elif token == "V":
                result.append(self.rng.choice(self.vowels))
            elif token == "C*":
                result.extend(self._generate_cluster(self.rng.randint(0, self.max_consonants)))
            elif token == "V*":
                result.extend(self.rng.choices(self.vowels, k=self.rng.randint(1, 2)))
            elif token == "C?":
                if self.rng.random() < 0.5:
                    result.append(self.rng.choice(self.consonants))
            elif token == "V?":
                if self.rng.random() < 0.5:
                    result.append(self.rng.choice(self.vowels))
        return "".join(result)

    def _sample_string(self) -> str:
        def _zero_truncated_poisson(rate: float) -> int:
            u: float = np.random.uniform(np.exp(-rate), 1)
            t: float = -np.log(u)
            return 1 + np.random.poisson(rate - t)

        def _beta_binomial(n: int, alpha: float, beta: float) -> int:
            p: float = np.random.beta(alpha, beta)
            return np.random.binomial(n, p)

        def _interleave_spaces(syllables: list[str], n_spaces: int) -> str:
            if n_spaces <= 0:
                return "".join(syllables)
            positions: list[int] = list(range(1, len(syllables)))
            chosen_positions: set[int] = set(
                random.sample(positions, min(n_spaces, len(positions)))
            )
            result_parts: list[str] = []
            for i, syllable in enumerate(syllables):
                result_parts.append(syllable)
                if i + 1 in chosen_positions:
                    result_parts.append(" ")
            return "".join(result_parts)

        syllables: list[str] = []
        num_syllables: int = min(
            _zero_truncated_poisson(self.avg_syllables_per_word),
            self.syllable_max,
        )
        for _ in range(num_syllables + 1):
            syllables.append(self._generate_syllable(self.syllable_structure))
        n_spaces: int = _beta_binomial(num_syllables - 1, self.space_alpha, self.space_beta)
        return _interleave_spaces(syllables, n_spaces)

    def _sample_morpheme(self) -> str:
        tokens = self._parse_syllable_format(self.syllable_structure)
        morpheme = self._generate_syllable(tokens)
        if not morpheme:
            morpheme = self._generate_syllable(["C", "V"])
        return morpheme

    def _build_pronoun_paradigms(self) -> list[dict[str, Any]]:
        pronoun_forms = list(self.pron_lex)
        bundles = feature_inventory(self.agreement_axes)
        while len(pronoun_forms) < len(bundles):
            pronoun_forms.append(self._sample_string())
        paradigms: list[dict[str, Any]] = []
        for bundle, form in zip(bundles, pronoun_forms):
            paradigms.append({"features": bundle, "form": form})
        return paradigms

    def _build_noun_paradigms(self) -> list[dict[str, Any]]:
        paradigms: list[dict[str, Any]] = []
        plural_suffix = self.agreement_suffixes.get("noun_plural", "")
        for lemma in self.noun_lemmas:
            gender = (
                self.rng.choice(self.gender_values)
                if self.latent_gender
                else None
            )
            forms = {
                "number=sg": lemma,
                "number=pl": lemma if not self.noun_number_marking else f"{lemma}{plural_suffix}",
            }
            paradigms.append(
                {
                    "lemma": lemma,
                    "forms": forms,
                    "features": FeatureBundle(person="3", gender=gender),
                }
            )
        return paradigms

    def _build_propn_paradigms(self) -> list[dict[str, Any]]:
        paradigms: list[dict[str, Any]] = []
        for lemma in self.propn_lex:
            gender = (
                self.rng.choice(self.gender_values)
                if self.latent_gender
                else None
            )
            paradigms.append(
                {
                    "lemma": lemma,
                    "features": FeatureBundle(person="3", number="sg", gender=gender),
                }
            )
        return paradigms

    def _build_verb_paradigms(self) -> list[dict[str, Any]]:
        paradigms: list[dict[str, Any]] = []
        bundles = feature_inventory(
            self.latent_axes,
            gender_values=self.gender_values,
        )
        for lemma in self.verb_lemmas:
            forms: dict[str, str] = {}
            for bundle in bundles:
                latent_key = bundle.key(self.latent_axes)
                surface_key = bundle.key(self.surface_axes)
                suffix = self.agreement_suffixes.get(surface_key, "") if self.verb_agreement else ""
                form = f"{lemma}{suffix}"
                if self.realize_gender and bundle.gender is not None:
                    form = f"{form}{self.agreement_suffixes.get(f'gender={bundle.gender}', '')}"
                forms[latent_key] = form
            paradigms.append({"lemma": lemma, "forms": forms})
        return paradigms

    def _flatten_noun_lexicon(self) -> list[str]:
        if not self.agreement_enabled or not self.noun_paradigms:
            return list(self.noun_lemmas)
        forms: list[str] = []
        for paradigm in self.noun_paradigms:
            forms.extend(paradigm["forms"].values())
        return list(dict.fromkeys(forms))

    def _flatten_verb_lexicon(self) -> list[str]:
        if not self.agreement_enabled or not self.verb_paradigms:
            return list(self.verb_lemmas)
        forms: list[str] = []
        for paradigm in self.verb_paradigms:
            forms.extend(paradigm["forms"].values())
        return list(dict.fromkeys(forms))

    def _flatten_pronoun_lexicon(self) -> list[str]:
        if not self.agreement_enabled or not self.pronoun_paradigms:
            return list(self.pron_lex)
        return [entry["form"] for entry in self.pronoun_paradigms]

    def choose_pronoun(self, rng: random.Random) -> tuple[str, FeatureBundle]:
        if not self.agreement_enabled or not self.pronouns_are_featured:
            surface = rng.choice(self.pron_lex)
            return surface, FeatureBundle()
        entry = rng.choice(self.pronoun_paradigms)
        return entry["form"], entry["features"]

    def choose_noun(self, rng: random.Random, number: str | None = None) -> tuple[str, FeatureBundle]:
        if not self.agreement_enabled or not self.noun_paradigms:
            surface = rng.choice(self.noun_lex)
            return surface, FeatureBundle(person="3")
        paradigm = rng.choice(self.noun_paradigms)
        chosen_number = number or rng.choice(("sg", "pl"))
        surface = paradigm["forms"][f"number={chosen_number}"]
        base_features = paradigm.get("features", FeatureBundle(person="3"))
        return surface, FeatureBundle(
            person=base_features.person or "3",
            number=chosen_number,
            gender=base_features.gender,
        )

    def choose_propn(self, rng: random.Random) -> tuple[str, FeatureBundle]:
        if not self.agreement_enabled or not self.propn_paradigms:
            return rng.choice(self.propn_lex), FeatureBundle(person="3", number="sg")
        entry = rng.choice(self.propn_paradigms)
        features = entry.get("features", FeatureBundle(person="3", number="sg"))
        return entry["lemma"], FeatureBundle(
            person=features.person or "3",
            number=features.number or "sg",
            gender=features.gender,
        )

    def choose_verb(
        self,
        rng: random.Random,
        features: FeatureBundle | None = None,
    ) -> tuple[str, FeatureBundle]:
        if not self.agreement_enabled or not self.verb_paradigms:
            return rng.choice(self.verb_lex), features or FeatureBundle()
        paradigm = rng.choice(self.verb_paradigms)
        target = FeatureUnifier.unify(features or FeatureBundle(), FeatureBundle()) or FeatureBundle()
        resolved_gender = target.gender
        if "gender" in self.latent_axes and resolved_gender is None:
            resolved_gender = self.gender_values[0]
        bundle_key = FeatureBundle(
            person=target.person or "3",
            number=target.number or "sg",
            gender=resolved_gender,
        ).key(self.latent_axes)
        return paradigm["forms"][bundle_key], FeatureBundle(
            person=target.person or "3",
            number=target.number or "sg",
            gender=resolved_gender,
        )

    def choose_det(self, rng: random.Random, definite: bool) -> str:
        return rng.choice(self.det_def_lex if definite else self.det_indef_lex)

    def choose_adj(self, rng: random.Random) -> str:
        return rng.choice(self.adj_lex)

    def choose_comp(self, rng: random.Random) -> str:
        return rng.choice(self.comp_lex)

    def choose_t(self, rng: random.Random) -> str:
        return rng.choice(self.tense_lex)

    @classmethod
    def english(cls) -> "CFGParams":
        return cls(
            head_initial=True,
            spec_initial=True,
            pro_drop=False,
            proper_with_det=False,
            verbs=["eat", "see", "love", "hear"],
            nouns=["tree", "horse", "dog", "cat", "apple"],
            propns=["john", "mary", "sue", "bob"],
            prons=["i", "you", "he", "we", "youall", "they"],
            adjs=["big", "small", "red", "green", "blue", "fuzzy", "round"],
            det_def=["the"],
            det_indef=["a"],
            comps=["that"],
        )

    @classmethod
    def german(cls) -> "CFGParams":
        return cls(
            head_initial=True,
            spec_initial=True,
            pro_drop=False,
            proper_with_det=False,
            verbs=["ess", "seh", "lieb", "hor"],
            nouns=["baum", "pferd", "hund", "katze", "apfel"],
            propns=["john", "maria", "sue", "bob"],
            prons=["ich", "du", "er", "wir", "ihr", "sie"],
            adjs=["gross", "klein", "rot", "grun", "blau", "unscharf", "rund"],
            det_def=["der"],
            det_indef=["ein"],
            comps=["dass"],
        )

    @classmethod
    def english_hf(cls) -> "CFGParams":
        params = cls.english()
        params.head_initial = False
        return params

    @classmethod
    def english_sf(cls) -> "CFGParams":
        params = cls.english()
        params.spec_initial = False
        return params


@dataclass
class SCFGParams:
    a: CFGParams
    b: CFGParams
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = secrets.token_hex(8)

    @property
    def agreement_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.a.agreement_enabled or self.b.agreement_enabled,
            "a": {
                "config": self.a.agreement_config.to_dict(),
                "suffixes": dict(self.a.agreement_suffixes),
                "pronoun_paradigms": serialize_bundles(self.a.pronoun_paradigms),
                "verb_paradigms": serialize_bundles(self.a.verb_paradigms),
                "noun_paradigms": serialize_bundles(self.a.noun_paradigms),
                "propn_paradigms": serialize_bundles(self.a.propn_paradigms),
            },
            "b": {
                "config": self.b.agreement_config.to_dict(),
                "suffixes": dict(self.b.agreement_suffixes),
                "pronoun_paradigms": serialize_bundles(self.b.pronoun_paradigms),
                "verb_paradigms": serialize_bundles(self.b.verb_paradigms),
                "noun_paradigms": serialize_bundles(self.b.noun_paradigms),
                "propn_paradigms": serialize_bundles(self.b.propn_paradigms),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        builder = RuleBuilder(self)
        rules = builder.build_rules()
        lexicon = builder.build_lexicon()
        agreement_lines = builder.build_agreement_summary()
        grammar_str = "\n".join(
            rules + builder.build_display_lexicon() + agreement_lines
        )
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "name": self.name,
            "grammar_str": grammar_str,
            "n_rules": len(rules),
            "n_words": len(lexicon),
            "agreement_metadata": self.agreement_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SCFGParams":
        return cls(
            name=data.get("name"),
            a=CFGParams.from_dict(data["a"]),
            b=CFGParams.from_dict(data["b"]),
        )

    @classmethod
    def english_english_hf(cls) -> "SCFGParams":
        return cls(a=CFGParams.english(), b=CFGParams.english_hf())

    @classmethod
    def english_english_sf(cls) -> "SCFGParams":
        return cls(a=CFGParams.english(), b=CFGParams.english_sf())


@dataclass
class Derivation:
    left_full: str
    left_phonetic: str
    right_full: str
    right_phonetic: str
    left_tree: str
    right_tree: str
    depth: int
    features: FeatureBundle = field(default_factory=FeatureBundle)
    trace: str = ""


class SCFG:
    @property
    def rules(self) -> list[str]:
        return self.builder.build_rules()

    @property
    def lexicon(self) -> list[str]:
        return self.builder.build_lexicon()

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def n_words(self) -> int:
        return len(self.lexicon)

    @property
    def as_cfg(self) -> str:
        return "\n".join(self.rules + self.lexicon)

    @property
    def recursive_parents(self) -> set[str]:
        recursive_parents: set[str] = set()
        for parent, rules in self.rules_dict.items():
            for prod_options in rules:
                a_prod = prod_options[0]
                if any(symbol in self.recursive_symbols for symbol in a_prod):
                    recursive_parents.add(parent)
        return recursive_parents

    def __init__(self, params: SCFGParams):
        self.params = params
        self.builder = RuleBuilder(params)
        self.start_symbol = "S"
        self.rules_dict = self._parse_rules()
        self.recursive_symbols: set[str] = {"CP_embed"}
        self.agreement_enabled = params.a.agreement_enabled or params.b.agreement_enabled

    def _parse_rules(self) -> dict[str, list[Rule]]:
        rules: dict[str, list[Rule]] = {}
        for line in self.as_cfg.strip().split("\n"):
            line = line.strip()
            if not line or "->" not in line:
                continue
            lhs, rhs_str = map(str.strip, line.split("->", 1))
            match = re.search(r"<(.*),\s*(.*)>", rhs_str)
            if not match:
                continue
            a_rhs_str, b_rhs_str = match.groups()
            a_symbols = tuple(a_rhs_str.strip().split())
            b_symbols = tuple(b_rhs_str.strip().split())
            rules.setdefault(lhs, []).append((a_symbols, b_symbols))
        return rules

    def sample(
        self,
        min_depth: int = 0,
        max_depth: int = 1,
        rng: random.Random | None = None,
    ) -> dict[str, Any]:
        if rng is None:
            rng = random.Random()
        assert min_depth >= 0
        assert min_depth <= max_depth
        if self.agreement_enabled:
            result = self._sample_agreement_recursive(
                self.start_symbol,
                rng=rng,
                current_depth=0,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            return {
                "left": " ".join(result.left_full.split()),
                "left_phonetic": " ".join(result.left_phonetic.split()),
                "right": " ".join(result.right_full.split()),
                "right_phonetic": " ".join(result.right_phonetic.split()),
                "left_tree": " ".join(result.left_tree.split()),
                "right_tree": " ".join(result.right_tree.split()),
                "depth": result.depth,
                "subject_features": result.features.to_dict(),
                "verb_features": result.features.to_dict(),
                "agreement_ok": True,
                "agreement_trace": result.trace,
            }

        result = self._sample_recursive(
            self.start_symbol,
            rng,
            current_depth=0,
            max_depth=max_depth,
            min_depth=min_depth,
        )
        return {
            "left": " ".join(result["left_full"].split()),
            "left_phonetic": " ".join(result["left_phonetic"].split()),
            "right": " ".join(result["right_full"].split()),
            "right_phonetic": " ".join(result["right_phonetic"].split()),
            "left_tree": " ".join(result["left_tree"].split()),
            "right_tree": " ".join(result["right_tree"].split()),
            "depth": result["depth"],
        }

    def _sample_recursive(
        self,
        symbol: str,
        rng: random.Random,
        current_depth: int,
        min_depth: int,
        max_depth: int,
    ) -> dict[str, Any]:
        if symbol not in self.rules_dict:
            clean_symbol = symbol.strip("'")
            phonetic_string = "" if clean_symbol.startswith("∅") else clean_symbol
            return {
                "left_full": clean_symbol,
                "left_phonetic": phonetic_string,
                "right_full": clean_symbol,
                "right_phonetic": phonetic_string,
                "left_tree": clean_symbol,
                "right_tree": clean_symbol,
                "depth": current_depth,
            }

        possible_rules = self.rules_dict[symbol]
        if current_depth < min_depth and symbol in self.recursive_parents:
            possible_rules = [
                rule for rule in possible_rules if any(s in self.recursive_symbols for s in rule[0])
            ]
        if current_depth >= max_depth:
            possible_rules = [
                rule for rule in possible_rules if not any(s in self.recursive_symbols for s in rule[0])
            ]
            if not possible_rules:
                return {
                    "left_full": "",
                    "left_phonetic": "",
                    "right_full": "",
                    "right_phonetic": "",
                    "left_tree": f"({symbol})",
                    "right_tree": f"({symbol})",
                    "depth": current_depth,
                }

        chosen_left_prod, chosen_right_prod = rng.choice(possible_rules)
        sub_derivations: dict[str, dict[str, Any]] = {}
        unique_non_terminals: list[str] = []
        seen: set[str] = set()
        for child in chosen_left_prod + chosen_right_prod:
            if child in self.rules_dict and child not in seen:
                unique_non_terminals.append(child)
                seen.add(child)
        for child in unique_non_terminals:
            new_depth = current_depth + (1 if child in self.recursive_symbols else 0)
            sub_derivations[child] = self._sample_recursive(
                child,
                rng=rng,
                current_depth=new_depth,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        max_depth_reached = current_depth
        for derivation in sub_derivations.values():
            max_depth_reached = max(max_depth_reached, derivation["depth"])

        def _assemble(sequence: tuple[str, ...], side: str) -> tuple[str, str, str]:
            full: list[str] = []
            phon: list[str] = []
            tree_parts: list[str] = []
            for item in sequence:
                if item in sub_derivations:
                    derivation = sub_derivations[item]
                    full.append(derivation[f"{side}_full"])
                    phon.append(derivation[f"{side}_phonetic"])
                    tree_parts.append(derivation[f"{side}_tree"])
                else:
                    clean_symbol = item.strip("'")
                    full.append(clean_symbol)
                    if not clean_symbol.startswith("∅"):
                        phon.append(clean_symbol)
                    tree_parts.append(clean_symbol)
            return " ".join(full), " ".join(phon), f"({symbol} {' '.join(tree_parts)})"

        left_full, left_phon, left_tree = _assemble(chosen_left_prod, "left")
        right_full, right_phon, right_tree = _assemble(chosen_right_prod, "right")
        return {
            "left_full": left_full,
            "left_phonetic": left_phon,
            "right_full": right_full,
            "right_phonetic": right_phon,
            "left_tree": left_tree,
            "right_tree": right_tree,
            "depth": max_depth_reached,
        }

    def _terminal_derivation(
        self,
        symbol: str,
        left_surface: str,
        right_surface: str,
        depth: int,
        features: FeatureBundle | None = None,
        trace: str = "",
    ) -> Derivation:
        left_phonetic = "" if left_surface.startswith("∅") else left_surface
        right_phonetic = "" if right_surface.startswith("∅") else right_surface
        return Derivation(
            left_full=left_surface,
            left_phonetic=left_phonetic,
            right_full=right_surface,
            right_phonetic=right_phonetic,
            left_tree=left_surface,
            right_tree=right_surface,
            depth=depth,
            features=features or FeatureBundle(),
            trace=trace,
        )

    def _combine_derivations(
        self,
        symbol: str,
        left_children: list[Derivation],
        right_children: list[Derivation],
        features: FeatureBundle | None = None,
        trace: str = "",
    ) -> Derivation:
        depth = 0
        for child in left_children + right_children:
            depth = max(depth, child.depth)
        left_full = " ".join([child.left_full for child in left_children if child.left_full])
        left_phonetic = " ".join([child.left_phonetic for child in left_children if child.left_phonetic])
        right_full = " ".join([child.right_full for child in right_children if child.right_full])
        right_phonetic = " ".join([child.right_phonetic for child in right_children if child.right_phonetic])
        left_tree = f"({symbol} {' '.join(child.left_tree for child in left_children if child.left_tree)})"
        right_tree = f"({symbol} {' '.join(child.right_tree for child in right_children if child.right_tree)})"
        return Derivation(
            left_full=left_full,
            left_phonetic=left_phonetic,
            right_full=right_full,
            right_phonetic=right_phonetic,
            left_tree=left_tree,
            right_tree=right_tree,
            depth=depth,
            features=features or FeatureBundle(),
            trace=trace,
        )

    def _order_head_comp(self, head: Derivation, comp: Derivation, head_initial: bool) -> list[Derivation]:
        return [head, comp] if head_initial else [comp, head]

    def _order_spec_head(self, spec: Derivation, head: list[Derivation], spec_initial: bool) -> list[Derivation]:
        return [spec, *head] if spec_initial else [*head, spec]

    def _sample_agreement_recursive(
        self,
        symbol: str,
        rng: random.Random,
        current_depth: int,
        min_depth: int,
        max_depth: int,
        inherited_features: FeatureBundle | None = None,
        role: str = "object",
    ) -> Derivation:
        if symbol == "S":
            return self._sample_agreement_recursive("CP_matrix", rng, current_depth, min_depth, max_depth)

        if symbol == "CP_matrix":
            cnull = self._terminal_derivation("CNULL", "∅", "∅", current_depth)
            tp = self._sample_agreement_recursive("TP", rng, current_depth, min_depth, max_depth)
            return self._combine_derivations("CP_matrix", [cnull, tp], [cnull, tp], features=tp.features, trace=tp.trace)

        if symbol == "CP_embed":
            comp = self._terminal_derivation(
                "C",
                self.params.a.choose_comp(rng),
                self.params.b.choose_comp(rng),
                current_depth,
            )
            tp = self._sample_agreement_recursive("TP", rng, current_depth + 1, min_depth, max_depth)
            return self._combine_derivations("CP_embed", [comp, tp], [comp, tp], features=tp.features, trace=tp.trace)

        if symbol == "TP":
            subject = self._sample_agreement_recursive(
                "NP_SUBJ",
                rng,
                current_depth,
                min_depth,
                max_depth,
                role="subject",
            )
            t = self._terminal_derivation(
                "T",
                self.params.a.choose_t(rng),
                self.params.b.choose_t(rng),
                current_depth,
            )
            vp = self._sample_agreement_recursive(
                "VP",
                rng,
                current_depth,
                min_depth,
                max_depth,
                inherited_features=subject.features,
            )
            left_tbar = self._order_head_comp(t, vp, self.params.a.head_initial)
            right_tbar = self._order_head_comp(t, vp, self.params.b.head_initial)
            left_children = self._order_spec_head(subject, left_tbar, self.params.a.spec_initial)
            right_children = self._order_spec_head(subject, right_tbar, self.params.b.spec_initial)
            return self._combine_derivations(
                "TP",
                left_children,
                right_children,
                features=subject.features,
                trace=f"subj={subject.features.key()}",
            )

        if symbol == "NP_SUBJ":
            choices = ["PRON", "DP"]
            if self.params.a.pro_drop or self.params.b.pro_drop:
                choices.append("PRO")
            choices.append("PROPN")
            choice = rng.choice(choices)
            if choice == "PRO":
                return self._terminal_derivation("PRO", "∅", "∅", current_depth, FeatureBundle(person="3", number="sg"), trace="pro")
            if choice == "PRON":
                left_surface, left_features = self.params.a.choose_pronoun(rng)
                right_surface, right_features = self.params.b.choose_pronoun(rng)
                features = FeatureUnifier.unify(left_features, right_features) or left_features or right_features
                return self._terminal_derivation("PRON", left_surface, right_surface, current_depth, features=features, trace=f"pron={features.key()}")
            if choice == "PROPN":
                left_surface, left_features = self.params.a.choose_propn(rng)
                right_surface, right_features = self.params.b.choose_propn(rng)
                features = FeatureUnifier.unify(left_features, right_features) or left_features
                return self._terminal_derivation("PROPN", left_surface, right_surface, current_depth, features=features, trace="propn=3sg")
            return self._sample_agreement_recursive("DP", rng, current_depth, min_depth, max_depth, role=role)

        if symbol == "VP":
            verb = self._sample_agreement_recursive(
                "V",
                rng,
                current_depth,
                min_depth,
                max_depth,
                inherited_features=inherited_features,
            )
            obj = self._sample_agreement_recursive("OBJ_PHRASE", rng, current_depth, min_depth, max_depth)
            left_children = self._order_head_comp(verb, obj, self.params.a.head_initial)
            right_children = self._order_head_comp(verb, obj, self.params.b.head_initial)
            return self._combine_derivations("VP", left_children, right_children, features=inherited_features or FeatureBundle(), trace=verb.trace)

        if symbol == "V":
            left_surface, left_features = self.params.a.choose_verb(rng, inherited_features)
            right_surface, right_features = self.params.b.choose_verb(rng, inherited_features)
            features = FeatureUnifier.unify(left_features, right_features) or left_features or right_features
            return self._terminal_derivation("V", left_surface, right_surface, current_depth, features=features, trace=f"verb={features.key()}")

        if symbol == "OBJ_PHRASE":
            if current_depth < min_depth:
                choice = "CP_embed"
            elif current_depth >= max_depth:
                choice = "DP"
            else:
                choice = rng.choice(["DP", "CP_embed"])
            return self._sample_agreement_recursive(choice, rng, current_depth, min_depth, max_depth)

        if symbol == "DP":
            use_propn = rng.random() < 0.25
            if use_propn:
                left_surface, features = self.params.a.choose_propn(rng)
                right_surface, _ = self.params.b.choose_propn(rng)
                left_children = [self._terminal_derivation("PROPN", left_surface, right_surface, current_depth, features=features)]
                right_children = list(left_children)
                if self.params.a.proper_with_det:
                    left_det = self._terminal_derivation("DET_def", self.params.a.choose_det(rng, True), self.params.a.choose_det(rng, True), current_depth)
                    left_children = [left_det, left_children[0]]
                if self.params.b.proper_with_det:
                    right_det = self._terminal_derivation("DET_def", self.params.b.choose_det(rng, True), self.params.b.choose_det(rng, True), current_depth)
                    right_children = [right_det, right_children[0]]
                return self._combine_derivations("DP", left_children, right_children, features=features, trace="dp-propn")

            definite = rng.random() < 0.5
            number = rng.choice(("sg", "pl")) if self.agreement_enabled else None
            left_det_surface = self.params.a.choose_det(rng, definite)
            right_det_surface = self.params.b.choose_det(rng, definite)
            left_det = self._terminal_derivation("DET", left_det_surface, right_det_surface, current_depth)
            noun = self._sample_agreement_recursive(
                "NP",
                rng,
                current_depth,
                min_depth,
                max_depth,
                inherited_features=FeatureBundle(person="3", number=number),
            )
            return self._combine_derivations("DP", [left_det, noun], [left_det, noun], features=noun.features, trace=f"dp={noun.features.key()}")

        if symbol == "NP":
            include_adj = bool(self.params.a.adj_lex and self.params.b.adj_lex and rng.random() < 0.35)
            noun_head = self._sample_agreement_recursive(
                "N_HEAD",
                rng,
                current_depth,
                min_depth,
                max_depth,
                inherited_features=inherited_features,
            )
            if not include_adj:
                return noun_head
            adj = self._sample_agreement_recursive("AdjP", rng, current_depth, min_depth, max_depth)
            return self._combine_derivations("NP", [adj, noun_head], [adj, noun_head], features=noun_head.features, trace=noun_head.trace)

        if symbol == "N_HEAD":
            if rng.random() < 0.25:
                left_surface, left_features = self.params.a.choose_propn(rng)
                right_surface, right_features = self.params.b.choose_propn(rng)
                features = FeatureUnifier.unify(left_features, right_features) or left_features
                return self._terminal_derivation("PROPN", left_surface, right_surface, current_depth, features=features, trace="nhead-propn")
            target_number = inherited_features.number if inherited_features else None
            left_surface, left_features = self.params.a.choose_noun(rng, target_number)
            right_surface, right_features = self.params.b.choose_noun(rng, target_number)
            features = FeatureUnifier.unify(left_features, right_features) or left_features or right_features
            return self._terminal_derivation("N", left_surface, right_surface, current_depth, features=features, trace=f"noun={features.key()}")

        if symbol == "AdjP":
            return self._terminal_derivation(
                "ADJ",
                self.params.a.choose_adj(rng),
                self.params.b.choose_adj(rng),
                current_depth,
            )

        return self._terminal_derivation(symbol, symbol, symbol, current_depth)


class RuleBuilder:
    def __init__(self, params: CFGParams | SCFGParams):
        self.params = params

    @property
    def is_sync(self) -> bool:
        return isinstance(self.params, SCFGParams)

    def emit(self, lhs: str, rhs: str | tuple[str, ...]) -> str:
        if isinstance(rhs, str):
            return f"{lhs} -> {rhs}"
        assert len(rhs) == 2
        return f"{lhs} -> <{rhs[0]}, {rhs[1]}>"

    def _shell_rules(
        self,
        head: str,
        spec: str,
        comp: str,
        head_initial: bool,
        spec_initial: bool,
        head_initial_b: bool | None = None,
        spec_initial_b: bool | None = None,
    ) -> list[str]:
        rules: list[str] = []
        phrase = f"{head}P"
        xbar = f"{head}BAR"
        if head_initial_b is not None and spec_initial_b is not None:
            lspec = f"{spec} {xbar}" if spec_initial else f"{xbar} {spec}"
            rspec = f"{spec} {xbar}" if spec_initial_b else f"{xbar} {spec}"
            rules.append(f"{phrase} -> <{lspec}, {rspec}>")
            lhead = f"{head} {comp}" if head_initial else f"{comp} {head}"
            rhead = f"{head} {comp}" if head_initial_b else f"{comp} {head}"
            rules.append(f"{xbar} -> <{lhead}, {rhead}>")
            return rules
        if spec_initial:
            rules.append(f"{phrase} -> {spec} {xbar}")
        else:
            rules.append(f"{phrase} -> {xbar} {spec}")
        if head_initial:
            rules.append(f"{xbar} -> {head} {comp}")
        else:
            rules.append(f"{xbar} -> {comp} {head}")
        return rules

    def _lex(self, pos: str, words: list[str] | tuple[list[str], list[str]]) -> list[str]:
        if isinstance(words, tuple):
            awords, bwords = words
            return [f"{pos} -> <'{aw}', '{bw}'>" for aw, bw in zip(awords, bwords)]
        return [f"{pos} -> '{word}'" for word in words]

    def _paired_verb_forms(self) -> tuple[list[str], list[str]]:
        if not self.params.a.agreement_enabled and not self.params.b.agreement_enabled:
            return self.params.a.verb_lex, self.params.b.verb_lex
        left: list[str] = []
        right: list[str] = []
        count = min(len(self.params.a.verb_paradigms), len(self.params.b.verb_paradigms))
        keys = [
            bundle.key(self.params.a.latent_axes)
            for bundle in feature_inventory(
                self.params.a.latent_axes,
                gender_values=self.params.a.gender_values,
            )
        ]
        for index in range(count):
            for key in keys:
                left.append(self.params.a.verb_paradigms[index]["forms"].get(key, self.params.a.verb_paradigms[index]["lemma"]))
                right.append(self.params.b.verb_paradigms[index]["forms"].get(key, self.params.b.verb_paradigms[index]["lemma"]))
        return left, right

    def _paired_noun_forms(self) -> tuple[list[str], list[str]]:
        if not self.params.a.agreement_enabled and not self.params.b.agreement_enabled:
            return self.params.a.noun_lex, self.params.b.noun_lex
        left: list[str] = []
        right: list[str] = []
        count = min(len(self.params.a.noun_paradigms), len(self.params.b.noun_paradigms))
        for index in range(count):
            for key in ("number=sg", "number=pl"):
                left.append(self.params.a.noun_paradigms[index]["forms"].get(key, self.params.a.noun_paradigms[index]["lemma"]))
                right.append(self.params.b.noun_paradigms[index]["forms"].get(key, self.params.b.noun_paradigms[index]["lemma"]))
        return left, right

    def _paired_pronoun_forms(self) -> tuple[list[str], list[str]]:
        if not self.params.a.agreement_enabled and not self.params.b.agreement_enabled:
            return self.params.a.pron_lex, self.params.b.pron_lex
        left = [entry["form"] for entry in self.params.a.pronoun_paradigms]
        right = [entry["form"] for entry in self.params.b.pronoun_paradigms]
        count = min(len(left), len(right))
        return left[:count], right[:count]

    def _feature_label(self, bundle: FeatureBundle) -> str:
        parts: list[str] = []
        if bundle.person is not None:
            parts.append(bundle.person)
        if bundle.number is not None:
            parts.append(bundle.number)
        if bundle.gender is not None:
            parts.append(bundle.gender)
        return ".".join(parts) if parts else "default"

    def _display_pronoun_paradigms(self) -> list[str]:
        if not (self.params.a.agreement_enabled or self.params.b.agreement_enabled):
            return self._lex("PRON", self._paired_pronoun_forms())
        lines: list[str] = []
        count = min(
            len(self.params.a.pronoun_paradigms),
            len(self.params.b.pronoun_paradigms),
        )
        for index in range(count):
            left_entry = self.params.a.pronoun_paradigms[index]
            right_entry = self.params.b.pronoun_paradigms[index]
            label = self._feature_label(left_entry["features"])
            lines.append(
                f"PRON[{label}] -> <'{left_entry['form']}', '{right_entry['form']}'>"
            )
        return lines

    def _display_noun_paradigms(self) -> list[str]:
        if not (self.params.a.agreement_enabled or self.params.b.agreement_enabled):
            return self._lex("N", self._paired_noun_forms())
        lines: list[str] = []
        count = min(len(self.params.a.noun_paradigms), len(self.params.b.noun_paradigms))
        for index in range(count):
            left_entry = self.params.a.noun_paradigms[index]
            right_entry = self.params.b.noun_paradigms[index]
            left_features = left_entry.get("features", FeatureBundle())
            stem = f"N{index + 1}"
            lemma_label = f"{stem}[lemma]"
            if left_features.gender is not None:
                lemma_label = f"{stem}[lemma.{left_features.gender}]"
            lines.append(
                f"{lemma_label} -> <'{left_entry['lemma']}', '{right_entry['lemma']}'>"
            )
            sg_label = f"{stem}[sg]"
            pl_label = f"{stem}[pl]"
            if left_features.gender is not None:
                sg_label = f"{stem}[sg.{left_features.gender}]"
                pl_label = f"{stem}[pl.{left_features.gender}]"
            lines.append(
                f"{sg_label} -> <"
                f"'{left_entry['forms']['number=sg']}', "
                f"'{right_entry['forms']['number=sg']}'>"
            )
            lines.append(
                f"{pl_label} -> <"
                f"'{left_entry['forms']['number=pl']}', "
                f"'{right_entry['forms']['number=pl']}'>"
            )
        return lines

    def _display_propn_paradigms(self) -> list[str]:
        if not (self.params.a.agreement_enabled or self.params.b.agreement_enabled):
            return self._lex("PROPN", (self.params.a.propn_lex, self.params.b.propn_lex))
        lines: list[str] = []
        count = min(len(self.params.a.propn_paradigms), len(self.params.b.propn_paradigms))
        for index in range(count):
            left_entry = self.params.a.propn_paradigms[index]
            right_entry = self.params.b.propn_paradigms[index]
            label = self._feature_label(left_entry["features"])
            lines.append(
                f"PROPN{index + 1}[{label}] -> "
                f"<'{left_entry['lemma']}', '{right_entry['lemma']}'>"
            )
        return lines

    def _display_verb_paradigms(self) -> list[str]:
        if not (self.params.a.agreement_enabled or self.params.b.agreement_enabled):
            return self._lex("V", self._paired_verb_forms())
        lines: list[str] = []
        count = min(len(self.params.a.verb_paradigms), len(self.params.b.verb_paradigms))
        bundles = feature_inventory(
            self.params.a.latent_axes,
            gender_values=self.params.a.gender_values,
        )
        for index in range(count):
            left_entry = self.params.a.verb_paradigms[index]
            right_entry = self.params.b.verb_paradigms[index]
            stem = f"V{index + 1}"
            lines.append(
                f"{stem}[lemma] -> <'{left_entry['lemma']}', '{right_entry['lemma']}'>"
            )
            for bundle in bundles:
                key = bundle.key(self.params.a.latent_axes)
                label = self._feature_label(bundle)
                lines.append(
                    f"{stem}[{label}] -> "
                    f"<'{left_entry['forms'][key]}', '{right_entry['forms'][key]}'>"
                )
        return lines

    def build_display_lexicon(self) -> list[str]:
        if not self.is_sync:
            return self.build_lexicon()
        lines: list[str] = []
        lines += self._lex("DET", (self.params.a.det_def_lex, self.params.b.det_def_lex))
        lines += self._lex("T", (self.params.a.tense_lex, self.params.b.tense_lex))
        lines += self._display_verb_paradigms()
        lines += self._display_noun_paradigms()
        lines += self._display_propn_paradigms()
        lines += self._display_pronoun_paradigms()
        lines += self._lex("ADJ", (self.params.a.adj_lex, self.params.b.adj_lex))
        lines += self._lex("C", (self.params.a.comp_lex, self.params.b.comp_lex))
        lines.append("CNULL -> <'∅', '∅'>")
        if self.params.a.pro_drop or self.params.b.pro_drop:
            lines.append("PRO -> <'∅', '∅'>")
        return lines

    def build_rules(self) -> list[str]:
        rules: list[str] = []
        if self.is_sync:
            rules.append(self.emit("S", ("CP_matrix", "CP_matrix")))
            rules.append(self.emit("CP_matrix", ("CNULL TP", "CNULL TP")))
            rules.append(self.emit("CP_embed", ("C TP", "C TP")))
            rules += self._shell_rules(
                head="T",
                spec="NP_SUBJ",
                comp="VP",
                head_initial=self.params.a.head_initial,
                spec_initial=self.params.a.spec_initial,
                head_initial_b=self.params.b.head_initial,
                spec_initial_b=self.params.b.spec_initial,
            )
            if self.params.a.pro_drop or self.params.b.pro_drop:
                rules.append(self.emit("NP_SUBJ", ("PRO", "PRO")))
            rules.append(self.emit("NP_SUBJ", ("PRON", "PRON")))
            rules.append(self.emit("NP_SUBJ", ("PROPN", "PROPN")))
            rules.append(self.emit("NP_SUBJ", ("DP", "DP")))
            rules += self._shell_rules(
                head="V",
                spec="",
                comp="OBJ_PHRASE",
                head_initial=self.params.a.head_initial,
                spec_initial=self.params.a.spec_initial,
                head_initial_b=self.params.b.head_initial,
                spec_initial_b=self.params.b.spec_initial,
            )
            rules.append(self.emit("OBJ_PHRASE", ("DP", "DP")))
            rules.append(self.emit("OBJ_PHRASE", ("CP_embed", "CP_embed")))
            rules.append(self.emit("DP", ("DET NP", "DET NP")))
            rules.append(self.emit("NP", ("N_HEAD", "N_HEAD")))
            rules.append(self.emit("NP", ("AdjP N_HEAD", "AdjP N_HEAD")))
            rules.append(self.emit("AdjP", ("ADJ", "ADJ")))
            rules.append(self.emit("N_HEAD", ("N", "N")))
            rules.append(self.emit("N_HEAD", ("PROPN", "PROPN")))
            return rules

        rules.append(self.emit("S", "CP_matrix"))
        rules.append(self.emit("CP_matrix", "CNULL TP"))
        rules.append(self.emit("CP_embed", "C TP"))
        rules += self._shell_rules(
            head="T",
            spec="NP_SUBJ",
            comp="VP",
            head_initial=self.params.head_initial,
            spec_initial=self.params.spec_initial,
        )
        if self.params.pro_drop:
            rules.append(self.emit("NP_SUBJ", "PRO"))
        rules.append(self.emit("NP_SUBJ", "PRON"))
        rules.append(self.emit("NP_SUBJ", "PROPN"))
        rules.append(self.emit("NP_SUBJ", "DP"))
        rules += self._shell_rules(
            head="V",
            spec="",
            comp="OBJ_PHRASE",
            head_initial=self.params.head_initial,
            spec_initial=self.params.spec_initial,
        )
        rules.append(self.emit("OBJ_PHRASE", "DP"))
        rules.append(self.emit("OBJ_PHRASE", "CP_embed"))
        rules.append(self.emit("DP", "DET NP"))
        rules.append(self.emit("NP", "N_HEAD"))
        rules.append(self.emit("NP", "AdjP N_HEAD"))
        rules.append(self.emit("AdjP", "ADJ"))
        rules.append(self.emit("N_HEAD", "N"))
        rules.append(self.emit("N_HEAD", "PROPN"))
        return rules

    def build_lexicon(self) -> list[str]:
        rules: list[str] = []
        if self.is_sync:
            rules += self._lex("DET", (self.params.a.det_def_lex, self.params.b.det_def_lex))
            rules += self._lex("T", (self.params.a.tense_lex, self.params.b.tense_lex))
            rules += self._lex("V", self._paired_verb_forms())
            rules += self._lex("N", self._paired_noun_forms())
            rules += self._lex("PROPN", (self.params.a.propn_lex, self.params.b.propn_lex))
            rules += self._lex("PRON", self._paired_pronoun_forms())
            rules += self._lex("ADJ", (self.params.a.adj_lex, self.params.b.adj_lex))
            rules += self._lex("C", (self.params.a.comp_lex, self.params.b.comp_lex))
            rules.append("CNULL -> <'∅', '∅'>")
            if self.params.a.pro_drop or self.params.b.pro_drop:
                rules.append("PRO -> <'∅', '∅'>")
            return rules

        rules += self._lex("DET", self.params.det_def_lex)
        rules += self._lex("T", self.params.tense_lex)
        rules += self._lex("V", self.params.verb_lex)
        rules += self._lex("N", self.params.noun_lex)
        rules += self._lex("PROPN", self.params.propn_lex)
        rules += self._lex("PRON", self.params.pron_lex)
        rules += self._lex("ADJ", self.params.adj_lex)
        rules += self._lex("C", self.params.comp_lex)
        rules.append("CNULL -> '∅'")
        if self.params.pro_drop:
            rules.append("PRO -> '∅'")
        return rules

    def build_agreement_summary(self) -> list[str]:
        if not self.is_sync:
            return []
        if not (self.params.a.agreement_enabled or self.params.b.agreement_enabled):
            return []
        return [
            "# Agreement metadata:",
            f"# a.active_axes={','.join(self.params.a.agreement_axes)} latent_gender={self.params.a.latent_gender} realize_gender={self.params.a.realize_gender}",
            f"# b.active_axes={','.join(self.params.b.agreement_axes)} latent_gender={self.params.b.latent_gender} realize_gender={self.params.b.realize_gender}",
            "# Verb paradigms are realized via synthetic suffixation keyed by person/number.",
        ]
