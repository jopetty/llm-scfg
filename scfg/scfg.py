import random
import re
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Set, Tuple, Union

import numpy as np

Rule = Tuple[Tuple[str, ...], Tuple[str, ...]]

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

CYRILLIC_CONSONANTS: list[str] = list(
    "бвгджзйклмнпрстфхцчшщ"
)
CYRILLIC_VOWELS: list[str] = list("аеёиоуыэюя")
CYRILLIC_SONORITY_HIERARCHY: dict[str, float] = {
    "п": 0.1, "б": 0.15,
    "т": 0.1, "д": 0.15,
    "к": 0.1, "г": 0.15,
    "ф": 0.2, "в": 0.25,
    "с": 0.2, "з": 0.25,
    "ш": 0.2, "ж": 0.25,
    "х": 0.2,
    "ц": 0.2, "ч": 0.3, "щ": 0.3,
    "м": 0.4, "н": 0.4,
    "р": 0.5, "л": 0.5,
    "й": 0.6
}

YIDDISH_CONSONANTS: list[str] = list("בגדהװזשחטיּכּכלמנפּפֿצקרשת")
YIDDISH_VOWELS: list[str] = list("אַאָוּיִײײַױע")
YIDDISH_SONORITY_HIERARCHY: dict[str, float] = {
    "פּ": 0.10,  # p (stop)
    "ב": 0.15,   # b (stop)
    "ט": 0.10,   # t (stop)
    "ד": 0.15,   # d (stop)
    "כּ": 0.10,  # k (stop)
    "ג": 0.15,   # g (stop)
    "ק": 0.10,   # k (stop)
    "פֿ": 0.20,  # f (fricative)
    "װ": 0.25,  # v (fricative)
    "ס": 0.20,  # s (fricative)
    "ז": 0.25,  # z (fricative)
    "ש": 0.20,  # sh (fricative)
    "כ": 0.20,  # kh (fricative)
    "ח": 0.20,  # kh (fricative)
    "ה": 0.20,  # h (fricative)
    "צ": 0.30,  # ts (affricate)
    "מ": 0.40,  # m (nasal)
    "נ": 0.40,  # n (nasal)
    "ל": 0.50,  # l (liquid)
    "ר": 0.50,  # r (liquid)
    "י": 0.60   # y (glide)
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

    # Parameters
    head_initial: bool = True
    spec_initial: bool = True
    pro_drop: bool = False
    proper_with_det: bool = False
    syllable_structure: str | None = None
    avg_syllables_per_word: float = 2
    max_consonants: int = 2
    rng_seed: int = 42

    # Lexicon
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
    p_space: float = 0.1
    orthography: str = "latin"

    def to_dict(self) -> Dict[str, Any]:
        param_dict = asdict(self)

        # override lexicon with generated values
        param_dict["verbs"] = self.verb_lex
        param_dict["nouns"] = self.noun_lex
        param_dict["propns"] = self.propn_lex
        param_dict["prons"] = self.pron_lex
        param_dict["adjs"] = self.adj_lex
        param_dict["det_def"] = self.det_def_lex
        param_dict["det_indef"] = self.det_indef_lex
        param_dict["comps"] = self.comp_lex
        param_dict["tenses"] = self.tense_lex
        param_dict["asps"] = self.asp_lex

        return param_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CFGParams":
        return cls(**data)

    # Properties
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
        def _resolve(val):
            if isinstance(val, int):
                return [self._sample_string() for _ in range(val)]
            return list(val)

        self.rng = random.Random(self.rng_seed)

        # Set vowels, consonants, and sonority hierarchy based on orthography
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

        # Determine syllable structure
        if self.syllable_structure is None:
            self.syllable_structure = self.rng.choice(SYLLABLE_STRUCTURES)

        self.verb_lex = _resolve(self.verbs)
        self.noun_lex = _resolve(self.nouns)
        self.propn_lex = _resolve(self.propns)
        self.pron_lex = _resolve(self.prons)
        self.adj_lex = _resolve(self.adjs)
        self.det_def_lex = _resolve(self.det_def)
        self.det_indef_lex = _resolve(self.det_indef)
        self.comp_lex = _resolve(self.comps)
        self.tense_lex = _resolve(self.tenses)
        self.asp_lex = _resolve(self.asps)

    def _parse_syllable_format(self, template: str):
        tokens: list[str] = re.findall(r"C\*|V\*|C\?|V\?|C|V", template)
        return tokens

    def _generate_cluster(self, size) -> str:
        

        chars: list[str] = list(self.sonority_hierarchy.keys())
        weights: list[float] = [self.sonority_hierarchy[c] for c in chars]
        cluster: str = ""

        for _ in range(size):
            c: str = self.rng.choices(chars, weights=weights, k=1)[0]
            cluster = cluster + c

        return cluster

    def _generate_syllable(self, tokens: list[str]) -> str:
        result: list[str] = []

        for token in tokens:
            if token == "C":
                result.append(self.rng.choice(self.consonants))
            elif token == "V":
                result.append(self.rng.choice(self.vowels))
            elif token == "C*":
                n: int = self.rng.randint(0, self.max_consonants)
                result.extend(self._generate_cluster(n))
            elif token == "V*":
                n: int = self.rng.randint(1, 2)
                result.extend(self.rng.choices(self.vowels, k=n))
            elif token == "C?":
                if self.rng.random() < 0.5:
                    result.append(self.rng.choice(self.consonants))
            elif token == "V?":
                if self.rng.random() < 0.5:
                    result.append(self.rng.choice(self.vowels))
        return "".join(result)

    def _sample_string(self):
        def _zero_truncated_poisson(rate: float) -> int:
            """Sample from a zero-truncated Poisson distribution."""
            u: float = np.random.uniform(np.exp(-rate), 1)
            t: float = -np.log(u)
            return 1 + np.random.poisson(rate - t)

        string: str = ""
        lambda_poisson: float = self.avg_syllables_per_word
        num_syllables: int = _zero_truncated_poisson(lambda_poisson)
        for _ in range(num_syllables + 1):
            if self.rng.random() < self.p_space:
                string += " "
            string += self._generate_syllable(
                self.syllable_structure,
            )

        return string

    # Class instances
    @classmethod
    def english(cls):
        return cls(
            head_initial=True,
            spec_initial=True,
            pro_drop=False,
            proper_with_det=False,
            verbs=["eats", "sees", "loves", "hears"],
            nouns=["tree", "horse", "dog", "cat", "apple"],
            propns=["john", "mary", "sue", "bob"],
            prons=["he", "she", "they", "it"],
            adjs=["big", "small", "red", "green", "blue", "fuzzy", "round"],
            det_def=["the"],
            det_indef=["a"],
            comps=["that"],
        )

    @classmethod
    def german(cls):
        return cls(
            head_initial=True,
            spec_initial=True,
            pro_drop=False,
            proper_with_det=False,
            verbs=["isst", "sieht", "liebt", "hört"],
            nouns=["baum", "pferd", "hund", "katze", "apfel"],
            propns=["john", "maria", "sue", "bob"],
            prons=["er", "sie", "sie", "es"],
            adjs=["gross", "klein", "rot", "grün", "blau", "unscharf", "rund"],
            det_def=["der"],
            det_indef=["ein"],
            comps=["dass"],
        )

    @classmethod
    def english_hf(cls):
        """English grammar parameters with head-finality."""
        return cls(
            head_initial=False,
            spec_initial=True,
            pro_drop=False,
            proper_with_det=False,
            verbs=["eats", "sees", "loves", "hears"],
            nouns=["tree", "horse", "dog", "cat", "apple"],
            propns=["john", "mary", "sue", "bob"],
            prons=["he", "she", "they", "it"],
            adjs=["big", "small", "red", "green", "blue", "fuzzy", "round"],
            det_def=["the"],
            det_indef=["a"],
            comps=["that"],
        )

    @classmethod
    def english_sf(cls):
        """English grammar parameters with specifier-finality."""
        return cls(
            head_initial=True,
            spec_initial=False,
            pro_drop=False,
            proper_with_det=False,
            verbs=["eats", "sees", "loves", "hears"],
            nouns=["tree", "horse", "dog", "cat", "apple"],
            propns=["john", "mary", "sue", "bob"],
            prons=["he", "she", "they", "it"],
            adjs=["big", "small", "red", "green", "blue", "fuzzy", "round"],
            det_def=["the"],
            det_indef=["a"],
            comps=["that"],
        )


@dataclass
class SCFGParams:
    a: CFGParams
    b: CFGParams

    name: str | None = None

    def __post_init__(self):
        # populate the name if not provided
        if self.name is None:
            # generate a random 16-character hash
            self.name = secrets.token_hex(8)

    def to_dict(self) -> dict[str, Any]:
        builder = RuleBuilder(self)
        rules = builder.build_rules()
        lexicon = builder.build_lexicon()
        grammar_str = "\n".join(rules + lexicon)
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "name": self.name,
            "grammar_str": grammar_str,
            "n_rules": len(rules),
            "n_words": len(lexicon),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SCFGParams":
        return cls(
            name=data.get("name"),
            a=CFGParams.from_dict(data["a"]),
            b=CFGParams.from_dict(data["b"]),
        )

    # Class instances
    @classmethod
    def english_english_hf(cls):
        return cls(
            a=CFGParams.english(),
            b=CFGParams.english_hf(),
        )

    @classmethod
    def english_english_sf(cls):
        return cls(
            a=CFGParams.english(),
            b=CFGParams.english_sf(),
        )


class SCFG:
    """
    A synchronous context-free grammar.
    """

    @property
    def rules(self) -> list[str]:
        return self.builder.build_rules()

    @property
    def lexicon(self) -> list[str]:
        return self.builder.build_lexicon()

    @property
    def n_rules(self) -> int:
        """Number of rules in the grammar."""
        return len(self.rules)

    @property
    def n_words(self) -> int:
        return len(self.lexicon)

    @property
    def as_cfg(self) -> str:
        return "\n".join(self.rules + self.lexicon)

    @property
    def recursive_parents(self) -> Set[str]:
        """
        Returns a set of parent symbols that can produce recursive symbols.
        """
        recursive_parents: Set[str] = set()
        for parent, rules in self.rules_dict.items():
            for prod_options in rules:
                a_prod = prod_options[0]
                if any(symbol in self.recursive_symbols for symbol in a_prod):
                    recursive_parents.add(parent)
        return recursive_parents

    def __init__(self, params: SCFGParams):
        self.params = params
        self.builder = RuleBuilder(params)
        self.start_symbol: str = "S"
        self.rules_dict: Dict[str, list[Rule]] = self._parse_rules()
        self.recursive_symbols: Set[str] = {"CP_embed"}

    def _parse_rules(self) -> Dict[str, list[Rule]]:
        """
        Parses a string representation of an SCFG into a dictionary format.
        Example output: {'TBAR': [(('T', 'VP'), ('VP', 'T'))]}
        """
        rules: Dict[str, list[Rule]] = {}
        for line in self.as_cfg.strip().split("\n"):
            line: str = line.strip()
            if not line or "->" not in line:
                continue

            lhs, rhs_str = map(str.strip, line.split("->", 1))

            # Extract a and b productions from the < > bracketed part.
            match = re.search(r"<(.*),\s*(.*)>", rhs_str)
            if not match:
                continue
            a_rhs_str, b_rhs_str = match.groups()

            # Split into symbols (non-terminals or 'terminals').
            a_symbols = tuple(a_rhs_str.strip().split())
            b_symbols = tuple(b_rhs_str.strip().split())

            if lhs not in rules:
                rules[lhs] = []
            rules[lhs].append((a_symbols, b_symbols))
        return rules

    def sample(
        self, min_depth: int = 0, max_depth: int = 1, rng: random.Random | None = None
    ) -> Dict[str, Union[str, int]]:
        """
        Samples a synchronized pair of strings from the grammar.

        Args:
            min_depth: The minimum nesting depth for recursive rules.
            max_depth: The maximum nesting depth for recursive rules.
            rng: An optional random number generator for deterministic sampling.

        Returns:
            A dictionary with full and phonetic (null-filtered) derivations,
            parse trees, and maximum depth reached during sampling.
        """
        if rng is None:
            rng = random.Random()

        assert min_depth >= 0, "Minimum depth must be non-negative."
        assert min_depth <= max_depth, "Minimum depth cannot exceed maximum depth."

        result: dict[str, str] = self._sample_recursive(
            self.start_symbol,
            rng,
            current_depth=0,
            max_depth=max_depth,
            min_depth=min_depth,
        )

        # Clean up whitespace in all generated strings before returning
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
    ) -> Dict[str, Union[str, int]]:
        """
        Recursively expands a symbol, returning full and phonetic derivations,
        and parse trees.

        Args:
            symbol: The non-terminal symbol to expand.
            rng: The random number generator.
            current_depth: The current recursion depth.
            min_depth: The minimum required recursion depth.
            max_depth: The maximum allowed recursion depth.

        Returns:
            A dictionary with keys: 'left_full', 'left_phonetic', 'right_full',
            'right_phonetic', 'left_tree', 'right_tree', 'depth'.
        """
        # Base case: The symbol is a terminal.
        if symbol not in self.rules_dict:
            clean_symbol: str = symbol.strip("'")
            full_string: str = clean_symbol
            # Phonetic string is empty if it's a null symbol.
            phonetic_string = "" if clean_symbol.startswith("∅") else full_string
            # For a terminal, left and right derivations are identical.
            return {
                "left_full": full_string,
                "left_phonetic": phonetic_string,
                "right_full": full_string,
                "right_phonetic": phonetic_string,
                "left_tree": full_string,
                "right_tree": full_string,
                "depth": current_depth,
            }

        # Recursive step: The symbol is a non-terminal.
        possible_rules: list[Rule] = self.rules_dict[symbol]

        if current_depth < min_depth:
            # Filter out rules that _could_ be recursive but aren't.
            # This involves looking at self.recursive_symbols, finding their
            # parents, and ensuring that we only sample rules result in these
            # recursive symbols from those parents. Eg, given rules like
            #   OBJ_PHRASE -> <DP, DP>
            #   OBJ_PHRASE -> <CP_embed, CP_embed>
            # since CP_embed is recursive, if the current symbol is `OBJ_PHRASE`,
            # we filter out the <DP, DP> rule from possible_rules.

            # print(self.rules_dict.items())

            if symbol in self.recursive_parents:
                possible_rules = [
                    rule
                    for rule in possible_rules
                    if any(s in self.recursive_symbols for s in rule[0])
                ]

        if current_depth >= max_depth:
            possible_rules = [
                rule
                for rule in possible_rules
                if not any(s in self.recursive_symbols for s in rule[0])
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
                }  # Cannot expand further

        chosen_left_prod, chosen_right_prod = rng.choice(possible_rules)
        sub_derivations: Dict[str, Dict[str, str]] = {}
        unique_non_terminals: list[str] = []
        seen: set[str] = set()
        for s in chosen_left_prod + chosen_right_prod:  # keeps lhs/rhs order
            if s in self.rules_dict and s not in seen:
                unique_non_terminals.append(s)
                seen.add(s)

        for s in unique_non_terminals:
            new_depth = current_depth + (1 if s in self.recursive_symbols else 0)
            sub_derivations[s] = self._sample_recursive(
                symbol=s,
                rng=rng,
                current_depth=new_depth,
                max_depth=max_depth,
                min_depth=min_depth,
            )

        # Track the maximum depth reached across all sub-derivations
        max_depth_reached = current_depth
        for derivation in sub_derivations.values():
            max_depth_reached = max(max_depth_reached, derivation["depth"])

        # Assemble the four component strings and two trees.
        left_full, left_phon = [], []
        right_full, right_phon = [], []
        left_tree_parts, right_tree_parts = [], []

        # Assemble left-side strings and tree.
        for s_left in chosen_left_prod:
            if s_left in sub_derivations:  # Non-terminal
                derivation = sub_derivations[s_left]
                left_full.append(derivation["left_full"])
                left_phon.append(derivation["left_phonetic"])
                left_tree_parts.append(derivation["left_tree"])
            else:  # Terminal
                clean_symbol = s_left.strip("'")
                left_full.append(clean_symbol)
                if not clean_symbol.startswith("∅"):
                    left_phon.append(clean_symbol)
                left_tree_parts.append(clean_symbol)

        # Assemble right-side strings and tree.
        for s_right in chosen_right_prod:
            if s_right in sub_derivations:  # Non-terminal
                derivation = sub_derivations[s_right]
                right_full.append(derivation["right_full"])
                right_phon.append(derivation["right_phonetic"])
                right_tree_parts.append(derivation["right_tree"])
            else:  # Terminal
                clean_symbol = s_right.strip("'")
                right_full.append(clean_symbol)
                if not clean_symbol.startswith("∅"):
                    right_phon.append(clean_symbol)
                right_tree_parts.append(clean_symbol)

        left_tree = f"({symbol} {' '.join(left_tree_parts)})"
        right_tree = f"({symbol} {' '.join(right_tree_parts)})"

        return {
            "left_full": " ".join(left_full),
            "left_phonetic": " ".join(left_phon),
            "right_full": " ".join(right_full),
            "right_phonetic": " ".join(right_phon),
            "left_tree": left_tree,
            "right_tree": right_tree,
            "depth": max_depth_reached,
        }


class RuleBuilder:
    """
    Build CFG rules from a set of parameters.
    """

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
        phrase: str = f"{head}P"
        xbar: str = f"{head}BAR"

        if head_initial_b is not None and spec_initial_b is not None:
            lspec: str = f"{spec} {xbar}" if spec_initial else f"{xbar} {spec}"
            rspec: str = f"{xbar} {spec}" if spec_initial_b else f"{spec} {xbar}"
            rules.append(f"{phrase} -> <{lspec}, {rspec}>")

            lhead: str = f"{head} {comp}" if head_initial else f"{comp} {head}"
            rhead: str = f"{comp} {head}" if head_initial_b else f"{head} {comp}"
            rules.append(f"{xbar} -> <{lhead}, {rhead}>")
        else:  # Monolingual rules
            if spec_initial:
                rules.append(f"{phrase} -> {spec} {xbar}")
            else:
                rules.append(f"{phrase} -> {xbar} {spec}")

            if head_initial:
                rules.append(f"{xbar} -> {head} {comp}")
            else:
                rules.append(f"{xbar} -> {comp} {head}")

        return rules

    def _lex(self, pos: str, words: list[str] | Tuple[list[str], ...]) -> list[str]:
        if isinstance(words, tuple):
            awords, bwords = words
            return [f"{pos} -> <'{aw}', '{bw}'>" for aw, bw in zip(awords, bwords)]
        else:
            return [f"{pos} -> '{word}'" for word in words]

    @property
    def is_sync(self) -> bool:
        return isinstance(self.params, SCFGParams)

    def __init__(self, params: CFGParams | SCFGParams):
        self.params = params

    def emit(self, lhs: str, rhs: str | Tuple[str, ...]) -> str | None:
        if isinstance(rhs, str):
            return f"{lhs} -> {rhs}"
        elif isinstance(rhs, tuple):
            assert len(rhs) == 2, "Right-hand side must be a tuple of length 2."
            return f"{lhs} -> <{rhs[0]}, {rhs[1]}>"

    def build_rules(self):
        rules: list[str] = []

        # S layer
        if self.is_sync:
            rules.append(self.emit("S", ("CP_matrix", "CP_matrix")))
            rules.append(self.emit("CP_matrix", ("CNULL TP", "CNULL TP")))
            rules.append(self.emit("CP_embed", ("C TP", "C TP")))
        else:
            rules.append(self.emit("S", "CP_matrix"))
            rules.append(self.emit("CP_matrix", "CNULL TP"))
            rules.append(self.emit("CP_embed", "C TP"))

        # TP layer
        if self.is_sync:
            rules += self._shell_rules(
                head="T",
                spec="NP_SUBJ",
                comp="VP",
                head_initial=self.params.a.head_initial,
                spec_initial=self.params.a.spec_initial,
                head_initial_b=self.params.b.head_initial,
                spec_initial_b=self.params.b.spec_initial,
            )
        else:
            rules += self._shell_rules(
                head="T",
                spec="NP_SUBJ",
                comp="VP",
                head_initial=self.params.head_initial,
                spec_initial=self.params.spec_initial,
            )

        # Subject layer
        if self.is_sync:
            if self.params.a.pro_drop:
                rules.append(self.emit("NP_SUBJ", ("PRO", "PRO")))
            rules.append(self.emit("NP_SUBJ", ("PRON", "PRON")))
            if not self.params.a.proper_with_det:
                rules.append(self.emit("NP_SUBJ", ("PROPN", "PROPN")))
            rules.append(self.emit("NP_SUBJ", ("DP", "DP")))
        else:
            if self.params.pro_drop:
                rules.append(self.emit("NP_SUBJ", "PRO"))
            rules.append(self.emit("NP_SUBJ", "PRON"))
            if not self.params.proper_with_det:
                rules.append(self.emit("NP_SUBJ", "PROPN"))
            rules.append(self.emit("NP_SUBJ", "DP"))

        # VP shell
        if self.is_sync:
            # Use shell_rules for VP shell
            rules += self._shell_rules(
                head="V",
                spec="",  # No specifier in VP shell
                comp="OBJ_PHRASE",
                head_initial=self.params.a.head_initial,
                spec_initial=self.params.a.spec_initial,
                head_initial_b=self.params.b.head_initial,
                spec_initial_b=self.params.b.spec_initial,
            )
            # OBJ_PHRASE alternatives
            rules.append(self.emit("OBJ_PHRASE", ("DP", "DP")))
            rules.append(self.emit("OBJ_PHRASE", ("CP_embed", "CP_embed")))
        else:
            rules += self._shell_rules(
                head="V",
                spec="",  # No specifier in VP shell
                comp="OBJ_PHRASE",
                head_initial=self.params.head_initial,
                spec_initial=self.params.spec_initial,
            )
            rules.append(self.emit("OBJ_PHRASE", "DP"))
            rules.append(self.emit("OBJ_PHRASE", "CP_embed"))

        # DP layer
        if self.is_sync:
            # Definite DP
            rules += self._shell_rules(
                head="DET",
                spec="",  # No specifier in DP shell
                comp="NP",
                head_initial=self.params.a.head_initial,
                spec_initial=self.params.a.spec_initial,
                head_initial_b=self.params.b.head_initial,
                spec_initial_b=self.params.b.spec_initial,
            )
            rules.append(self.emit("DP", ("DP_def", "DP_def")))
            rules.append(self.emit("DP", ("DP_indef", "DP_indef")))
            rules.append(self.emit("DP_def", ("DET_def NP", "DET_def NP")))
            rules.append(self.emit("DP_indef", ("DET_indef NP", "DET_indef NP")))
            left_pwd = getattr(self.params.a, "proper_with_det", False)
            right_pwd = getattr(self.params.b, "proper_with_det", False)
            if left_pwd and right_pwd:
                rules.append(self.emit("DP_def", ("DET_def PROPN", "DET_def PROPN")))
            elif not left_pwd and not right_pwd:
                rules.append(self.emit("DP_def", ("PROPN", "PROPN")))
        else:
            rules.append(self.emit("DP", "DET_defP"))
            rules.append(self.emit("DP", "DET_indefP"))
            rules += self._shell_rules(
                head="DET_def",
                spec="",  # No specifier in DP shell
                comp="NP",
                head_initial=self.params.head_initial,
                spec_initial=self.params.spec_initial,
            )
            rules += self._shell_rules(
                head="DET_indef",
                spec="",  # No specifier in DP shell
                comp="NP",
                head_initial=self.params.head_initial,
                spec_initial=self.params.spec_initial,
            )
            if getattr(self.params, "proper_with_det", False):
                rules.append(self.emit("DP_def", "DET_def PROPN"))
            else:
                rules.append(self.emit("DP_def", "PROPN"))

        # NP layer
        if self.is_sync:
            rules.append(self.emit("NP", ("N_HEAD", "N_HEAD")))
            rules.append(self.emit("NP", ("AdjP NP", "AdjP NP")))
            rules.append(self.emit("NP_COMMON", ("N", "N")))
            rules.append(self.emit("NP_COMMON", ("AdjP NP_COMMON", "AdjP NP_COMMON")))
            rules.append(self.emit("AdjP", ("ADJ", "ADJ")))
        else:
            rules.append(self.emit("NP", "N_HEAD"))
            rules.append(self.emit("NP", "AdjP NP"))
            rules.append(self.emit("NP_COMMON", "N"))
            rules.append(self.emit("NP_COMMON", "AdjP NP_COMMON"))
            rules.append(self.emit("AdjP", "ADJ"))

        # N_HEAD rules
        if self.is_sync:
            left_pwd = getattr(self.params.a, "proper_with_det", False)
            right_pwd = getattr(self.params.b, "proper_with_det", False)
            if not left_pwd and not right_pwd:
                for cat in ("N", "PROPN"):
                    rules.append(self.emit("N_HEAD", (cat, cat)))
            else:
                left_alts = ("PROPN",) if left_pwd else ("N", "PROPN")
                right_alts = ("PROPN",) if right_pwd else ("N", "PROPN")
                for ls in left_alts:
                    for rs in right_alts:
                        rules.append(self.emit("N_HEAD", (ls, rs)))
        else:
            rules.append(self.emit("N_HEAD", "PROPN"))
            if getattr(self.params, "proper_with_det", True):
                rules.append(self.emit("N_HEAD", "N"))

        return rules

    def build_lexicon(self):
        rules = []
        if self.is_sync:
            rules += self._lex(
                "DET_def",
                (self.params.a.det_def_lex, self.params.b.det_def_lex),
            )
            rules += self._lex(
                "DET_indef",
                (self.params.a.det_indef_lex, self.params.b.det_indef_lex),
            )
            rules += self._lex(
                "T",
                (self.params.a.tense_lex, self.params.b.tense_lex),
            )
            rules += self._lex(
                "ASP",
                (self.params.a.asp_lex, self.params.b.asp_lex),
            )
            rules += self._lex(
                "V",
                (self.params.a.verb_lex, self.params.b.verb_lex),
            )
            rules += self._lex(
                "N",
                (self.params.a.noun_lex, self.params.b.noun_lex),
            )
            rules += self._lex(
                "PROPN",
                (self.params.a.propn_lex, self.params.b.propn_lex),
            )
            rules += self._lex(
                "PRON",
                (self.params.a.pron_lex, self.params.b.pron_lex),
            )
            rules += self._lex(
                "ADJ",
                (self.params.a.adj_lex, self.params.b.adj_lex),
            )
            rules += self._lex(
                "C",
                (self.params.a.comp_lex, self.params.b.comp_lex),
            )
            rules.append("CNULL -> <'∅', '∅'>")
            if self.params.a.pro_drop or self.params.b.pro_drop:
                rules.append("PRO -> <'∅', '∅'>")
        else:
            # Monolingual lexicon
            rules += self._lex("DET_def", self.params.det_def_lex)
            rules += self._lex("DET_indef", self.params.det_indef_lex)
            rules += self._lex("T", [f"{x}" for x in self.params.tense_lex])
            rules += self._lex("ASP", [f"{x}" for x in self.params.asp_lex])
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
