from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PERSON_VALUES: tuple[str, ...] = ("1", "2", "3")
NUMBER_VALUES: tuple[str, ...] = ("sg", "pl")
GENDER_VALUES: tuple[str, ...] = ("masc", "fem")


@dataclass(frozen=True)
class FeatureBundle:
    person: str | None = None
    number: str | None = None
    gender: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FeatureBundle":
        if not data:
            return cls()
        return cls(
            person=data.get("person"),
            number=data.get("number"),
            gender=data.get("gender"),
        )

    def key(self, axes: tuple[str, ...] = ("person", "number")) -> str:
        parts: list[str] = []
        for axis in axes:
            value = getattr(self, axis)
            if value is not None:
                parts.append(f"{axis}={value}")
        return "|".join(parts) if parts else "default"


@dataclass(frozen=True)
class AgreementConfig:
    enabled: bool = False
    active_axes: tuple[str, ...] = ("number", "person")
    strategy: str = "suffix"
    verb_agreement: bool = True
    noun_number_marking: bool = True
    pronouns_are_featured: bool = True
    latent_gender: bool = False
    realize_gender: bool = False
    gender_values: tuple[str, ...] = GENDER_VALUES

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active_axes": list(self.active_axes),
            "strategy": self.strategy,
            "verb_agreement": self.verb_agreement,
            "noun_number_marking": self.noun_number_marking,
            "pronouns_are_featured": self.pronouns_are_featured,
            "latent_gender": self.latent_gender,
            "realize_gender": self.realize_gender,
            "gender_values": list(self.gender_values),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgreementConfig":
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled", False),
            active_axes=tuple(data.get("active_axes", ("number", "person"))),
            strategy=data.get("strategy", "suffix"),
            verb_agreement=data.get("verb_agreement", True),
            noun_number_marking=data.get("noun_number_marking", True),
            pronouns_are_featured=data.get("pronouns_are_featured", True),
            latent_gender=data.get("latent_gender", False),
            realize_gender=data.get("realize_gender", False),
            gender_values=tuple(data.get("gender_values", GENDER_VALUES)),
        )


class FeatureUnifier:
    @staticmethod
    def unify(left: FeatureBundle, right: FeatureBundle) -> FeatureBundle | None:
        merged: dict[str, str | None] = {}
        for axis in ("person", "number", "gender"):
            left_value = getattr(left, axis)
            right_value = getattr(right, axis)
            if left_value is not None and right_value is not None and left_value != right_value:
                return None
            merged[axis] = left_value if left_value is not None else right_value
        return FeatureBundle(**merged)


def feature_inventory(
    axes: tuple[str, ...] = ("person", "number"),
    gender_values: tuple[str, ...] = GENDER_VALUES,
) -> list[FeatureBundle]:
    bundles: list[FeatureBundle] = []
    people = PERSON_VALUES if "person" in axes else (None,)
    numbers = NUMBER_VALUES if "number" in axes else (None,)
    genders = gender_values if "gender" in axes else (None,)
    for person in people:
        for number in numbers:
            for gender in genders:
                bundles.append(
                    FeatureBundle(person=person, number=number, gender=gender)
                )
    return bundles


def build_suffix_inventory(
    forms: list[str],
    axes: tuple[str, ...],
    gender_values: tuple[str, ...] = GENDER_VALUES,
) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for bundle, form in zip(feature_inventory(axes, gender_values=gender_values), forms):
        inventory[bundle.key(axes)] = form
    return inventory


def serialize_bundles(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in items:
        converted = dict(item)
        if "features" in converted and isinstance(converted["features"], FeatureBundle):
            converted["features"] = converted["features"].to_dict()
        if "forms" in converted:
            converted["forms"] = dict(converted["forms"])
        serialized.append(converted)
    return serialized


def deserialize_bundles(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not items:
        return []
    deserialized: list[dict[str, Any]] = []
    for item in items:
        converted = dict(item)
        if "features" in converted:
            converted["features"] = FeatureBundle.from_dict(converted["features"])
        if "forms" in converted:
            converted["forms"] = dict(converted["forms"])
        deserialized.append(converted)
    return deserialized
