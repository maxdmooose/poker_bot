# evolution/species_registry.py

from typing import Callable, Dict, Any

from bots.heuristic.heuristic_mutate import (
    create_initial_heuristic_bot,
    mutate_heuristic_bot,
)
from bots.lookup.lookup_mutate import (
    create_initial_lookup_bot,
    mutate_lookup_bot,
)
from bots.blueprint.blueprint_mutate import (
    create_initial_blueprint_bot,
    mutate_blueprint_bot,
)

SpeciesInitFn = Callable[[dict], dict]
SpeciesMutateFn = Callable[[dict, dict], dict]

SPECIES_REGISTRY: Dict[str, Dict[str, Any]] = {
    "heuristic": {
        "init": create_initial_heuristic_bot,
        "mutate": mutate_heuristic_bot,
    },
    "lookup": {
        "init": create_initial_lookup_bot,
        "mutate": mutate_lookup_bot,
    },
    "blueprint": {
        "init": create_initial_blueprint_bot,
        "mutate": mutate_blueprint_bot,
    },
}
