"""Conservative exact-type compartment map on verified FlyWire annotations."""
import json
from pathlib import Path

import numpy as np

from flying.data.mushroom_body import load_roles


def resolve_compartments(directory, ids, registry='configs/compartments.json'):
    roles, annotations = load_roles(directory, ids)
    specification = json.loads(Path(registry).read_text())
    if specification['version'] != 1:
        raise ValueError('Unknown compartment registry version')
    mapping, assigned = {}, set()
    for compartment, entry in specification['compartments'].items():
        mapping[compartment] = {}
        for role, field in [('DAN', 'dan_types'), ('MBON', 'mbon_types')]:
            mask = annotations.cell_type.isin(entry[field]).to_numpy()
            indices = np.flatnonzero(mask)
            if not len(indices) or np.any(roles[indices] != role):
                raise ValueError('Missing compartment type or inconsistent role')
            if not annotations.iloc[indices].hemibrain_type.isin(entry[field]).all():
                raise ValueError('Conflicting type annotation; do not guess')
            if not (annotations.iloc[indices].side == 'left').all():
                raise ValueError('This registry application expects left circuit cells')
            if assigned.intersection(indices):
                raise ValueError('Ambiguous compartment assignment')
            assigned.update(indices)
            mapping[compartment][role] = indices
    audit = []
    for compartment, entry in mapping.items():
        for role, indices in entry.items():
            for i in indices:
                audit.append(dict(root_id=ids[i], index=int(i), role=role, compartment=compartment,
                                  cell_type=annotations.iloc[i].cell_type))
    return roles, mapping, dict(cells=audit, unmapped_modulatory_or_output_cells=int(
        sum(r in ['DAN', 'MBON'] and i not in assigned for i, r in enumerate(roles))),
        registry=specification)
