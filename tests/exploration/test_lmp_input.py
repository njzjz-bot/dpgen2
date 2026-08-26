import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import scipy.constants as pc

from dpgen2.exploration.task.lmp.lmp_input import (
    make_lmp_input,
)


class TestLmpInput(unittest.TestCase):
    @patch(
        "dpgen2.exploration.task.lmp.lmp_input._sample_sphere",
        return_value=np.array([1.0, 0.0, 0.0]),
    )
    @patch("dpgen2.exploration.task.lmp.lmp_input.dpdata.System")
    def test_pka_mass_uses_zero_based_atom_type(self, mocked_system, _mocked_sphere):
        """Use the first mass-map entry for dpdata atom type zero."""
        mocked_system.return_value.data = {"atom_types": np.array([0])}

        lmp_input = make_lmp_input(
            "conf.lmp",
            "nve",
            ["model.pb"],
            nsteps=1,
            dt=0.001,
            neidelay=None,
            trj_freq=1,
            mass_map=[10.0, 20.0],
            temp=300.0,
            pka_e=1.0,
        )

        expected_velocity = np.sqrt(
            pc.electron_volt
            / (0.5 * 10.0 * 1e-3 / pc.Avogadro * (pc.angstrom / pc.pico) ** 2)
        )
        self.assertIn(
            f"velocity        first set {expected_velocity:f} 0.000000 0.000000",
            lmp_input,
        )


if __name__ == "__main__":
    unittest.main()
