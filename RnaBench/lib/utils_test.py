import pytest

from RnaBench.lib.utils import (
is_valid,
has_nc,
)

@pytest.mark.parametrize(("structure"),
                         [
                         ('...((((...))))...'),
                         ('...([[[(((...))]]]))...'),
                         pytest.param('((...)', marks=pytest.mark.xfail(reason='Unbalanced brackets')),
                         pytest.param('([...)...]...(...))', marks=pytest.mark.xfail(reason='Unbalanced brackets')),
                         ]
                        )
def test_is_valid(structure):
    is_valid(structure)


def test_has_nc():
    pass