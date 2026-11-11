# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import Mock

from lmcache.v1.multiprocess.recurrent_reservations import (
    HistoryReservation,
    RecurrentReservations,
)

# First Party
from lmcache.v1.transport.recurrent_history import HistoryResponse


def test_missing_token_is_fenced_only_for_this_server_generation() -> None:
    reservations = RecurrentReservations()
    entry = Mock(spec=HistoryReservation)
    entry.result.return_value = HistoryResponse(settled=True)
    retired = reservations.add(entry)
    reservations.acknowledge(retired)

    assert reservations.query(retired).settled
    assert not reservations.query(retired).successful
    assert not RecurrentReservations().query(retired).settled
