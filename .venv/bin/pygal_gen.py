#!/home/shaotingf/LMCache/.venv/bin/python3
# -*- coding: utf-8 -*-
# This file is part of pygal
#
# A python svg graph plotting library
# Copyright © 2012-2016 Kozea
#
# This library is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pygal. If not, see <http://www.gnu.org/licenses/>.
import argparse

import pygal

parser = argparse.ArgumentParser(
    description='Generate pygal chart in command line',
    prog='pygal_gen')

parser.add_argument('-t', '--type', dest='type', default='Line',
                    choices=map(lambda x: x.__name__, pygal.CHARTS),
                    help='Kind of chart to generate')

parser.add_argument('-o', '--output', dest='filename', default='pygal_out.svg',
                    help='Filename to write the svg to')

parser.add_argument('-s', '--serie', dest='series', nargs='+', action='append',
                    help='Add a serie in the form (title val1 val2...)')

parser.add_argument('--version', action='version',
                    version='pygal %s' % pygal.__version__)

for key in pygal.config.CONFIG_ITEMS:
    opt_name = key.name
    val = key.value
    opts = {}
    if key.type == list:
        opts['type'] = key.subtype
        opts['nargs'] = '+'
    else:
        opts['type'] = key.type

    if opts['type'] == bool:
        del opts['type']
        opts['action'] = 'store_true' if not val else 'store_false'
        if val:
            opt_name = 'no-' + opt_name
    if key.name == 'interpolate':
        opts['choices'] = list(pygal.interpolate.INTERPOLATIONS.keys())
    parser.add_argument(
        '--%s' % opt_name, dest=key.name, default=val, **opts)

config = parser.parse_args()

chart = getattr(pygal, config.type)(**vars(config))

for serie in config.series:
    chart.add(serie[0], map(float, serie[1:]))

chart.render_to_file(config.filename)
