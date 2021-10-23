# Copyright (c) Open-MMLab. All rights reserved.

__version__ = '0.1.0'
short_version = __version__


def parse_version_info(version_str):
    version_info_ = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info_.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info_.append(int(patch_version[0]))
            version_info_.append(f'rc{patch_version[1]}')
    return tuple(version_info_)


version_info = parse_version_info(__version__)
