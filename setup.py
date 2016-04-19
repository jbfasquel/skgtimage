from __future__ import print_function
from setuptools import setup,find_packages
#import io
#import os
#import sys

DESCRIPTION = 'A set of python routines for core-based guidance in sequential image analysis'
with open('README.rst') as f: LONG_DESCRIPTION = f.read()

setup(
    name='skgtimage',
    version='0.1',
    url='Not yet defined',
    license='BSD License',
    author='Jean-Baptiste Fasquel',
    install_requires=['scipy>=0.12.0','networkx>=1.10','numpy>=1.8.0','matplotlib>=1.4.0'], #Warning: order of install: numpy before scipy -> numpy at the end ???
    #cmdclass={'test': PyTest}, or clean
    author_email='Jean-Baptiste.Fasquel [at] univ-angers.fr',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    #include_package_data=True,
    #packages=find_packages(exclude=[".*"]), #to remove files such as .DStore (macos), avoids packages=['skgtimage','skgtimage.core',...]
    packages=find_packages(exclude=[".*","doc*"]), #to remove files such as .DStore (macos), avoids packages=['skgtimage','skgtimage.core',...]
    zip_safe=False, #if True, installation leads bundles the lib within a .egg archive,
    classifiers = [
                                'Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2.6',
                                 'Programming Language :: Python :: 2.7',
        ],
)


