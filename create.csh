#!/bin/csh

dada_db -d -k dada
dada_db -d -k d0d0

#nsamps: 256; nchans: 336; nbeams: 36
dada_db -k dada -a 32768 -b 12386304 -n 8
dada_db -k d0d0 -a 32768 -b 12386304 -n 8
